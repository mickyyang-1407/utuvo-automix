"""Local web UI for the automix engine.

A single-file server, stdlib only: drop stems in a browser, pick a style, get a
mix.  It is deliberately *local* (127.0.0.1) and deliberately thin -- every
decision still comes from :mod:`automix.pipeline`, so the UI cannot drift away
from the engine that the tests cover.

Upload protocol
---------------
No multipart parsing: ``cgi`` was removed in Python 3.13, and hand-writing a
multipart parser to move bytes we already have is a bad trade.  Instead each
file is POSTed as a raw body with its name in ``X-Filename``, which is a few
lines to read and impossible to get subtly wrong.

    POST /api/session                  -> {"id": ...}
    POST /api/session/<id>/stem        (X-Filename: kick.wav)
    POST /api/session/<id>/reference   (X-Filename: ref.wav)
    POST /api/session/<id>/run         {"style": ..., "intensity": 1.0, "balance_preserve": 0.35, "blend": 0.7}
    GET  /api/styles
    GET  /api/session/<id>/audio/mix|raw|reference
    GET  /api/session/<id>/plan | /report
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import multiprocessing
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import analyze, dsp, plan as planner, pipeline, reference as refmod, render, source_balance as balancemod

def _resource_root() -> str:
    """Where the web assets and presets live.

    In a PyInstaller bundle everything is unpacked under ``sys._MEIPASS``; in a
    normal checkout it is the lab directory two levels above this file.  Getting
    this wrong makes the packaged app start and then 404 its own UI.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(sys._MEIPASS)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


HERE = os.path.dirname(os.path.abspath(__file__))
MVP = os.path.dirname(HERE)
LAB = os.path.dirname(MVP)
if getattr(sys, "frozen", False):
    WEB_DIR = os.path.join(_resource_root(), "web")
    PRESET_DIR = os.path.join(_resource_root(), "presets")
else:
    WEB_DIR = os.path.join(MVP, "web")
    PRESET_DIR = os.path.join(LAB, "presets")

MAX_UPLOAD = 256 * 1024 * 1024  # per file, generous for a 24-bit stem set

#: a real session is rarely more than a few dozen stems; a cap keeps a runaway
#: client from filling the disk, and the error is explicit rather than "disk full"
MAX_STEMS = 64
#: total audio on disk for one session (~8 GB: 12 stems x 10 minutes of 24-bit)
MAX_SESSION_BYTES = 8 * 1024 * 1024 * 1024

#: a 10-minute, 30-stem session should still finish well inside this
RENDER_TIMEOUT_S = 900.0
ALLOWED_EXT = analyze.AUDIO_EXTS

_sessions: dict[str, dict] = {}
_lock = threading.Lock()
#: Only one render at a time.  Each render is a child process with a peak of
#: several GB on a three-minute song; two at once is how a laptop with 8 GB
#: starts swapping instead of mixing.
_render_slot = threading.Semaphore(1)


#: sessions are disposable: after this long they are removed on the next start
SESSION_TTL_H = 12.0
#: and never keep more than this many around
MAX_SESSIONS = 40


def _prune(root: str, ttl_h: float = SESSION_TTL_H, keep: int = MAX_SESSIONS) -> int:
    """Delete stale session folders.  Returns how many went away.

    A three-minute project leaves ~400 MB of WAV behind, and a UI that is left
    open overnight creates one per render.  Without this the temp directory
    quietly becomes the biggest thing on the disk.
    """
    if not os.path.isdir(root):
        return 0
    now = time.time()
    entries = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            entries.append((os.path.getmtime(path), path))
        except OSError:
            continue
    entries.sort(reverse=True)  # newest first

    removed = 0
    for i, (mtime, path) in enumerate(entries):
        too_old = (now - mtime) > ttl_h * 3600.0
        too_many = i >= keep
        if too_old or too_many:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


#: how many columns the waveform drawing gets
PEAK_BUCKETS = 600


def _peaks(path: str, sess: Session) -> dict:
    """A tiny peak envelope for the UI's waveform, plus section markers."""
    audio, sr = analyze.load_stem_file(path, sess.decode_policy)
    mono = np.max(np.abs(audio), axis=1)
    n = mono.shape[0]
    buckets = min(PEAK_BUCKETS, n)
    if buckets <= 0:
        return {"peaks": [], "sections": []}
    edges = np.linspace(0, n, buckets + 1).astype(int)
    peaks = [round(float(mono[edges[i] : edges[i + 1]].max() or 0.0), 4) for i in range(buckets)]
    sections = [
        {"start_s": sec["start_s"], "end_s": sec["end_s"], "label": sec["label"]}
        for sec in ((sess.result or {}).get("sections") or [])
    ]
    return {"peaks": peaks, "duration_s": round(n / sr, 2), "sections": sections}


def _finite_number(value, lo=None, hi=None):
    """Return a float inside the range, or None when the value is unusable.

    Anything that is not a number (or is not finite, or falls outside the
    range) is treated as "not set" rather than repaired: a saved value we
    cannot vouch for must not become a control the user never chose.
    """
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            return None
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    else:
        try:
            v = float(value)
        except (OverflowError, ValueError):
            # a number too large to be a float is not a number we can use
            return None
    if not math.isfinite(v):
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def _saved_number(value, lo=None, hi=None):
    """A number read back from a saved manifest: no type coercion.

    A saved value that needs repairing is a value we cannot vouch for, so the
    saved path is strict about types.  The request path stays lenient: a
    numeric string there was accepted before and still is.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        # a number too large to be a float is not a number we can use
        return None
    if not math.isfinite(v):
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def requested_controls(payload: dict) -> dict:
    """The controls the user asked for, normalised for storage.

    Only requested values are kept.  An empty target stays empty and an unset
    balance-preserve stays unset: the numbers the engine fitted or resolved
    are results, and saving those would claim the user chose them.
    """
    style = payload.get("style")
    profile = payload.get("source_balance_profile")
    return {
        "style": style if isinstance(style, str) and style else None,
        "intensity": _finite_number(payload.get("intensity"), 0.0, 1.0),
        "source_balance_profile": profile if isinstance(profile, str) and profile else None,
        "balance_preserve": _finite_number(payload.get("balance_preserve"), 0.0, 1.0),
        "blend": _finite_number(payload.get("blend"), 0.0, 1.0),
        "target_lufs": _finite_number(payload.get("target_lufs")),
        "export_stems": payload.get("export_stems") is True,
        "reverb_amount": _finite_number(payload.get("reverb_amount"), 0.0, 1.0),
    }


#: the loudness targets the planner accepts (plan.py: "target_lufs {t} is
#: outside the supported -24..-6 range").  A saved target outside this range
#: could never be rendered, so it must not be handed back to the UI.
TARGET_LUFS_MIN = -24.0
TARGET_LUFS_MAX = -6.0


def clean_controls(raw) -> dict | None:
    """Return a usable controls snapshot, or None when there is none.

    A damaged manifest must not stop a session from opening, so every field is
    checked and a value we cannot vouch for drops the whole snapshot: the UI
    then reports the session as one whose choices cannot be restored, which is
    true, instead of restoring half of it and calling that a full restore.

    The checks follow the sources of truth: the bundled presets for the style,
    the source-balance profiles the engine accepts, and the planner's own
    target range.
    """
    if not isinstance(raw, dict):
        return None
    style = raw.get("style")
    if not isinstance(style, str) or not style:
        return None
    if not any(s["id"] == style for s in _styles()):
        return None
    profile = raw.get("source_balance_profile")
    if profile is not None and profile not in balancemod.PROFILES:
        return None

    def number(key, lo, hi):
        """(value, usable): a missing value is "not set", a bad one is fatal."""
        raw_value = raw.get(key)
        if raw_value is None:
            return None, True
        value = _saved_number(raw_value, lo, hi)
        return value, value is not None

    intensity, ok_i = number("intensity", 0.0, 1.0)
    balance, ok_b = number("balance_preserve", 0.0, 1.0)
    blend, ok_l = number("blend", 0.0, 1.0)
    reverb, ok_r = number("reverb_amount", 0.0, 1.0)
    target, ok_t = number("target_lufs", TARGET_LUFS_MIN, TARGET_LUFS_MAX)
    if not all((ok_i, ok_b, ok_l, ok_r, ok_t)):
        return None
    export = raw.get("export_stems")
    if export is None:
        export = False
    elif not isinstance(export, bool):
        return None
    return {
        "style": style,
        "intensity": intensity,
        "source_balance_profile": profile,
        "balance_preserve": balance,
        "blend": blend,
        "target_lufs": target,
        "export_stems": export,
        "reverb_amount": reverb,
    }


def _clean_capture_groups(raw) -> tuple[list[dict], str | None]:
    """Keep only the group declarations we can act on.

    A saved state we cannot vouch for must not become a grouping we cannot
    vouch for: an unusable declaration is dropped and reported, never guessed
    and never repaired into a different grouping.

    A stereo pair keeps its mode, its ordered left/right members and its
    shared trim.  Dropping the mode would restore a pair as a same-source
    group -- a different declaration wearing the same name -- and dropping
    the trim would silently change the pair's level, so a pair we cannot
    read back whole is dropped instead.
    """
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "the saved capture groups were not a list, so none were restored"
    groups: list[dict] = []
    dropped = 0
    bad_pan = 0
    bad_pair = 0
    for item in raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        gid = item.get("id")
        members = item.get("members")
        if not isinstance(gid, str) or not gid.strip() or not isinstance(members, list):
            dropped += 1
            continue
        if any(not isinstance(m, str) or not m.strip() for m in members):
            dropped += 1
            continue
        pan = item.get("pan", 0.0)
        pan_value = None
        if not isinstance(pan, bool) and isinstance(pan, (int, float)):
            try:
                candidate = float(pan)
            except (OverflowError, ValueError):
                candidate = None
            if candidate is not None and math.isfinite(candidate) and -1.0 <= candidate <= 1.0:
                pan_value = candidate
        if pan_value is None:
            # A pan we cannot vouch for takes its declaration down with it: it
            # is never repaired into a value nobody chose.
            dropped += 1
            bad_pan += 1
            continue
        mode = item.get("mode")
        if "mode" in item and item["mode"] != "stereo_pair":
            # A mode that is present but is not the one literal mode is a
            # declaration we cannot read back.  It is never read as "no mode":
            # that would turn an ambiguous pair intent into a collapsed
            # same-source group.
            dropped += 1
            bad_pair += 1
            continue
        if mode == "stereo_pair":
            trim = item.get("trim_db", 0.0)
            trim_value = None
            if not isinstance(trim, bool) and isinstance(trim, (int, float)):
                try:
                    candidate = float(trim)
                except (OverflowError, ValueError):
                    candidate = None
                if candidate is not None and math.isfinite(candidate) and -12.0 <= candidate <= 12.0:
                    trim_value = candidate
            if trim_value is None:
                dropped += 1
                bad_pair += 1
                continue
            if len(members) != 2 or len(set(members)) != 2:
                dropped += 1
                bad_pair += 1
                continue
            groups.append(
                {
                    "id": gid.strip(),
                    "members": list(members),
                    "pan": pan_value,
                    "mode": "stereo_pair",
                    "trim_db": trim_value,
                }
            )
            continue
        if "trim_db" in item:
            # a shared trim without the pair mode is a field nobody chose:
            # it is refused, not ignored
            dropped += 1
            bad_pair += 1
            continue
        groups.append({"id": gid.strip(), "members": list(members), "pan": pan_value})
    error = None
    if dropped:
        error = f"{dropped} saved capture group declaration(s) could not be restored"
        if bad_pan:
            error += f" ({bad_pan} had a pan that is not a number between -1 and 1)"
        if bad_pair:
            error += (
                f" ({bad_pair} had a stereo pair mode, member count or shared trim "
                "that could not be read)"
            )
    return groups, error


def _same_capture_groups(left: list, right: list) -> bool:
    """Are two group declarations the same declaration?

    The lists are compared as lists: the same groups, in the same order,
    with the same members and the same pan.  Both sides have already been
    normalized -- a member named by its file name and the same member named
    by its canonical track id are one member -- and a pan of 0 and a pan of
    0.0 are one pan.  Nothing else is inferred: no set semantics, no
    guessing that two different declarations mean the same thing.

    The mode and the shared trim are part of the declaration too: a pair that
    became a same-source group, or a pair whose shared trim moved, is a
    different declaration and makes the previous mix stale.
    """
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if a.get("id") != b.get("id"):
            return False
        if list(a.get("members") or []) != list(b.get("members") or []):
            return False
        try:
            pan_a = float(a.get("pan", 0.0))
            pan_b = float(b.get("pan", 0.0))
        except (TypeError, ValueError):
            return False
        if pan_a != pan_b:
            return False
        mode_a = a.get("mode")
        mode_b = b.get("mode")
        if mode_a != mode_b:
            return False
        if mode_a == "stereo_pair":
            try:
                trim_a = float(a.get("trim_db", 0.0))
                trim_b = float(b.get("trim_db", 0.0))
            except (TypeError, ValueError):
                return False
            if trim_a != trim_b:
                return False
    return True


#: the decoder a session keeps when its manifest does not name one.  Every
#: session saved before the field existed was imported, peaked, role-checked
#: and rendered with ``float32_v1`` (the default from R43 to R46), so that is
#: what it keeps: reopening an old project must not quietly change how its
#: mono files are decoded.  A new session uses the current default.
PRE_POLICY_SESSION_DECODE_POLICY = analyze.DECODE_POLICY_FLOAT32


def _without_source_refs(result):
    """A result saved before the styles stopped citing people still names them;
    the restored copy drops the list (display only -- the plan file is untouched)."""
    style = result.get("style") if isinstance(result, dict) else None
    intent = style.get("intent") if isinstance(style, dict) else None
    if isinstance(intent, dict) and intent.get("source_refs"):
        result = dict(result)
        result["style"] = dict(result["style"])
        result["style"]["intent"] = dict(intent, source_refs=[])
    return result


def _restored_decode_policy(value) -> str:
    """The decoder policy a saved session continues with."""
    if isinstance(value, str) and value in analyze.DECODE_POLICIES:
        return value
    return PRE_POLICY_SESSION_DECODE_POLICY


class Session:
    def __init__(self, root: str, sid: str | None = None):
        self.id = sid or uuid.uuid4().hex[:12]
        self.dir = os.path.join(root, self.id)
        self.stems_dir = os.path.join(self.dir, "stems")
        os.makedirs(self.stems_dir, exist_ok=True)
        self.reference_path: str | None = None
        self.mix_path: str | None = None
        self.raw_path: str | None = None
        #: the level-matched comparison pair, when the published result
        #: says the pair is trustworthy.  They are the *monitoring* views,
        #: not the export: /audio/mix and /download/mix stay the mix.
        self.monitor_mix_path: str | None = None
        self.monitor_raw_path: str | None = None
        self.plan: dict | None = None
        self.report: dict | None = None
        self.result: dict | None = None
        self.roles: dict[str, str] = {}  # user overrides, by filename
        # a small counter that says how many manual role changes this session
        # has accepted; it lets a client order role snapshots instead of
        # guessing from the order the answers happen to arrive in
        self.role_revision = 0
        # Transient automatic roles/quick numbers only, never decoded audio or
        # manual overrides. Concurrent role reads share the same analysis.
        self._role_analysis_cache = None
        self._role_analysis_lock = threading.Lock()
        self.trims: dict[str, float] = {}  # manual gain nudges, by track id
        self.muted: set[str] = set()  # track ids the user does not want in the mix
        self.pans: dict[str, float] = {}  # manual pan overrides, by track id
        #: declared capture groups: which tracks are the same physical source
        #: recorded by several microphones.  These are the user's drafts; they
        #: are checked against the real tracks when a mix is planned.
        self.capture_groups: list[dict] = []
        #: why a saved group state could not be restored, if it could not
        self.capture_groups_error: str | None = None
        self.reverb_amount = 1.0
        #: how this session's files are decoded, for as long as it exists: a
        #: new session takes the current default, a restored one keeps its own
        self.decode_policy = analyze.DEFAULT_DECODE_POLICY
        # the requested controls of the last successful mix; None when the
        # session has never rendered (or the saved copy was unusable)
        self.controls: dict | None = None
        self.project_name = ""
        self.draft_controls: dict | None = None
        self.proc = None  # the render child, while one is running
        self.cancel_requested = False
        self.progress: dict | None = None
        #: the session is busy while a render publishes its result: a state
        #: change must not run into the middle of a commit
        self.busy = False
        #: one state change at a time, and never during a commit
        self.lock = threading.Lock()

    #: what a session needs to survive a server restart
    MANIFEST = "session.json"

    def manifest_data(self, **overrides) -> dict:
        """The editable state, exactly as it is written to the manifest.

        Keyword overrides let the publication path write the values of the
        render it is about to publish *without* putting them into the live
        session first: a request that arrives while the commit runs must not
        be able to save them, and a rollback must not have to undo them.
        """
        data = {
            "id": self.id,
            "roles": self.roles,
            "role_revision": self.role_revision,
            "trims": self.trims,
            "muted": sorted(self.muted),
            "pans": self.pans,
            "capture_groups": self.capture_groups,
            "reverb_amount": self.reverb_amount,
            "reference": (
                os.path.relpath(self.reference_path, self.dir) if self.reference_path else None
            ),
            "has_mix": self.has_mix(),
            # the controls the last successful mix was rendered with; None for
            # a session that has not rendered one (or that predates the
            # snapshot), so the UI can say so instead of guessing
            "controls": self.controls,
            "project_name": self.project_name,
            "draft_controls": self.draft_controls,
            "decode_policy": self.decode_policy,
        }
        data.update(overrides)
        return data

    def save(self) -> None:
        """Persist the editable state so reopening the app resumes the project."""
        try:
            with open(os.path.join(self.dir, self.MANIFEST), "w", encoding="utf-8") as fh:
                json.dump(self.manifest_data(), fh, ensure_ascii=False, indent=2)
        except OSError:
            pass  # persistence is a convenience; never fail a request over it

    def write_manifest(self, path: str, **overrides) -> None:
        """Write the manifest to ``path``; raises when it cannot be written.

        The publication path uses this one: a manifest that cannot be written
        is a failed render, not a convenience.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest_data(**overrides), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, root: str, sid: str) -> "Session | None":
        path = os.path.join(root, sid, cls.MANIFEST)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        sess = cls(root, sid=sid)
        sess.roles = {str(k): str(v) for k, v in (data.get("roles") or {}).items()}
        try:
            # an old manifest has no revision; it counts as 0
            sess.role_revision = max(0, int(data.get("role_revision", 0) or 0))
        except (TypeError, ValueError):
            sess.role_revision = 0
        sess.trims = {str(k): float(v) for k, v in (data.get("trims") or {}).items()}
        sess.muted = set(data.get("muted") or [])
        sess.pans = {str(k): float(v) for k, v in (data.get("pans") or {}).items()}
        sess.capture_groups, sess.capture_groups_error = _clean_capture_groups(
            data.get("capture_groups")
        )
        try:
            sess.reverb_amount = planner._manual_reverb_amount(data.get("reverb_amount", 1.0))
        except planner.PlanError:
            sess.reverb_amount = 1.0
        # a session saved before the snapshot existed (or a damaged one) must
        # still open; the UI then reports the mix as "choices not saved"
        sess.controls = clean_controls(data.get("controls"))
        sess.project_name = str(data.get("project_name") or "")
        sess.draft_controls = clean_controls(data.get("draft_controls"))
        sess.decode_policy = _restored_decode_policy(data.get("decode_policy"))
        ref = data.get("reference")
        if ref:
            candidate = os.path.join(sess.dir, ref)
            if os.path.exists(candidate):
                sess.reference_path = candidate
        # The manifest says whether the state it was written from had a valid
        # mix.  An invalidated mix (a pan, a mute, a new stem) leaves its old
        # file on disk, but it must not come back after a restart, so the
        # flag decides and the file on disk does not: only a strict boolean
        # true restores it, a manifest from before the flag existed keeps the
        # old behaviour, and a flag that is present but is not a boolean
        # cannot establish a valid result.
        if "has_mix" in data:
            mix_valid = data["has_mix"] is True
        else:
            mix_valid = True  # a manifest from before the flag existed
        if mix_valid and os.path.exists(os.path.join(sess.dir, "mix.wav")):
            sess.mix_path = os.path.join(sess.dir, "mix.wav")
        if os.path.exists(os.path.join(sess.dir, "raw.wav")):
            sess.raw_path = os.path.join(sess.dir, "raw.wav")
        result_path = os.path.join(sess.dir, "result.json")
        # a result without a mix is an orphan of a render that is not there
        # any more; it is not restored
        if sess.mix_path is not None and os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    sess.result = _without_source_refs(json.load(fh))
            except (OSError, json.JSONDecodeError):
                sess.invalidate_result()
        # The level-matched pair is only restored when the result says the
        # pair is trustworthy *and* the files are really there.  An old
        # session has no such pair, and its raw.wav may be limited: it is
        # never presented as a matched comparison.
        monitoring = (sess.result or {}).get("monitoring") or {}
        if monitoring.get("status") == "matched":
            mix_view = os.path.join(sess.dir, "monitor_mix.wav")
            raw_view = os.path.join(sess.dir, "monitor_raw.wav")
            if os.path.exists(mix_view) and os.path.exists(raw_view):
                sess.monitor_mix_path = mix_view
                sess.monitor_raw_path = raw_view
        return sess

    def stem_names(self) -> list[str]:
        return sorted(
            n
            for n in os.listdir(self.stems_dir)
            if os.path.splitext(n)[1].lower() in ALLOWED_EXT
        )

    def _role_source_identity(self):
        files = []
        for name in self.stem_names():
            info = os.stat(os.path.join(self.stems_dir, name))
            files.append((name, info.st_dev, info.st_ino, info.st_size,
                          info.st_mtime_ns, info.st_ctime_ns))
        return self.decode_policy, tuple(files)

    def has_mix(self) -> bool:
        return self.mix_path is not None and os.path.exists(self.mix_path)

    def invalidate_result(self) -> None:
        """The render this session holds is no longer what it holds.

        The comparison pair belongs to a render: when the render is gone,
        the pair goes with it.  The files on disk are left alone -- they are
        the user's, and a recovery file may be the only copy of something.
        What goes is the claim that they are current.
        """
        self.result = None
        self.monitor_mix_path = None
        self.monitor_raw_path = None

    def monitoring_state(self) -> dict:
        """What may honestly be said about the comparison pair, right now.

        The pair is only current when the session still holds the result
        that claims it, that result belongs to a mix that is still there,
        and both files are really on disk.  When it is not, the answer says
        why -- and it never says "level matched".
        """
        monitoring = (self.result or {}).get("monitoring") or {}
        if monitoring.get("status") != "matched":
            return dict(monitoring)
        if self.result is None or not self.has_mix():
            return {"status": "unavailable", "reason": "remixed-required"}
        mix = self.monitor_mix_path
        raw = self.monitor_raw_path
        if not mix or not raw:
            return {"status": "unavailable", "reason": "file-incomplete"}
        if not (os.path.exists(mix) and os.path.exists(raw)):
            return {"status": "unavailable", "reason": "file-incomplete"}
        return dict(monitoring)

    def monitoring_pair(self) -> tuple[str | None, str | None]:
        """The paths of the comparison pair, or (None, None).

        The one place that decides whether a session may serve a comparison
        view: an invalidated render, a result that claims a pair the
        session does not have, and a file that is gone all mean there is
        nothing to serve.
        """
        if self.monitoring_state().get("status") != "matched":
            return None, None
        return self.monitor_mix_path, self.monitor_raw_path


def _project_root() -> str:
    return os.path.abspath(os.path.expanduser(
        os.environ.get("AUTOMIX_PROJECT_ROOT") or "~/Music/UTUVO Automix Projects"
    ))


def _check_project_roots(project_root: str, session_root: str) -> None:
    disposable = os.path.realpath(session_root)
    if os.path.commonpath([os.path.realpath(project_root), disposable]) == disposable:
        raise ValueError("project storage must be outside disposable sessions")


def _project_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{12}", value):
        raise ValueError("invalid project id")
    return value


def _project_manifest(directory: str) -> dict:
    if os.path.islink(directory) or os.path.islink(os.path.join(directory, Session.MANIFEST)):
        raise ValueError("project links are not supported")
    with open(os.path.join(directory, Session.MANIFEST), encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("invalid project manifest")
    reference = data.get("reference")
    if reference is not None and (
        not isinstance(reference, str) or not reference or
        os.path.basename(reference) != reference or reference in {".", ".."} or
        not os.path.isfile(os.path.join(directory, reference))
    ):
        raise ValueError("project reference is missing or outside the project")
    return data


def _check_project_tree(directory: str) -> None:
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise ValueError("project must be a regular directory")
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            mode = os.lstat(os.path.join(root, name)).st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError("project links and special files are not supported")


def _copy_project_file(source: str, destination: str) -> str:
    # A file that changes to a link or special file during a copy is rejected.
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as incoming:
        if not stat.S_ISREG(os.fstat(incoming.fileno()).st_mode):
            raise ValueError("project files must be regular files")
        with open(destination, "xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    return destination


def _copy_project_tree(source: str, stage: str) -> None:
    _check_project_tree(source)
    # Links are never dereferenced during the copy; also reject a link that
    # appeared after the initial check before this copy can be published.
    shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=True,
                    copy_function=_copy_project_file)
    _check_project_tree(stage)


def _rebase_project_plan(directory: str, final: str) -> None:
    path = os.path.join(directory, "mixplan.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    if not isinstance(plan, dict) or not isinstance(plan.get("source"), dict):
        raise ValueError("invalid saved MixPlan source")
    plan["source"]["stems_dir"] = os.path.join(final, "stems")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)


def _publish_project_copy(stage: str, final: str) -> None:
    # Reserve the new name exclusively: even an existing empty directory is
    # never replaced. Only this request's empty reservation is replaced.
    os.mkdir(final)
    try:
        os.replace(stage, final)
    except Exception:
        os.rmdir(final)
        raise


def _save_project(sess: Session, root: str, name, controls) -> dict:
    """Called with the session lock held; publish a new, complete snapshot."""
    if not isinstance(name, str) or len(name) > 120:
        raise ValueError("project name must be at most 120 characters")
    draft = clean_controls(controls)
    if draft is None:
        raise ValueError("project controls are invalid")
    if not sess.stem_names():
        raise ValueError("import stems before saving a project")
    _check_project_roots(root, os.path.dirname(sess.dir))
    if os.path.islink(root):
        raise ValueError("project storage cannot be a link")
    os.makedirs(root, exist_ok=True)
    project_id = uuid.uuid4().hex[:12]
    final = os.path.join(root, project_id)
    stage = tempfile.mkdtemp(prefix=".project-stage-", dir=root)
    saved_at = time.time()
    name = name.strip() or "Untitled project"
    try:
        _copy_project_tree(sess.dir, stage)
        sess.write_manifest(os.path.join(stage, Session.MANIFEST),
                            id=project_id, project_name=name,
                            draft_controls=draft, saved_at=saved_at)
        _project_manifest(stage)
        _rebase_project_plan(stage, final)
        _publish_project_copy(stage, final)
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)
    return {"id": project_id, "name": name, "saved_at": saved_at, "path": final}


def _list_projects(root: str) -> list[dict]:
    if not os.path.exists(root):
        return []
    if os.path.islink(root):
        raise ValueError("project storage cannot be a link")
    projects = []
    for name in os.listdir(root):
        if not re.fullmatch(r"[0-9a-f]{12}", name):
            continue
        directory = os.path.join(root, name)
        try:
            data = _project_manifest(directory)
            saved_at = _saved_number(data.get("saved_at"), 0.0)
            if saved_at is None:
                continue
            projects.append({"id": name, "name": str(data.get("project_name") or "Untitled project"),
                             "saved_at": saved_at})
        except (OSError, ValueError):
            continue  # do not list a snapshot whose manifest cannot be read
    return sorted(projects, key=lambda item: item["saved_at"], reverse=True)


def _open_project(project_root: str, project_id: str, session_root: str) -> Session:
    _check_project_roots(project_root, session_root)
    source = os.path.join(project_root, _project_id(project_id))
    if os.path.islink(project_root):
        raise ValueError("project storage cannot be a link")
    _project_manifest(source)
    if os.path.commonpath([os.path.realpath(source), os.path.realpath(session_root)]) == os.path.realpath(source):
        raise ValueError("working sessions cannot be inside a saved project")
    sid = uuid.uuid4().hex[:12]
    final = os.path.join(session_root, sid)
    stage = tempfile.mkdtemp(prefix=".project-stage-", dir=session_root)
    published = False
    try:
        _copy_project_tree(source, stage)
        data = _project_manifest(stage)
        if clean_controls(data.get("draft_controls")) is None:
            raise ValueError("saved project controls are invalid")
        data["id"] = sid
        with open(os.path.join(stage, Session.MANIFEST), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        _rebase_project_plan(stage, final)
        # A copied old directory must not immediately look expired to prune.
        os.utime(stage, None)
        _publish_project_copy(stage, final)
        published = True
        sess = Session.load(session_root, sid)
        if sess is None:
            raise ValueError("saved project could not be restored")
        return sess
    except Exception:
        if published:
            shutil.rmtree(final)
        raise
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)


def _safe_name(raw: str) -> str:
    """Strip any path, keep the extension, refuse anything exotic.

    ``X-Filename`` arrives percent-encoded because HTTP headers are latin-1 and
    a UTF-8 filename would otherwise be mangled into mojibake before we ever
    see it.  ``unquote`` is a no-op for plain ASCII names, so old clients and
    curl still work.
    """
    name = os.path.basename(unquote(raw or "")).strip()
    # control characters (a newline in a file name is legal on disk but
    # breaks the plan's readability and the UI's layout) become underscores
    name = "".join(ch if ch.isprintable() else "_" for ch in name).strip()
    if not name or name.startswith("."):
        raise ValueError("bad filename")
    if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
        raise ValueError(f"unsupported file type: {name}")
    # a stem set can legitimately contain spaces and unicode; just no separators
    return name.replace("/", "_").replace("\\", "_")


def _styles() -> list[dict]:
    out = []
    if not os.path.isdir(PRESET_DIR):
        return out
    english = planner.style_text_en(PRESET_DIR)
    for name in sorted(os.listdir(PRESET_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            p = planner.load_preset(os.path.splitext(name)[0], PRESET_DIR)
        except planner.PlanError:
            continue
        out.append(
            {
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "description": p.get("description", ""),
                "intent": p.get("intent"),
                "en": english.get(p["id"]),
                "target_lufs": p.get("target_lufs"),
                "true_peak_db": p.get("true_peak_db"),
            }
        )
    return out


#: loudness at or below this is silence, not a level: no gain can be
#: computed for it and no match can be claimed
SILENCE_LUFS = -70.0

#: How far below the render's ceiling the untreated sum stays when it is
#: raised to meet the mix.  The comparison files are float, so nothing clips
#: inside the file; the margin is for the playback path, where the browser's
#: own volume meets the inter-sample peaks.
MONITOR_TP_MARGIN_DB = 0.2


def _raw_sum_audio(stems: list) -> tuple[np.ndarray, int]:
    """The untreated sum of the native stems, with nothing applied to it.

    The same timeline, rate, phase and channel assumption the comparison has
    always used: the longest stem decides the length, a shorter one is padded
    with silence, and the stems are summed as they come.  No gain and no
    limiter -- the level is the caller's decision, not this function's.
    """
    n = max(s.audio.shape[0] for s in stems)
    acc = np.zeros((n, 2))
    for s in stems:
        a = s.audio
        if a.shape[0] < n:
            a = np.vstack([a, np.zeros((n - a.shape[0], 2))])
        acc += a
    return acc, stems[0].sr


def _raw_sum(stems: list, match_lufs: float, ceiling_db: float) -> tuple[np.ndarray, int]:
    """The untreated sum with its historical treatment: gain matched to the
    mix and then held under the ceiling.

    This is what ``raw.wav`` has always been, and it stays that way so the
    file keeps its meaning for everything that already reads it.  It is *not*
    the comparison view: the limiter bends the level over time, so the file
    is no longer one constant gain away from the mix.  The level-matched pair
    is planned by :func:`_monitor_plan` and written as float files.
    """
    acc, sr = _raw_sum_audio(stems)
    cur = dsp.loudness_lufs(acc, sr)
    if cur > SILENCE_LUFS:
        acc = acc * dsp.db_to_lin(match_lufs - cur)
    return dsp.true_peak_limit(acc, sr, ceiling_db), sr


# ------------------------------------------------------------- monitoring A/B
#
# The comparison between the finished mix and the untreated sum is only
# meaningful when both are the same loudness: otherwise the listener just
# picks the louder one.  The old code gain matched the sum and then ran it
# through the limiter, which bent the level over time -- the sum was no
# longer one gain away from the mix, so part of what the listener heard was
# the limiter and not the mix.
#
# The pair below is built the honest way.  Two views, one shared level, one
# constant gain each, nothing else applied: no limiter, no EQ, no polarity
# flip, no alignment and no sample rate conversion.  The mix view is the
# delivered PCM24 file decoded again, so the comparison is made against the
# audio that is actually shipped.


def _write_wav_float32(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a 32-bit float WAV.

    The comparison views are float on purpose: a PCM24 file would quantize
    the gain and change the very level the comparison is about.  Float32
    carries about 24 bits of mantissa, so the gain survives to ~1e-7 of full
    scale; what is lost is nothing that was in the signal to begin with.
    """
    frames = np.asarray(audio, dtype=np.float32)
    if frames.ndim == 1:
        frames = frames[:, None]
    n_frames, n_ch = (frames.shape[0], frames.shape[1]) if frames.ndim == 2 else (0, 1)
    payload = frames.tobytes()  # C order: frame0ch0, frame0ch1, ...
    header = b"".join([
        b"RIFF",
        struct.pack("<I", 36 + len(payload)),
        b"WAVE",
        b"fmt ",
        struct.pack("<IHHIIHH", 16, 3, n_ch, int(sr),
                    int(sr) * n_ch * 4, n_ch * 4, 32),
        b"data",
        struct.pack("<I", len(payload)),
    ])
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)


def _measure_side(arr: np.ndarray, sr: int) -> dict:
    """The loudness and the true peak of one monitoring input.

    An input that cannot be measured -- silence, or something too short to
    gate -- is reported as undefined instead of as a number: a level that
    does not exist cannot be matched to anything.
    """
    empty = {"lufs": None, "true_peak_db": None, "crest_db": None}
    if arr is None or getattr(arr, "size", 0) == 0:
        return empty
    try:
        lufs = float(dsp.loudness_lufs(arr, sr))
    except Exception:  # noqa: BLE001 - an unmeasurable input is undefined
        lufs = float("nan")
    try:
        peak = float(dsp.true_peak_db(arr, sr))
    except Exception:  # noqa: BLE001
        peak = float("nan")
    lufs_ok = math.isfinite(lufs) and lufs > SILENCE_LUFS
    peak_ok = math.isfinite(peak)
    return {
        "lufs": lufs if lufs_ok else None,
        "true_peak_db": peak if peak_ok else None,
        "crest_db": (peak - lufs) if (lufs_ok and peak_ok) else None,
    }


def _monitor_plan(mix_audio: np.ndarray, mix_sr: int, raw_audio: np.ndarray,
                  raw_sr: int, ceiling_db: float) -> dict:
    """Plan the shared level of the two monitoring views.

    The plan says how loud both views will be, which constant gain each one
    gets, and whether the pair can honestly be called level matched.  It
    never touches the audio: the caller applies the gains.

    The rules:

    * both views end at the same loudness, so the comparison is about sound
      and not about level;
    * that level is never above the rendered mix's own loudness, and never
      above what the untreated sum can reach under the ceiling;
    * the mix view is only ever attenuated -- what the renderer delivered
      already honours the ceiling, and raising it would make the comparison
      louder than the export it is supposed to represent;
    * the untreated sum is raised to meet the mix, so it keeps a small
      safety margin under the ceiling;
    * a side whose loudness is undefined (silence, or too short to measure)
      cannot be matched: the plan says so instead of pretending.
    """
    for name, arr in (("mix", mix_audio), ("raw", raw_audio)):
        if arr is None or not np.all(np.isfinite(arr)):
            # a corrupt input is never published: the render fails and the
            # previous result stays exactly as it was
            raise ValueError(f"the {name} monitoring input is not finite")

    ceiling = float(ceiling_db)
    monitor_ceiling = ceiling - MONITOR_TP_MARGIN_DB
    mix_side = _measure_side(mix_audio, mix_sr)
    raw_side = _measure_side(raw_audio, raw_sr)

    plan: dict = {
        "status": "matched",
        "level_lufs": None,
        "ceiling_db": ceiling,
        "monitor_ceiling_db": monitor_ceiling,
        "margin_db": MONITOR_TP_MARGIN_DB,
        "mix": mix_side,
        "raw": raw_side,
        "mix_gain_db": None,
        "raw_gain_db": None,
        "mix_peak_db": None,
        "raw_peak_db": None,
        "headroom_db": None,
        "precision": "float32",
    }

    missing = [name for name, side in (("mix", mix_side), ("raw", raw_side))
               if side["lufs"] is None or side["crest_db"] is None]
    if missing:
        # nothing to match: the views are written as they are, and the
        # result says in as many words that they are not level matched
        plan["status"] = "unavailable"
        plan["reason"] = "level-undefined"
        plan["unavailable"] = missing
        return plan

    mix_lufs = mix_side["lufs"]
    raw_lufs = raw_side["lufs"]
    # the mix is only ever attenuated, so it needs no margin of its own: the
    # ceiling the renderer honours is already in the file that is played
    mix_limit = min(mix_lufs, ceiling - mix_side["crest_db"])
    # the untreated sum is raised until it meets the mix, so it keeps the
    # safety margin under the ceiling
    raw_limit = monitor_ceiling - raw_side["crest_db"]
    level = min(mix_limit, raw_limit)

    mix_gain = level - mix_lufs
    raw_gain = level - raw_lufs
    plan["level_lufs"] = level
    plan["mix_gain_db"] = mix_gain
    plan["raw_gain_db"] = raw_gain
    plan["mix_peak_db"] = mix_side["true_peak_db"] + mix_gain
    plan["raw_peak_db"] = raw_side["true_peak_db"] + raw_gain
    plan["headroom_db"] = min(
        ceiling - plan["mix_peak_db"], ceiling - plan["raw_peak_db"]
    )
    return plan


# --------------------------------------------------------------------- worker
#
# Rendering happens in a child process.  Two reasons, both learned the hard way:
# Python does not hand memory back to the OS, so after one three-minute song the
# server was sitting on ~4 GB while completely idle; and a crash or an OOM kill
# inside a render would take the whole UI down with it.  The parent now owns
# nothing but file paths and stays small.


#: how much the processed-stem export may write before the server refuses
EXPORT_LIMIT_BYTES = 4 * 1024**3


def _export_size_bytes(plan: dict, n_stems: int, has_reverb: bool = True) -> int:
    """How much the processed-stem export will write, in bytes.

    24-bit stereo: three bytes per sample per channel, one file per stem
    plus the shared reverb return.
    """
    sr = int(plan["source"]["sample_rate"])
    seconds = float(plan["source"].get("duration_s") or 0.0)
    frames = int(plan["source"].get("frames") or np.ceil(seconds * sr))
    return (frames * 2 * 3 + 44) * (n_stems + int(has_reverb))


def _worker(payload: dict, conn) -> None:
    try:
        sess_dir = payload["sess_dir"]
        # where the render writes its output; the parent publishes it
        stage_dir = payload.get("stage_dir") or sess_dir
        stems_dir = payload["stems_dir"]
        roles = payload["roles"]
        # the handler always sends the session's own policy; a payload built
        # without one is decoded like any other new load
        decode_policy = analyze.check_decode_policy(
            payload.get("decode_policy", analyze.DEFAULT_DECODE_POLICY)
        )

        def progress(stage: str, done: int = 0, total: int = 0) -> None:
            try:
                conn.send({"progress": {"stage": stage, "done": done, "total": total}})
            except Exception:  # noqa: BLE001 - progress is best effort
                pass

        progress("loading")
        preset = planner.load_preset(payload["style"], PRESET_DIR)
        stems = analyze.load_stems(stems_dir, decode_policy)
        muted = set(payload.get("muted") or [])
        if muted:
            kept = [s for s in stems if analyze.track_id(s.name) not in muted]
            if not kept:
                raise planner.PlanError("every stem is muted; nothing to mix")
            stems = kept
        progress("roles")
        analyze.detect_roles(stems)
        for st in stems:
            if st.name in roles:
                st.role = roles[st.name]
                st.role_source = "user"
        analyze.align(stems)
        for i, st in enumerate(stems, 1):
            progress("measuring", i, len(stems))
            analyze.measure(st)

        progress("planning")
        total = analyze.system_memory_mb()
        if total:
            analyze.check_memory(stems, budget_mb=total * 0.55)

        ref = (refmod.analyse(payload["reference_path"], decode_policy)
               if payload["reference_path"] else None)

        trims = payload.get("trims") or {}

        progress("rendering")
        collect = {} if payload.get("export_stems") else None
        plan, audio, report = pipeline.mix(
            stems,
            preset,
            preset_path=payload["style"],
            intensity=payload["intensity"],
            balance_preserve=payload.get("balance_preserve"),
            source_balance_profile=payload.get("source_balance_profile", "auto"),
            reference=ref,
            blend=payload["blend"],
            trims=trims,
            pans=payload.get("pans") or {},
            target_lufs=payload.get("target_lufs"),
            true_peak_db=payload.get("true_peak_db"),
            collect=collect,
            reverb_amount=payload.get("reverb_amount"),
            capture_groups=payload.get("capture_groups"),
        )

        progress("writing")
        # Everything the render produces is written into the staging folder,
        # never into the session itself.  The parent publishes the set once
        # the child is done, so a render that dies halfway cannot replace a
        # good mix with a broken one.
        # A/B honesty: the reference is played back at the *mix's* loudness.
        # Comparing a -14 LUFS mix against a -9 LUFS commercial master tells you
        # nothing except which one is louder.
        if payload["reference_path"]:
            try:
                ref_audio, ref_sr = analyze.load_stem_file(payload["reference_path"], decode_policy)
                cur = dsp.loudness_lufs(ref_audio, ref_sr)
                if cur > -70.0:
                    ref_audio = ref_audio * dsp.db_to_lin(report["lufs"] - cur)
                ref_audio = dsp.true_peak_limit(ref_audio, ref_sr, report["ceiling_db"])
            except Exception:  # noqa: BLE001 - A/B is a nicety, never fail the mix
                traceback.print_exc()
            else:
                # A comparison that cannot be written is not a comparison.
                # The old one was matched to the old mix, so a failed write
                # fails the render instead of leaving a stale file next to
                # a new mix.
                # staged: a half-written comparison file is never published
                _staged_write(
                    os.path.join(stage_dir, "reference_matched.wav"),
                    lambda path: render.write_wav24(path, ref_audio, ref_sr),
                )
        # Keep the previous render so two styles can be compared without
        # trusting memory: the user runs pop, then rock, and A/Bs the two.
        # The rotation of the old mix to "previous" belongs to the parent,
        # together with the publication of the new one.
        mix_path = os.path.join(stage_dir, "mix.wav")
        prev_style = None
        if os.path.exists(os.path.join(sess_dir, "mix.wav")):
            try:
                with open(os.path.join(sess_dir, "result.json"), encoding="utf-8") as fh:
                    prev_style = (json.load(fh).get("style") or {}).get("id")
            except (OSError, json.JSONDecodeError):
                prev_style = None

        _staged_write(
            mix_path,
            lambda path: render.write_wav24(
                path, audio, int(plan["source"]["sample_rate"])
            ),
        )
        raw, raw_sr = _raw_sum(stems, report["lufs"], report["ceiling_db"])
        _staged_write(
            os.path.join(stage_dir, "raw.wav"),
            lambda path: render.write_wav24(path, raw, raw_sr),
        )

        # --- the level-matched comparison pair --------------------------
        # Two views of the same song at one shared loudness, each exactly
        # one constant gain away from its source: the mix that was just
        # written (the delivered PCM24 audio, decoded again) and the
        # untreated sum of the native stems.  No limiter, no EQ, no
        # alignment, no resampling -- the plan says what the gains are and
        # the plan is the only thing that touches the level.
        mix_in, mix_in_sr = analyze.load_stem_file(mix_path)
        raw_in, raw_in_sr = _raw_sum_audio(stems)
        monitoring = _monitor_plan(mix_in, mix_in_sr, raw_in, raw_in_sr,
                                   report["ceiling_db"])
        monitoring["export_lufs"] = report["lufs"]
        monitoring["mix_view"] = "monitor_mix.wav"
        monitoring["raw_view"] = "monitor_raw.wav"
        monitoring["level_offset_db"] = (
            monitoring["level_lufs"] - report["lufs"]
            if monitoring["level_lufs"] is not None
            else None
        )
        mix_gain = dsp.db_to_lin(monitoring["mix_gain_db"] or 0.0)
        raw_gain = dsp.db_to_lin(monitoring["raw_gain_db"] or 0.0)
        _staged_write(
            os.path.join(stage_dir, "monitor_mix.wav"),
            lambda path: _write_wav_float32(path, mix_in * mix_gain, mix_in_sr),
        )
        _staged_write(
            os.path.join(stage_dir, "monitor_raw.wav"),
            lambda path: _write_wav_float32(path, raw_in * raw_gain, raw_in_sr),
        )

        exported = []
        if collect:
            progress("exporting")
            # the export writes 24-bit stereo WAVs: one per stem plus the
            # shared reverb return.  Refuse before writing rather than after,
            # so the user is not left with a half-written directory.
            est = _export_size_bytes(plan, len(collect.get("stems") or {}),
                                     collect.get("reverb") is not None and bool(np.any(collect["reverb"])))
            if est > EXPORT_LIMIT_BYTES:
                raise planner.PlanError(
                    "the export would be about %.1f GB, over the %.1f GB limit; "
                    "drop some stems, shorten the song, or export from the CLI"
                    % (est / 1073741824.0, EXPORT_LIMIT_BYTES / 1073741824.0)
                )
            # the same writer the CLI uses: one implementation, so the note and
            # the master chain file cannot drift apart from the command line
            # The whole directory is staged: the parent replaces the published
            # cache in one step, after everything else succeeded, so a failed
            # export leaves the previous cache alone and never prunes it.
            exported = _staged_write(
                os.path.join(stage_dir, "processed"),
                lambda path: render.write_stem_export(
                    path, plan, collect, report, audio
                ),
            )

        plan["source"]["stems_dir"] = stems_dir
        _staged_write(
            os.path.join(stage_dir, "mixplan.json"),
            lambda path: render.write_plan_json(path, plan),
        )
        _staged_write(
            os.path.join(stage_dir, "report.md"),
            lambda path: render.write_report(path, plan, report, stems),
        )

        conn.send(
            {
                "ok": True,
                "payload": {
                    "lufs": report["lufs"],
                    "target_lufs": report["target_lufs"],
                    "true_peak_db": report["true_peak_db"],
                    "ceiling_db": report["ceiling_db"],
                    "crest_db": report["crest_db"],
                    "width": report["width"],
                    "mono_loss_db": report.get("mono_loss_db"),
                    "wet_ratio_db": report.get("wet_ratio_db"),
                    "wet_ratio_measurement": report.get("wet_ratio_measurement"),
                    "correlation": report["correlation"],
                    "warnings": list(plan.get("warnings") or []) + list(report.get("warnings") or []),
                    "style": plan["style"],
                    "source_balance": plan["style"]["source_balance"],
                    "sections": plan.get("sections") or [],
                    "reference_passes": report.get("reference_passes", 1),
                    "reference_tone": report.get("reference_tone"),
                    "unmask": plan.get("unmask"),
                    # what the capture groups protected, and where that
                    # protection stops: the UI shows it, the renderer ignores it
                    "capture_groups": plan.get("capture_groups"),
                    "prev_style": prev_style,
                    "exported": exported,
                    "export_summary": report.get("export_summary"),
                    # what the comparison pair is: the shared level, the
                    # two constant gains, and whether the pair can honestly
                    # be called level matched.  The export target lives
                    # beside it, not inside it.
                    "monitoring": monitoring,
                    "sample_rate": int(plan["source"]["sample_rate"]),
                    "duration_s": round(float(plan["source"]["duration_s"]), 2),
                    "verdicts": [
                        {"check": name, "measured": value, "verdict": verdict}
                        for name, value, verdict in render._verdicts(plan, report)
                    ],
                    "tracks": [
                        {
                            "id": t["id"],
                            "role": t["role"],
                            "gain_db": t["gain_db"],
                            "pan": t["pan"],
                            "hpf_hz": t["hpf_hz"],
                            "send": (t.get("sends") or {}).get("plate", 0.0),
                            "trim_db": t.get("trim_db", 0.0),
                            "ride_db": (
                                [
                                    min(g for _, g in t["ride"]),
                                    max(g for _, g in t["ride"]),
                                ]
                                if t.get("ride")
                                else None
                            ),
                            "evidence": t.get("evidence", []),
                            "dynamic_eq": next((r.get("dynamic_eq") for r in report["tracks"] if r["id"] == t["id"]), None),
                        }
                        for t in plan["tracks"]
                    ],
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the server
        traceback.print_exc()
        try:
            conn.send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    # let the parent map a user error to 4xx instead of 500
                    "kind": type(exc).__name__,
                }
            )
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------- publication
#
# A render produces several files that have to agree with each other: the
# mix, the raw sum, the plan, the receipt, the result, the manifest and the
# processed-stem export.  Writing them straight into the session folder means
# a failure halfway through leaves a mix that cannot be decoded next to a
# plan that describes a different mix.
#
# So the child renders into a staging folder inside the session (same
# filesystem, so every step is a rename) and the *parent* publishes the set,
# with a backup of every file it replaces, so a failed publication can be
# rolled back to the last complete render.
#
# os.replace is atomic per file.  The set of files is not one atomic unit: a
# power cut between two renames can still leave a half-published session.
# This is a failure-safety net for ordinary errors, not a crash-proof store.

#: staging and backup folders live inside the session folder, so every
#: publication step is a same-filesystem rename
STAGE_PREFIX = ".stage-"
BACKUP_PREFIX = ".backup-"


class PublishError(RuntimeError):
    """A publication step failed; the session was rolled back when possible."""


def _remove_path(path: str) -> None:
    """Remove a file, a folder or a link if it is there."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _new_staging_dir(sess_dir: str) -> str:
    """A fresh staging folder inside the session, on the same filesystem."""
    path = os.path.join(sess_dir, STAGE_PREFIX + uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=False)
    return path


def _staged_write(target: str, write):
    """Run ``write`` against a scratch name, then rename it into place.

    A render that dies halfway through writing a 24-bit WAV must not leave a
    file that looks finished: the final name only appears once every byte is
    on disk, so an interrupted write can never be published.
    """
    scratch = target + ".part"
    try:
        result = write(scratch)
        os.replace(scratch, target)
        return result
    except BaseException:
        try:
            if os.path.isdir(scratch):
                shutil.rmtree(scratch, ignore_errors=True)
            else:
                os.unlink(scratch)
        except OSError:
            pass
        raise


class _Publication:
    """Parent-controlled publication of one staged render into a session.

    Before the first file is moved, every published file it will replace is
    moved aside into a backup folder.  If a later step fails, those backups
    are put back, so the session keeps the bytes it had before the failed
    render.  The backup folder is removed only after the last step succeeded
    or the rollback completed.
    """

    def __init__(self, sess_dir: str, stage_dir: str):
        self.sess_dir = sess_dir
        self.stage_dir = stage_dir
        self.backup_dir = os.path.join(sess_dir, BACKUP_PREFIX + uuid.uuid4().hex[:12])
        self._backed_up: list[str] = []
        self._created: set[str] = set()
        self._rotated = False
        #: the files that are the only copy of something that was published:
        #: they are never deleted to make a rollback look complete
        self._sole_copy: set[str] = set()
        #: the folders this publication created; an empty one is removed
        #: again when the publication is rolled back
        self._created_folders: list[str] = []

    def _target(self, name: str) -> str:
        return os.path.join(self.sess_dir, name)

    def _backup_path(self, name: str) -> str:
        return os.path.join(self.backup_dir, name)

    def backup(self, name: str) -> bool:
        """Move an existing published file or folder aside.

        Returns True when there was one.  A link is moved, not followed: the
        session never writes through anything it did not create.
        """
        target = self._target(name)
        if not os.path.exists(target) and not os.path.islink(target):
            return False
        backup = self._backup_path(name)
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        os.replace(target, backup)
        self._backed_up.append(name)
        return True

    def publish(self, name: str) -> None:
        """Move a staged file or folder into the session."""
        self.backup(name)
        os.replace(os.path.join(self.stage_dir, name), self._target(name))
        if name not in self._backed_up:
            self._created.add(name)

    def publish_folder(self, name: str) -> None:
        """Merge a staged folder into the published one, then prune.

        The published folder is not replaced wholesale: the server owns the
        files it generated, and only those are pruned.  Anything else that
        happens to sit in there -- a file the user left, a folder, a link --
        stays where it is, exactly as it was before.
        """
        staged = os.path.join(self.stage_dir, name)
        target = self._target(name)
        if not os.path.exists(target):
            # a folder this publication created: an empty one is removed
            # again when the publication is rolled back
            self._created_folders.append(name)
        os.makedirs(target, exist_ok=True)
        current = set()
        for entry in sorted(os.scandir(staged), key=lambda e: e.name):
            if not entry.is_file(follow_symlinks=False):
                continue
            current.add(entry.name)
            rel = os.path.join(name, entry.name)
            self.backup(rel)
            os.replace(entry.path, os.path.join(target, entry.name))
            if rel not in self._backed_up:
                self._created.add(rel)
        # prune what the server generated and no longer owns
        for entry in sorted(os.scandir(target), key=lambda e: e.name):
            if (entry.name in current
                    or os.path.splitext(entry.name)[1].lower()
                    not in (".wav", ".json", ".txt")
                    or not entry.is_file(follow_symlinks=False)):
                continue
            # moving it aside is the removal: the backup folder is thrown
            # away when the publication is done, and put back if it fails
            self.backup(os.path.join(name, entry.name))

    def rotate_previous_mix(self) -> None:
        """Make the mix that was current a moment ago the previous mix."""
        if "mix.wav" not in self._backed_up:
            return
        os.replace(self._backup_path("mix.wav"), self._target("mix_prev.wav"))
        self._rotated = True
        if "mix_prev.wav" not in self._backed_up:
            self._created.add("mix_prev.wav")

    def check_targets(self, names: list[str], has_export: bool) -> None:
        """Refuse to publish onto anything that is not a plain file we own.

        A link is never written *through*: publishing into a linked folder
        would write outside the session.  A folder is never replaced by a
        file: the folder is the user's, and replacing it would throw it
        away.  Both are refused before anything is moved, so the session
        keeps the result it had.
        """
        for name in names:
            target = self._target(name)
            if os.path.islink(target):
                raise PublishError(
                    f"{name} is a link; refusing to publish through it"
                )
            if os.path.exists(target) and not os.path.isfile(target):
                raise PublishError(
                    f"{name} is not a regular file; refusing to replace it"
                )
        if not has_export:
            return
        folder = self._target("processed")
        if os.path.islink(folder):
            raise PublishError(
                "processed is a link; refusing to publish through it"
            )
        if os.path.exists(folder) and not os.path.isdir(folder):
            raise PublishError(
                "processed is not a folder; refusing to replace it"
            )
        staged = os.path.join(self.stage_dir, "processed")
        if not os.path.isdir(staged):
            return
        for entry in os.scandir(staged):
            if not entry.is_file(follow_symlinks=False):
                continue
            dest = os.path.join(folder, entry.name)
            if os.path.islink(dest):
                raise PublishError(
                    f"processed/{entry.name} is a link; refusing to replace it"
                )
            if os.path.exists(dest) and not os.path.isfile(dest):
                raise PublishError(
                    f"processed/{entry.name} is not a regular file; "
                    "refusing to replace it"
                )

    def rollback(self) -> list[str]:
        """Put the previous files back.  Returns what could not be restored."""
        problems: list[str] = []
        if self._rotated:
            # the mix that was current is now mix_prev.wav; move it back
            try:
                os.replace(self._target("mix_prev.wav"), self._target("mix.wav"))
                self._rotated = False
            except OSError as exc:
                problems.append(f"mix.wav: {exc}")
                # mix_prev.wav holds the only copy of that mix now: keep it
                self._created.discard("mix_prev.wav")
                self._sole_copy.add("mix_prev.wav")
        for name in sorted(self._created):
            try:
                _remove_path(self._target(name))
            except OSError as exc:
                problems.append(f"{name}: {exc}")
        for name in reversed(self._backed_up):
            if name in self._sole_copy:
                # this file is the only copy of a mix that was published: it
                # stays where it is, and the older version stays in the
                # backup folder.  Nothing is thrown away to make the
                # rollback look complete.
                continue
            backup = self._backup_path(name)
            if not os.path.exists(backup):
                continue
            try:
                _remove_path(self._target(name))
                os.replace(backup, self._target(name))
            except OSError as exc:
                problems.append(f"{name}: {exc}")
        for name in self._created_folders:
            # a folder this publication created and nothing else uses
            try:
                folder = self._target(name)
                if os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
            except OSError:
                pass
        return problems

    def recovery_paths(self) -> list[str]:
        """Where the files that are the only copy of something were kept."""
        return [self._target(name) for name in sorted(self._sole_copy)]

    def discard(self) -> None:
        """Remove the backup folder: the publication is done or was undone."""
        _remove_path(self.backup_dir)


def _publish_render(sess: "Session", stage_dir: str, result: dict,
                    controls: dict | None, prev_style: str | None,
                    reverb_amount: float, expect_reference: bool = False) -> None:
    """Publish one staged render into the session, or leave the session alone.

    Everything the child produced is already in ``stage_dir``.  This moves it
    into place in the parent process, with a backup of every file it
    replaces, so that a failure during publication leaves the previous
    complete render in place.  Raises :class:`PublishError` when the
    publication failed; the session keeps its old bytes unless the rollback
    itself failed, in which case the backups are kept and named in the error.
    """
    sess_dir = sess.dir
    old_state = (sess.controls, sess.result, sess.mix_path, sess.raw_path,
                 sess.monitor_mix_path, sess.monitor_raw_path,
                 sess.reverb_amount, sess.draft_controls)
    result = dict(result)
    result["muted"] = sorted(sess.muted)
    pub = _Publication(sess_dir, stage_dir)
    # The commit holds the session lock: a state change must not run into
    # the middle of it, and the manifest that is about to be published must
    # not be written by somebody else in between.
    sess.lock.acquire()
    try:
        # The parent writes the result and the manifest into the staging
        # folder first, so they are published like every other output.  The
        # controls of this render go into the manifest, but the live session
        # only learns about them once the publication succeeded: a request
        # that arrives while the commit runs must not be able to save them,
        # and a rollback must not have to undo them.
        with open(os.path.join(stage_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)
        if prev_style:
            with open(os.path.join(stage_dir, "prev_style.txt"), "w", encoding="utf-8") as fh:
                fh.write(prev_style)
        # the manifest is written with has_mix=True: the mix is published in
        # the same batch, and a rollback puts the old manifest back
        sess.write_manifest(os.path.join(stage_dir, sess.MANIFEST), has_mix=True,
                            controls=controls, reverb_amount=reverb_amount,
                            draft_controls=None)

        optional = []
        if os.path.exists(os.path.join(stage_dir, "reference_matched.wav")):
            optional.append("reference_matched.wav")
        has_export = os.path.isdir(os.path.join(stage_dir, "processed"))

        # The level-matched comparison pair.  A render that says it is level
        # matched must deliver the pair with it: publishing a mix next to a
        # pair that is missing (or a pair of an older render) is exactly the
        # stale comparison this pair exists to end.  A render that claims no
        # matched pair -- an old one, or one whose level could not be
        # measured -- does not need them, so they stay optional there.
        pair = ["monitor_mix.wav", "monitor_raw.wav"]
        monitoring = result.get("monitoring") or {}
        claims_pair = monitoring.get("status") == "matched"

        outputs = ["mix.wav", "raw.wav", "mixplan.json", "report.md",
                   "result.json"]
        if claims_pair:
            outputs = outputs[:2] + pair + outputs[2:]
        else:
            # a render that does not claim a pair must not leave the previous
            # pair on disk: it belonged to the previous render, and a stale
            # pair next to a new mix is exactly what this pair ends
            for name in pair:
                pub.backup(name)

        required = outputs + [sess.MANIFEST]
        for name in required + optional:
            if not os.path.exists(os.path.join(stage_dir, name)):
                raise PublishError(f"the render did not produce {name}")

        # nothing is written through a link, and no folder is replaced by a
        # file: both are refused before anything moves, so the session keeps
        # the result it had
        pub.check_targets(
            outputs + optional + [sess.MANIFEST, "mix_prev.wav",
                                  "prev_style.txt"],
            has_export,
        )

        # 1. the previous mix and its label are moved aside first: the
        #    rotation at the end puts the old mix in their place
        pub.backup("mix_prev.wav")
        pub.backup("prev_style.txt")
        # 2. every file is replaced one at a time -- the old one is moved
        #    aside right before the new one takes its place, so the session
        #    never sits without a mix for longer than one rename
        for name in outputs + optional:
            pub.publish(name)
        # 3. the generated export cache is merged in and pruned, but only
        #    now: a render that failed before this point left it alone
        if has_export:
            pub.publish_folder("processed")
        # 4. the label of the mix that is about to be rotated
        if prev_style:
            pub.publish("prev_style.txt")
        # 5. the mix that was current becomes the previous mix, exactly once
        pub.rotate_previous_mix()
        # 6. the manifest goes last: it is the record of the render, and it
        #    says so only once the rest of it is in place
        pub.publish(sess.MANIFEST)
        # 7. a render that was supposed to produce a comparison but could
        #    not must not publish a new mix next to the old comparison: the
        #    old one was matched to the old mix.  Failing here keeps the
        #    whole previous set, and nothing is removed on the way out.
        if expect_reference and "reference_matched.wav" not in optional:
            raise PublishError(
                "the comparison with the reference could not be written; "
                "the previous render was kept"
            )
        # the session now describes the published render.  The controls
        # become the session's controls only now, so a request that arrived
        # while the commit was running could not have saved them.
        sess.controls = controls
        sess.draft_controls = None
        sess.reverb_amount = reverb_amount
        sess.mix_path = os.path.join(sess_dir, "mix.wav")
        sess.raw_path = os.path.join(sess_dir, "raw.wav")
        # the pair belongs to the render that produced it: it is only
        # current when the published result says it is level matched
        if claims_pair:
            sess.monitor_mix_path = os.path.join(sess_dir, "monitor_mix.wav")
            sess.monitor_raw_path = os.path.join(sess_dir, "monitor_raw.wav")
        else:
            sess.monitor_mix_path = None
            sess.monitor_raw_path = None
        sess.result = result
    except Exception as exc:  # noqa: BLE001 - a failed publication, whatever broke
        problems = pub.rollback()
        (sess.controls, sess.result, sess.mix_path, sess.raw_path,
         sess.monitor_mix_path, sess.monitor_raw_path,
         sess.reverb_amount, sess.draft_controls) = old_state
        if problems:
            # the backups stay where they are: they are the only copy of the
            # files that could not be restored.  The staging folder stays
            # too: it holds the render that could not be published.
            kept = pub.recovery_paths()
            raise PublishError(
                "publication failed (%s) and the rollback did not complete (%s); "
                "the previous files are kept in %s, the render that could not "
                "be published is kept in %s%s"
                % (exc, "; ".join(problems), pub.backup_dir, stage_dir,
                   (" and the only copy of the previous mix is at %s"
                    % ", ".join(kept)) if kept else "")
            ) from exc
        pub.discard()
        _remove_path(stage_dir)
        raise PublishError(
            "publication failed, the previous render was kept: %s" % exc
        ) from exc
    finally:
        sess.lock.release()
    pub.discard()
    _remove_path(stage_dir)
    # the result that was published: the answer the client gets is this
    # snapshot, not whatever the session holds a moment later
    return result


def _byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """One inclusive bytes range, None to ignore, ValueError if unsatisfiable."""
    if not value or size == 0:
        return None
    unit, separator, spec = value.strip(" \t").partition("=")
    if not separator or unit.lower() != "bytes":
        return None
    match = re.fullmatch(r"([0-9]*)-([0-9]*)", spec.strip(" \t"))
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if first and last:
        a, b = first.lstrip("0") or "0", last.lstrip("0") or "0"
        if (len(a), a) > (len(b), b):
            return None  # reversed endpoints are malformed, not unsatisfiable

    def bounded(digits: str) -> int:
        digits = digits.lstrip("0") or "0"
        limit = str(size)
        # Never pass an unbounded numeral to int(), including leading zeros.
        return size if (len(digits), digits) > (len(limit), limit) else int(digits)

    if not first:
        count = bounded(last)
        if count == 0:
            raise ValueError("bad range")
        return size - count, size - 1
    start = bounded(first)
    if start >= size:
        raise ValueError("bad range")
    end = min(bounded(last), size - 1) if last else size - 1
    return start, end


class Handler(BaseHTTPRequestHandler):
    server_version = "utuvo-automix/0.2"

    # ---------------------------------------------------------------- plumbing

    def log_message(self, fmt, *args):  # quieter default logging
        if os.environ.get("AUTOMIX_VERBOSE"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, code: int, msg: str):
        self._json(code, {"error": msg})

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_UPLOAD:
            raise ValueError("file too large")
        chunks = []
        left = length
        while left > 0:
            chunk = self.rfile.read(min(left, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)

    def _session(self, sid: str) -> Session | None:
        with _lock:
            return _sessions.get(sid)

    def _serve_file(self, path: str, ctype: str, download: str | None = None):
        # A file that is opened while a render is publishing its result must
        # be the one that was published: the session lock keeps the open and
        # the commit apart.  The bytes are read outside the lock, so a long
        # download does not hold the session up.
        session = getattr(self, "_session_scope", None)
        lock = session.lock if session is not None else contextlib.nullcontext()
        with lock:
            # the file is chosen under the same lock that guards the commit:
            # a download must not see the old choice next to the new file
            if callable(path):
                path = path()
            if not os.path.exists(path):
                return self._error(404, "not ready")
            size = os.path.getsize(path)
            start, end = 0, size - 1

            try:
                selected = _byte_range(self.headers.get("Range"), size)
            except ValueError:
                return self._send(
                    416, json.dumps({"error": "bad range"}, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                    {"Accept-Ranges": "bytes", "Content-Range": f"bytes */{size}"},
                )
            if selected is not None:
                start, end = selected

            handle = open(path, "rb")
        with handle:
            handle.seek(start)
            body = handle.read(end - start + 1)

        extra = {"Accept-Ranges": "bytes"}
        if download:
            extra["Content-Disposition"] = f'attachment; filename="{download}"'
        if selected is not None:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(200, body, ctype, extra)

    # --------------------------------------------------------------------- GET

    def do_GET(self):
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        self._session_scope = None

        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")

        if path == "/api/styles":
            return self._json(200, {"styles": _styles()})


        if path == "/api/projects":
            try:
                root = getattr(self.server, "project_root", _project_root())
                return self._json(200, {"projects": _list_projects(root), "root": root})
            except (OSError, ValueError) as exc:
                return self._error(500, f"cannot read saved projects: {exc}")

        if path == "/api/roles":
            return self._json(200, {"roles": list(analyze.ROLES)})

        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "session":
            sess = self._session(parts[2])
            if sess is None:
                return self._error(404, "unknown session")
            # the files of this session are opened under its lock, so a
            # download cannot see half of a commit
            self._session_scope = sess

            if len(parts) == 4 and parts[3] == "state":
                # The whole answer is taken under the session lock, so it
                # describes one moment: a render that is publishing its
                # result cannot be read half-published.
                with sess.lock:
                    # The roles and the revision are copied together under
                    # the shared lock, so the answer describes one moment.
                    # Reading them one after the other could mix a role map
                    # from before a commit with a revision from after it.
                    with _lock:
                        role_state = {
                            "manual_roles": dict(sess.roles),
                            "role_revision": sess.role_revision,
                        }
                    answer = {
                        "id": sess.id,
                        "stems": sess.stem_names(),
                        "has_reference": sess.reference_path is not None,
                        "has_mix": sess.has_mix(),
                        "source_balance": (sess.result or {}).get("source_balance"),
                        "style": (sess.result or {}).get("style"),
                        "muted": sorted(sess.muted),
                        "pans": dict(sess.pans),
                        # the declared capture groups, so a reopened session
                        # shows the same grouping the user made
                        "capture_groups": [dict(g) for g in sess.capture_groups],
                        "capture_groups_error": sess.capture_groups_error,
                        # the engine's own track id for every file, so a client
                        # never has to guess the id it has to declare with
                        "track_ids": {
                            name: analyze.track_id(name) for name in sess.stem_names()
                        },
                        "reverb_amount": sess.reverb_amount,
                        # the user's own role choices, so the UI can show them
                        # right away instead of waiting for the slow detection
                        # call to come back
                        "manual_roles": role_state["manual_roles"],
                        # how many manual role changes this session has accepted;
                        # a client uses it to order role snapshots
                        "role_revision": role_state["role_revision"],
                        # the requested controls of the last successful mix;
                        # None for a session that has no saved snapshot
                        "controls": sess.controls,
                        "project_name": sess.project_name,
                        "draft_controls": sess.draft_controls,
                        "has_prev": os.path.exists(os.path.join(sess.dir, "mix_prev.wav")),
                    }
                return self._json(200, answer)
            if len(parts) == 6 and parts[3] == "processed" and parts[4] == "file":
                raw = unquote(parts[5])
                # README.txt and _master_chain.json are not audio, so they
                # cannot go through _safe_name
                base = os.path.basename(raw)
                if base == "README.txt":
                    return self._serve_file(
                        os.path.join(sess.dir, "processed", "README.txt"),
                        "text/plain; charset=utf-8",
                    )
                if base in ("_master_chain.json", "_export_report.json"):
                    return self._serve_file(
                        os.path.join(sess.dir, "processed", base),
                        "application/json",
                    )
                try:
                    name = _safe_name(raw)
                except ValueError:
                    return self._error(400, "bad name")
                content_type = "application/json" if name.endswith(".json") else "text/plain; charset=utf-8" if name.endswith(".txt") else "audio/wav"
                return self._serve_file(os.path.join(sess.dir, "processed", name), content_type)

            if len(parts) == 6 and parts[3] == "audio" and parts[4] == "stem":
                try:
                    name = _safe_name(parts[5])
                except ValueError as exc:
                    return self._error(400, str(exc))
                path = os.path.join(sess.stems_dir, name)
                if not os.path.isfile(path):
                    return self._error(404, "no such stem")
                return self._serve_file(path, "audio/wav")

            if len(parts) == 6 and parts[3] == "audio" and parts[4] == "monitor":
                # The level-matched comparison views: float, one constant
                # gain each, the same loudness.  They are *not* the export --
                # the export is /audio/mix and /download/mix -- and a session
                # that has no trustworthy pair simply does not have them.
                kind = parts[5]
                # The path is resolved inside the same critical section that
                # opens the file, the way the other audio routes do it: an
                # edit that invalidates the render while the request is on
                # its way cannot be served as a current comparison, and an
                # edit that waits for the lock waits for the file too.
                if kind == "mix":
                    return self._serve_file(
                        lambda: sess.monitoring_pair()[0] or "", "audio/wav")
                if kind == "raw":
                    return self._serve_file(
                        lambda: sess.monitoring_pair()[1] or "", "audio/wav")
                return self._error(404, "no such audio")

            if len(parts) == 4 and parts[3] == "peaks":
                kind = (parse_qs(urlparse(self.path).query).get("kind") or ["mix"])[0]
                # the file is chosen under the session lock, so the peaks
                # belong to the mix the session is showing
                with sess.lock:
                    matched = os.path.join(sess.dir, "reference_matched.wav")
                    mon_mix, mon_raw = sess.monitoring_pair()
                    path = {
                        "mix": sess.mix_path,
                        "raw": sess.raw_path,
                        "monitor_mix": mon_mix,
                        "monitor_raw": mon_raw,
                        "reference": (
                            matched if os.path.exists(matched)
                            else sess.reference_path
                        ),
                    }.get(kind)
                    if not path or not os.path.exists(path):
                        return self._error(404, "not ready")
                return self._json(200, _peaks(path, sess))

            if len(parts) == 5 and parts[3] == "audio":
                kind = parts[4]
                if kind == "mix":
                    # the path is picked under the session lock, in the same
                    # moment the file is opened
                    return self._serve_file(lambda: sess.mix_path or "", "audio/wav")
                if kind == "processed":
                    return self._error(400, "file name required")

                if kind == "prev":
                    return self._serve_file(
                        os.path.join(sess.dir, "mix_prev.wav"), "audio/wav"
                    )
                if kind == "raw":
                    # the historical untreated sum: gain matched to the mix
                    # and then held under the ceiling, so it is *not* the
                    # level-matched comparison view (the monitoring routes
                    # above are).  Kept as it was for everything that
                    # already reads it.
                    return self._serve_file(lambda: sess.raw_path or "", "audio/wav")
                if kind == "stem":
                    # audition one raw stem: the fastest way to answer "is this
                    # really the bass?" without rendering anything
                    return self._error(400, "stem name required")

                if kind == "reference" and sess.reference_path:
                    matched = os.path.join(sess.dir, "reference_matched.wav")
                    # the choice between the matched comparison and the raw
                    # reference is made under the session lock, in the same
                    # moment the file is opened: a download must not answer
                    # with the raw reference while the matched one is there
                    return self._serve_file(
                        lambda: (matched if os.path.exists(matched)
                                 else sess.reference_path), "audio/wav"
                    )
                return self._error(404, "no such audio")
            if len(parts) == 4 and parts[3] == "progress":
                return self._json(200, {"progress": sess.progress, "running": sess.proc is not None})
            if len(parts) == 4 and parts[3] == "roles":
                return self._detect_roles(sess)
            if len(parts) == 4 and parts[3] == "result":
                # the answer is one snapshot: an edit that is accepted while
                # the request is on its way must not turn it into "not ready"
                with sess.lock:
                    result = sess.result
                    if result is not None and isinstance(result.get("monitoring"), dict):
                        # a comparison the session can no longer be trusted
                        # with is not advertised as a matched one
                        result = dict(result)
                        result["monitoring"] = sess.monitoring_state()
                if result is None:
                    return self._error(404, "not ready")
                return self._json(200, result)
            if len(parts) == 4 and parts[3] == "plan":
                return self._serve_file(
                    os.path.join(sess.dir, "mixplan.json"), "application/json; charset=utf-8"
                )
            if len(parts) == 4 and parts[3] == "report":
                if not sess.has_mix():
                    return self._error(404, "not ready")
                return self._serve_file(
                    os.path.join(sess.dir, "report.md"), "text/markdown; charset=utf-8"
                )
            if len(parts) == 5 and parts[3] == "download" and parts[4] == "mix":
                return self._serve_file(lambda: sess.mix_path or "", "audio/wav",
                                        download="mix.wav")

        return self._error(404, "not found")

    # -------------------------------------------------------------------- POST

    def do_POST(self):
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        hold = None  # the session lock, held while a state change is applied

        try:
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "open":
                sess = _open_project(getattr(self.server, "project_root", _project_root()),
                                     parts[2], self.server.session_root)
                with _lock:
                    _sessions[sess.id] = sess
                return self._json(200, {"id": sess.id})

            if path == "/api/session":
                _prune(self.server.session_root)
                sess = Session(self.server.session_root)
                with _lock:
                    _sessions[sess.id] = sess
                return self._json(200, {"id": sess.id})

            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "session":
                sess = self._session(parts[2])
                if sess is None:
                    return self._error(404, "unknown session")
                action = parts[3] if len(parts) > 3 else ""

                if action in {"stem", "reference", "remove", "roles", "trim", "mute", "pan",
                             "capture_groups", "project"}:
                    if sess.proc is not None and sess.proc.is_alive():
                        return self._error(
                            409, "a render is running; wait for it or cancel it first"
                        )
                    # A session that is rendering or publishing refuses the
                    # edit instead of queueing it into the commit: a queued
                    # edit would run after the answer and could invalidate
                    # the result the run is about to report.
                    if sess.busy:
                        return self._error(
                            409,
                            "a render is publishing its result; try again in a moment",
                        )
                    # One state change at a time, and never during a commit:
                    # the render publishes the mix and the manifest together,
                    # and a save in between would put the old controls next
                    # to the new mix.
                    hold = sess.lock
                    hold.acquire()
                    if sess.busy:
                        return self._error(
                            409,
                            "a render is publishing its result; try again in a moment",
                        )

                if action == "project":
                    req = json.loads(self._body() or b"{}")
                    return self._json(200, _save_project(
                        sess, getattr(self.server, "project_root", _project_root()),
                        req.get("name", ""), req.get("controls")))

                if action == "stem":
                    name = _safe_name(self.headers.get("X-Filename", ""))
                    data = self._body()
                    if not data:
                        return self._error(400, "empty upload")
                    names = sess.stem_names()
                    if name not in names and len(names) >= MAX_STEMS:
                        return self._error(
                            413, f"a session holds at most {MAX_STEMS} stems; remove one first"
                        )
                    used = sum(
                        os.path.getsize(os.path.join(sess.stems_dir, n))
                        for n in names
                        if os.path.exists(os.path.join(sess.stems_dir, n))
                    )
                    if used + len(data) > MAX_SESSION_BYTES:
                        return self._error(
                            413,
                            f"session would exceed {MAX_SESSION_BYTES // (1024 ** 3)} GB of audio",
                        )
                    with open(os.path.join(sess.stems_dir, name), "wb") as fh:
                        fh.write(data)
                    # a new stem invalidates the previous render
                    sess.mix_path = None
                    sess.plan = None
                    sess.invalidate_result()
                    sess.save()
                    return self._json(200, {"stems": sess.stem_names()})

                if action == "reference":
                    name = _safe_name(self.headers.get("X-Filename", "reference.wav"))
                    data = self._body()
                    if not data:
                        return self._error(400, "empty upload")
                    dst = os.path.join(sess.dir, "reference" + os.path.splitext(name)[1].lower())
                    with open(dst, "wb") as fh:
                        fh.write(data)
                    sess.reference_path = dst
                    sess.save()
                    return self._json(200, {"ok": True})

                if action == "cancel":
                    proc = sess.proc
                    if proc is not None and proc.is_alive():
                        sess.cancel_requested = True
                        proc.terminate()
                        proc.join(5)
                        return self._json(200, {"cancelled": True})
                    return self._json(200, {"cancelled": False})

                if action == "mute":
                    req = json.loads(self._body() or b"{}")
                    tid = str(req.get("track_id") or "")
                    if not tid:
                        return self._error(400, "track_id required")
                    if bool(req.get("muted")):
                        sess.muted.add(tid)
                    else:
                        sess.muted.discard(tid)
                    sess.mix_path = None
                    sess.invalidate_result()
                    sess.save()
                    return self._json(200, {"muted": sorted(sess.muted)})

                if action == "pan":
                    req = json.loads(self._body() or b"{}")
                    tid = str(req.get("track_id") or "")
                    if not tid:
                        return self._error(400, "track_id required")
                    try:
                        value = float(req.get("pan"))
                    except (TypeError, ValueError):
                        return self._error(400, "pan must be a number")
                    if abs(value) > 1.0:
                        return self._error(400, "pan must be between -1 and 1")
                    sess.pans[tid] = round(value, 2)
                    sess.mix_path = None
                    sess.invalidate_result()
                    sess.save()
                    return self._json(200, {"pans": sess.pans})

                if action == "capture_groups":
                    # A draft is stored as it is: a group that has only one
                    # member yet is a legitimate state the user is in the
                    # middle of.  What is *not* allowed is a declaration we
                    # cannot read back, so the shape is checked here and the
                    # meaning is checked when a mix is planned.
                    req = json.loads(self._body() or b"{}")
                    raw = req.get("groups")
                    if raw is None:
                        raw = []
                    if not isinstance(raw, list):
                        return self._error(400, "groups must be a list")
                    groups = []
                    for item in raw:
                        if not isinstance(item, dict):
                            return self._error(400, "every capture group must be an object")
                        gid = item.get("id")
                        members = item.get("members")
                        if not isinstance(gid, str) or not gid.strip():
                            return self._error(400, "every capture group needs a name")
                        if not isinstance(members, list) or any(
                            not isinstance(m, str) or not m.strip() for m in members
                        ):
                            return self._error(
                                400, f"capture group {gid!r}: members must be a list of track ids"
                            )
                        pan = item.get("pan", 0.0)
                        if isinstance(pan, bool) or not isinstance(pan, (int, float)):
                            return self._error(
                                400, f"capture group {gid!r}: pan must be a finite number"
                            )
                        try:
                            pan_value = float(pan)
                        except (OverflowError, ValueError):
                            return self._error(
                                400, f"capture group {gid!r}: pan must be a finite number"
                            )
                        if not math.isfinite(pan_value):
                            return self._error(
                                400, f"capture group {gid!r}: pan must be a finite number"
                            )
                        if not -1.0 <= pan_value <= 1.0:
                            return self._error(
                                400, f"capture group {gid!r}: pan must be between -1 and 1"
                            )
                        # A stereo pair declares two channels and a shared
                        # trim.  The mode is a literal, the trim is a bounded
                        # number, and the two members are two different
                        # tracks -- nothing is inferred from the order the
                        # files were listed in.
                        mode = item.get("mode")
                        if "mode" in item and item["mode"] != "stereo_pair":
                            return self._error(
                                400, f"capture group {gid!r}: unsupported mode {item['mode']!r}"
                            )
                        trim_value = 0.0
                        if mode == "stereo_pair":
                            trim = item.get("trim_db", 0.0)
                            if isinstance(trim, bool) or not isinstance(trim, (int, float)):
                                return self._error(
                                    400, f"capture group {gid!r}: trim_db must be a finite number"
                                )
                            try:
                                trim_value = float(trim)
                            except (OverflowError, ValueError):
                                return self._error(
                                    400, f"capture group {gid!r}: trim_db must be a finite number"
                                )
                            if not math.isfinite(trim_value):
                                return self._error(
                                    400, f"capture group {gid!r}: trim_db must be a finite number"
                                )
                            if not -12.0 <= trim_value <= 12.0:
                                return self._error(
                                    400, f"capture group {gid!r}: trim_db must be between -12 and 12"
                                )
                            if len(members) != 2 or len(set(members)) != 2:
                                return self._error(
                                    400,
                                    f"capture group {gid!r}: a stereo pair needs exactly two "
                                    "different members -- the left channel and the right channel",
                                )
                        elif "trim_db" in item:
                            return self._error(
                                400,
                                f"capture group {gid!r}: trim_db needs the stereo_pair mode",
                            )
                        # A member may be named by its file name or by its
                        # canonical track id: the file name is what the user
                        # sees, the id is what the engine uses.  An unknown
                        # member is kept as it is, so the mix refuses it with
                        # a name instead of dropping it quietly.
                        file_ids = {
                            name: analyze.track_id(name) for name in sess.stem_names()
                        }
                        id_files = {tid: name for name, tid in file_ids.items()}
                        resolved = []
                        for member in members:
                            member = member.strip()
                            if member in file_ids:
                                resolved.append(file_ids[member])
                            elif member in id_files:
                                resolved.append(member)
                            else:
                                resolved.append(member)
                        if mode == "stereo_pair" and len(set(resolved)) != 2:
                            # two names that resolve to one track are one
                            # member, and a pair of one member is not a pair
                            return self._error(
                                400,
                                f"capture group {gid!r}: the left and the right member must be "
                                "two different tracks",
                            )
                        group = {"id": gid.strip(), "members": resolved, "pan": float(pan)}
                        if mode == "stereo_pair":
                            group["mode"] = "stereo_pair"
                            group["trim_db"] = trim_value
                        groups.append(group)
                    # Saving the same declaration again is not a change.  The
                    # web app posts the groups before every run, and throwing
                    # the finished mix away for that would lose the user's
                    # result.  A real change -- a different pan, different
                    # members, a group added or removed -- still makes the
                    # previous render stale.
                    unchanged = _same_capture_groups(groups, sess.capture_groups)
                    sess.capture_groups = groups
                    sess.capture_groups_error = None
                    if not unchanged:
                        sess.mix_path = None
                        sess.invalidate_result()
                    sess.save()
                    return self._json(200, {"capture_groups": sess.capture_groups, "changed": not unchanged})

                if action == "trim":
                    req = json.loads(self._body() or b"{}")
                    incoming = req.get("trims") or {}
                    if not isinstance(incoming, dict):
                        return self._error(400, "trims must be an object of track_id -> dB")
                    for tid, delta in incoming.items():
                        try:
                            value = float(delta)
                        except (TypeError, ValueError):
                            return self._error(400, f"bad trim for {tid}")
                        if abs(value) > 12.0:
                            return self._error(400, f"trim for {tid} is beyond +-12 dB")
                        key = str(tid)
                        if abs(value) < 1e-6:
                            sess.trims.pop(key, None)
                        else:
                            sess.trims[key] = round(value, 2)
                    # a trim changes the mix, so the previous render is stale
                    sess.mix_path = None
                    sess.invalidate_result()
                    sess.save()
                    return self._json(200, {"trims": sess.trims})

                if action == "roles":
                    req = json.loads(self._body() or b"{}")
                    incoming = req.get("roles") or {}
                    known = set(sess.stem_names())
                    # every name is checked before anything is written, so a
                    # rejected request changes nothing at all
                    updates: dict[str, str] = {}
                    for name, role in incoming.items():
                        # accept the file name ("keys.wav") or the bare track id
                        # ("keys"): a client that sends the latter used to get
                        # "unsupported file type: keys", which says nothing
                        raw = str(name)
                        safe = raw if raw in known else None
                        if safe is None and not raw.lower().endswith(ALLOWED_EXT):
                            for ext in ALLOWED_EXT:
                                if raw + ext in known:
                                    safe = raw + ext
                                    break
                        if safe is None:
                            return self._error(400, f"no such stem: {raw}")
                        if role in analyze.ROLES:
                            updates[safe] = str(role)
                    with _lock:
                        changed = False
                        for safe, value in updates.items():
                            if sess.roles.get(safe) != value:
                                sess.roles[safe] = value
                                changed = True
                        if changed:
                            # an accepted manual change advances the revision;
                            # a rejected one never does
                            sess.role_revision += 1
                        # a changed role invalidates the previous render
                        sess.mix_path = None
                        sess.plan = None
                        sess.invalidate_result()
                        sess.save()
                    return self._detect_roles(sess)

                if action == "remove":
                    req = json.loads(self._body() or b"{}")
                    name = _safe_name(str(req.get("name", "")))
                    target = os.path.join(sess.stems_dir, name)
                    if os.path.exists(target):
                        os.remove(target)
                    sess.mix_path = None
                    sess.plan = None
                    sess.invalidate_result()
                    return self._json(200, {"stems": sess.stem_names()})

                if action == "run":
                    return self._run(sess, self._body())

        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface it to the UI
            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}")
        finally:
            if hold is not None:
                hold.release()

        return self._error(404, "not found")

    # --------------------------------------------------------------------- run

    def _detect_roles(self, sess: Session):
        with sess._role_analysis_lock:
            try:
                identity = sess._role_source_identity()
                names = [entry[0] for entry in identity[1]]
                cached = sess._role_analysis_cache
                if cached is None or cached[0] != identity:
                    sess._role_analysis_cache = None
                    automatic = {}
                    if names:
                        stems = analyze.load_stems(sess.stems_dir, identity[0])
                        analyze.detect_roles(stems)
                        automatic = {s.name: {
                            "detected": s.role,
                            "source": s.role_source,
                            "id": analyze.track_id(s.name),
                            "metrics": analyze.quick_metrics(s),
                        } for s in stems}
                    # A concurrent upload/replacement must not give a changed
                    # source the old analysis. Failures never become a cache.
                    if identity != sess._role_source_identity():
                        raise analyze.StemError("source files changed during role analysis; try again")
                    sess._role_analysis_cache = (identity, automatic)
                automatic = sess._role_analysis_cache[1]
            except (analyze.StemError, OSError) as exc:
                sess._role_analysis_cache = None
                return self._error(400, str(exc))
        # the manual roles and the revision are copied together, so the answer
        # describes one state instead of two read at different moments
        with _lock:
            manual = dict(sess.roles)
            revision = sess.role_revision
        roles = {}
        for name, detected in automatic.items():
            roles[name] = dict(detected, role=manual.get(name, detected["detected"]),
                               source="user" if name in manual else detected["source"])
        return self._json(200, {"stems": names, "roles": roles, "role_revision": revision})

    def _run(self, sess: Session, body: bytes):
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._error(400, "bad json")

        try:
            if not isinstance(req, dict):
                raise ValueError("run options must be an object")
            profile, requested_k = balancemod.inputs(req.get("source_balance_profile", "auto"),
                                                   req.get("balance_preserve"))
            amount = planner._manual_reverb_amount(
                req.get("reverb_amount", None if sess.reverb_amount == 1.0 else sess.reverb_amount),
                optional=True,
            )
        except (ValueError, planner.PlanError) as exc:
            return self._error(400, str(exc))
        style = str(req.get("style") or "pop_2020s")
        if not sess.stem_names():
            return self._error(400, "no stems uploaded yet")
        try:
            planner.load_preset(style, PRESET_DIR)
        except planner.PlanError as exc:
            return self._error(400, str(exc))

        # A group whose member is excluded from the mix cannot be protected:
        # the audio it protects is not in the mix.  Say so instead of mixing
        # something that only looks protected.
        for group in sess.capture_groups:
            excluded = [m for m in group.get("members") or [] if m in sess.muted]
            if excluded:
                return self._error(
                    400,
                    f"capture group {group.get('id')!r} includes the excluded track(s) "
                    f"{', '.join(excluded)}; include them again or remove them from the group",
                )

        payload = {
            "sess_dir": sess.dir,
            "stems_dir": sess.stems_dir,
            "style": style,
            "intensity": max(0.0, min(1.0, float(req.get("intensity", 1.0)))),
            "balance_preserve": requested_k,
            "source_balance_profile": profile,
            "blend": max(0.0, min(1.0, float(req.get("blend", 0.7)))),
            "reference_path": sess.reference_path,
            "roles": dict(sess.roles),
            "trims": dict(sess.trims),
            "muted": sorted(sess.muted),
            "pans": dict(sess.pans),
            "capture_groups": [dict(g) for g in sess.capture_groups],
            "export_stems": bool(req.get("export_stems")),
            "target_lufs": req.get("target_lufs"),
            "true_peak_db": req.get("true_peak_db"),
            "reverb_amount": amount,
            "decode_policy": sess.decode_policy,
        }

        if not _render_slot.acquire(blocking=False):
            return self._error(
                429, "another render is already running; wait for it to finish"
            )
        try:
            return self._run_isolated(sess, payload, requested_controls(payload))
        finally:
            _render_slot.release()

    def _run_isolated(self, sess: Session, payload: dict, controls: dict | None = None):
        # a child process per render: memory goes back to the OS when it exits,
        # and a crash inside a render cannot take the UI with it
        # The child writes into a staging folder inside the session; the
        # parent publishes the set afterwards.  Cancellation is honoured up to
        # the moment the publication starts: after that the commit runs to
        # the end, because half a publication is worse than a late cancel.
        stage_dir = _new_staging_dir(sess.dir)
        payload = dict(payload)
        payload["stage_dir"] = stage_dir
        # The session is busy from here until the answer has been written.
        # An edit that arrives in that window is refused with a 409, not
        # queued into it: a queued edit must not be able to invalidate the
        # result this request is about to answer with.
        sess.busy = True
        publishing = False
        try:
            try:
                ctx = multiprocessing.get_context("spawn")
                parent_conn, child_conn = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_worker, args=(payload, child_conn),
                                   daemon=True)
                proc.start()
                child_conn.close()
            except Exception as exc:  # noqa: BLE001 - the child never ran
                return self._error(500, f"the render could not be started: {exc}")

            sess.proc = proc
            sess.cancel_requested = False
            sess.progress = {"stage": "starting", "done": 0, "total": 0}
            try:
                deadline = time.time() + RENDER_TIMEOUT_S
                resp = None
                while resp is None:
                    if not parent_conn.poll(0.2):
                        if time.time() > deadline:
                            proc.terminate()
                            proc.join(5)
                            return self._error(
                                504, f"render timed out after {RENDER_TIMEOUT_S}s")
                        continue
                    msg = parent_conn.recv()
                    if "progress" in msg:
                        sess.progress = msg["progress"]
                        continue
                    resp = msg
            except EOFError:
                proc.join(5)
                if sess.cancel_requested:
                    return self._error(499, "cancelled")
                return self._error(500, "render process died without reporting")
            finally:
                try:
                    parent_conn.close()
                except Exception:  # noqa: BLE001
                    pass
                sess.proc = None
                sess.progress = None
            proc.join(5)

            if not resp.get("ok"):
                msg = resp.get("error", "render failed")
                if "RAM" in msg:
                    code = 413
                elif resp.get("kind") in ("StemError", "PlanError", "RenderError", "ReferenceError"):
                    code = 400  # the input was the problem, not the server
                else:
                    code = 500
                return self._error(code, msg)

            # A cancel that arrived before the commit must not publish:
            # nothing has been moved into the session yet, so the previous
            # render is still the one on disk.
            if sess.cancel_requested:
                return self._error(499, "cancelled")

            publishing = True
            try:
                published = _publish_render(
                    sess,
                    stage_dir,
                    resp["payload"],
                    controls,
                    prev_style=resp["payload"].get("prev_style"),
                    reverb_amount=(
                        1.0 if payload.get("reverb_amount") is None
                        else payload["reverb_amount"]
                    ),
                    expect_reference=bool(payload.get("reference_path")),
                )
            except PublishError as exc:
                return self._error(500, str(exc))
            # the answer is the result that was published, not whatever the
            # session holds a moment later: an edit that was accepted in the
            # meantime must not turn it into something else
            return self._json(200, published)
        finally:
            if not publishing:
                # the publication cleans up its own staging folder; before
                # that it belongs to this request
                _remove_path(stage_dir)
            sess.busy = False


def _load_sessions(root: str) -> int:
    """Rebuild the in-memory sessions from their manifests at startup.

    The audio never left the disk, so losing the id->Session mapping on restart
    meant the UI's own URL could not resume a project that still existed.
    """
    if not os.path.isdir(root):
        return 0
    restored = 0
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        sess = Session.load(root, name)
        if sess is None:
            continue
        with _lock:
            _sessions[sess.id] = sess
        restored += 1
    return restored


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    if port < 0 or port > 65535:
        raise SystemExit(f"invalid port {port} (expected 0-65535)")
    root = os.path.join(tempfile.gettempdir(), "utuvo-automix-sessions")
    projects = _project_root()
    try:
        _check_project_roots(projects, root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    os.makedirs(root, exist_ok=True)
    pruned = _prune(root)
    if pruned:
        print(f"pruned {pruned} stale session(s)")
    restored = _load_sessions(root)
    if restored:
        print(f"restored {restored} session(s)")

    def bind(candidate: int):
        """Bind one address; None means that address is already in use."""
        try:
            # the bind itself is the reservation: probing the port with a
            # throwaway connection and closing it would let a second copy win
            # the race for the port we are about to publish
            return ThreadingHTTPServer((host, candidate), Handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                # a busy port is the only reason to move on; permission denied
                # or an unknown host is a real failure, not a busy port
                raise SystemExit(f"cannot bind {host}:{candidate}: {exc}") from exc
            return None

    # A double-clicked bundle has no console to complain in: if 8765 is taken
    # (a second copy, or the dev server) it must still come up somewhere.
    httpd = None
    if port == 0:
        # an explicit 0 asks the OS for one free port and nothing else
        httpd = bind(0)
        if httpd is None:
            raise SystemExit(f"cannot bind an ephemeral port on {host}")
    else:
        last = min(port + 19, 65535)
        for candidate in range(port, last + 1):
            httpd = bind(candidate)
            if httpd is not None:
                break
        if httpd is None:
            # every preferred address is taken: let the OS pick a free port
            print(f"ports {port}-{last} are busy, using an ephemeral port")
            httpd = bind(0)
            if httpd is None:
                raise SystemExit(f"cannot bind an ephemeral port on {host}")
    httpd.session_root = root  # type: ignore[attr-defined]
    httpd.project_root = projects  # created only by an explicit save
    # the OS may hand out a different port than the one we asked for, so the
    # published URL comes from the bound socket, never from `port`
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"automix UI  ->  {url}")
    print(f"sessions    ->  {root}")
    if open_browser and os.environ.get("AUTOMIX_NO_BROWSER") != "1":
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="local automix UI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--clean", action="store_true", help="delete old session folders first")
    args = ap.parse_args(argv)

    if args.clean:
        shutil.rmtree(os.path.join(tempfile.gettempdir(), "utuvo-automix-sessions"), ignore_errors=True)

    serve(args.host, args.port, args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
