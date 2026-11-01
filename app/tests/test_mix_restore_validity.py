"""An invalidated mix must stay invalid after a restart.

The server writes ``has_mix`` into the session manifest every time it saves
the state.  When a mix is invalidated (a pan, a mute, a new stem) the server
clears the live mix and the live result and saves the manifest, so the
manifest says ``has_mix: false`` -- but the old ``mix.wav`` and
``result.json`` are still on disk, because they are the recovery files of the
last successful render.

Reopening the session used to restore the mix anyway, because the restore
only asked whether ``mix.wav`` was there.  The session came back with a mix it
no longer had: the state said "no mix" while the running server answered
"yes, here is your mix", and the result endpoint answered 200 with a render
that had been thrown away.

These tests drive the real restore path (``Session.load`` and the HTTP result
endpoint) with temporary synthetic files: tiny generated WAVs, a hand-written
manifest, no real music, no external fixtures and no strings matched against
the source.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import pytest

from automix import server as srvmod
from automix.server import Handler, Session

SR = 48000
SID = "restorecase"

#: the result of a render that really happened once
RESULT = {"style": "rock", "sections": [{"name": "verse"}], "summary": "a finished render"}

#: the controls of that render, in the shape the manifest stores them
CONTROLS = {
    "style": "rock",
    "intensity": 0.8,
    "source_balance_profile": "balanced_stems",
    "balance_preserve": 0.35,
    "blend": 0.25,
    "target_lufs": -20.0,
    "export_stems": True,
    "reverb_amount": 0.5,
}

#: sentinel for "this manifest has no has_mix field at all"
ABSENT = object()


# ------------------------------------------------------------------ helpers


def _write_wav(path: str, frames: int = 240) -> None:
    """Write a real, tiny WAV file: a short tone, nothing borrowed."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440.0 * i / SR)))
            for i in range(frames)
        ))


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _make_session(root: str, sid: str = SID, *, flag=ABSENT, mix: bool = True,
                  result: bool = True, raw: bool = True,
                  reference: bool = True) -> str:
    """Create a session on disk and return its directory.

    ``flag`` is the value the manifest carries in ``has_mix``; ``ABSENT``
    leaves the field out, the way a manifest from before the field existed
    does.
    """
    d = os.path.join(root, sid)
    os.makedirs(os.path.join(d, "stems"), exist_ok=True)
    if mix:
        _write_wav(os.path.join(d, "mix.wav"))
    if raw:
        _write_wav(os.path.join(d, "raw.wav"))
    if reference:
        _write_wav(os.path.join(d, "reference.wav"))
    if result:
        with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(RESULT, fh)
    manifest = {
        "id": sid,
        "roles": {"guitar_01_L.wav": "guitar"},
        "role_revision": 3,
        "trims": {"guitar_01_L.wav": 1.5},
        "muted": ["guitar_01_R.wav"],
        "pans": {"guitar_01_L.wav": -0.25},
        "capture_groups": [
            {"id": "pair", "members": ["guitar_01_L.wav", "guitar_01_R.wav"], "pan": 0.0}
        ],
        "reverb_amount": 0.5,
        "reference": "reference.wav",
        "has_mix": True,
        "controls": dict(CONTROLS),
    }
    if flag is ABSENT:
        manifest.pop("has_mix")
    else:
        manifest["has_mix"] = flag
    with open(os.path.join(d, "session.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return d


def _get(base: str, path: str):
    req = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.fixture()
def http_server(tmp_path):
    """A real server on 127.0.0.1:0, with its sessions rebuilt from disk."""
    root = str(tmp_path / "sessions")
    os.makedirs(root, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.session_root = root  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    finally:
        httpd.shutdown()
        httpd.server_close()
        with srvmod._lock:
            srvmod._sessions.pop(SID, None)


# ------------------------------------------------------------- the behaviour


def test_an_invalidated_mix_does_not_come_back_after_a_restart(tmp_path):
    """has_mix false means "there is no mix", even though mix.wav is there."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=False)
    mix_bytes = _read(os.path.join(d, "mix.wav"))
    result_bytes = _read(os.path.join(d, "result.json"))
    manifest_bytes = _read(os.path.join(d, "session.json"))

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.mix_path is None
    assert reopened.result is None
    assert reopened.has_mix() is False
    # the recovery files are left alone: nothing is deleted, nothing is written
    assert _read(os.path.join(d, "mix.wav")) == mix_bytes
    assert _read(os.path.join(d, "result.json")) == result_bytes
    assert _read(os.path.join(d, "session.json")) == manifest_bytes


def test_an_invalidated_mix_stays_invalid_through_load_save_load(tmp_path):
    """Saving the restored session must not resurrect the mix either."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=False)
    mix_bytes = _read(os.path.join(d, "mix.wav"))
    result_bytes = _read(os.path.join(d, "result.json"))

    first = Session.load(root, SID)
    assert first is not None
    assert first.has_mix() is False
    first.save()

    # the manifest the session writes now still says there is no mix
    with open(os.path.join(d, "session.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["has_mix"] is False

    second = Session.load(root, SID)
    assert second is not None
    assert second.mix_path is None
    assert second.result is None
    assert second.has_mix() is False
    # the old output files are still exactly as they were
    assert _read(os.path.join(d, "mix.wav")) == mix_bytes
    assert _read(os.path.join(d, "result.json")) == result_bytes


def test_a_session_invalidated_by_the_server_stays_invalid_after_a_restart(tmp_path):
    """The server's own path: invalidate, save, reopen."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=ABSENT)
    # the manifest starts as a saved session with a mix
    sess = Session(root, SID)
    sess.mix_path = os.path.join(d, "mix.wav")
    sess.result = dict(RESULT)
    sess.save()
    with open(os.path.join(d, "session.json"), encoding="utf-8") as fh:
        assert json.load(fh)["has_mix"] is True

    # what the pan, mute and stem handlers do: the mix is gone, the files stay
    sess.mix_path = None
    sess.result = None
    sess.save()
    with open(os.path.join(d, "session.json"), encoding="utf-8") as fh:
        assert json.load(fh)["has_mix"] is False

    reopened = Session.load(root, SID)
    assert reopened is not None
    assert reopened.mix_path is None
    assert reopened.result is None
    assert reopened.has_mix() is False
    assert os.path.exists(os.path.join(d, "mix.wav"))
    assert os.path.exists(os.path.join(d, "result.json"))


def test_a_valid_mix_comes_back_after_a_restart(tmp_path):
    """has_mix true and the files are there: the mix and its result return."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=True)

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.mix_path == os.path.join(d, "mix.wav")
    assert reopened.result == RESULT
    assert reopened.has_mix() is True


def test_a_manifest_from_before_the_flag_still_restores_the_mix(tmp_path):
    """A manifest with no has_mix field at all keeps the old behaviour."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=ABSENT)

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.mix_path == os.path.join(d, "mix.wav")
    assert reopened.result == RESULT
    assert reopened.has_mix() is True


def test_a_valid_flag_without_a_mix_does_not_restore_an_orphan_result(tmp_path):
    """A result without a mix belongs to a render that is not there any more."""
    root = str(tmp_path / "sessions")
    _make_session(root, flag=True, mix=False)

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.mix_path is None
    assert reopened.result is None
    assert reopened.has_mix() is False


@pytest.mark.parametrize("flag", [None, "true", "yes", 1, 0, 1.0, [], {}, ["true"]])
def test_a_broken_flag_cannot_bring_the_mix_back(tmp_path, flag):
    """A flag that is not a boolean says nothing about a valid result."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=flag)
    mix_bytes = _read(os.path.join(d, "mix.wav"))
    result_bytes = _read(os.path.join(d, "result.json"))
    manifest_bytes = _read(os.path.join(d, "session.json"))

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.mix_path is None
    assert reopened.result is None
    assert reopened.has_mix() is False
    # a broken flag is not repaired into a good one, and nothing is deleted
    assert _read(os.path.join(d, "session.json")) == manifest_bytes
    assert _read(os.path.join(d, "mix.wav")) == mix_bytes
    assert _read(os.path.join(d, "result.json")) == result_bytes


def test_the_rest_of_the_state_survives_an_invalidated_restore(tmp_path):
    """Only the mix and its result are held back; the work is not lost."""
    root = str(tmp_path / "sessions")
    d = _make_session(root, flag=False)

    reopened = Session.load(root, SID)

    assert reopened is not None
    assert reopened.roles == {"guitar_01_L.wav": "guitar"}
    assert reopened.role_revision == 3
    assert reopened.trims == {"guitar_01_L.wav": 1.5}
    assert reopened.muted == {"guitar_01_R.wav"}
    assert reopened.pans == {"guitar_01_L.wav": -0.25}
    assert reopened.capture_groups == [
        {"id": "pair", "members": ["guitar_01_L.wav", "guitar_01_R.wav"], "pan": 0.0}
    ]
    assert reopened.capture_groups_error is None
    assert reopened.reverb_amount == 0.5
    assert reopened.reference_path == os.path.join(d, "reference.wav")
    assert reopened.raw_path == os.path.join(d, "raw.wav")
    # the controls of the last successful mix are still there to be shown
    assert reopened.controls == CONTROLS


def test_the_result_endpoint_answers_404_after_a_restart_with_an_invalidated_mix(http_server):
    """The running server must not serve a render that was thrown away."""
    base, root = http_server
    _make_session(root, flag=False)
    assert srvmod._load_sessions(root) == 1

    code, body = _get(base, f"/api/session/{SID}/result")

    assert code == 404, body[:200]


def test_the_result_endpoint_answers_200_after_a_restart_with_a_valid_mix(http_server):
    """The same server, with a mix the manifest vouches for."""
    base, root = http_server
    _make_session(root, flag=True)
    assert srvmod._load_sessions(root) == 1

    code, body = _get(base, f"/api/session/{SID}/result")

    assert code == 200, body[:200]
    assert json.loads(body) == RESULT
