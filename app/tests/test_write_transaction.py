"""Failure safety of the render publication.

A render produces a set of files that have to agree with each other: the
mix, the raw sum, the plan, the receipt, the result, the manifest and the
processed-stem export.  These tests drive the *real* publication path --
real files, real renames, real rollbacks -- and inject the failures a full
disk, a killed child or a cancel produces.

Everything here is synthetic: no real session, no real audio, no network
outside 127.0.0.1.
"""

from __future__ import annotations

import io
import json
import os
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from automix import server as srvmod
from automix.server import Handler, PublishError, Session, _publish_render, _staged_write


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("write_tx_sessions"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.session_root = root  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    finally:
        httpd.shutdown()
        httpd.server_close()


def _req(base, path, data=None, method="GET", headers=None):
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _wav_bytes(freq=220.0, dur=1.0, sr=48000, amp=0.3) -> bytes:
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    pcm = (np.stack([mono, mono], axis=1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _new_session(base) -> str:
    code, body, _ = _req(base, "/api/session", data=b"", method="POST")
    assert code == 200
    return json.loads(body)["id"]


def _upload(base, sid, name, data):
    return _req(base, f"/api/session/{sid}/stem", data=data, method="POST",
                headers={"X-Filename": name,
                         "Content-Type": "application/octet-stream"})


def _two_stems(base, sid):
    assert _upload(base, sid, "kick.wav", _wav_bytes(60.0))[0] == 200
    assert _upload(base, sid, "lead.wav", _wav_bytes(440.0))[0] == 200


def _render(base, sid, opts=None):
    payload = json.dumps(opts or {"style": "pop_2020s"}).encode()
    code, body, _ = _req(base, f"/api/session/{sid}/run", data=payload,
                         method="POST",
                         headers={"Content-Type": "application/json"})
    return code, body


def _state(base, sid):
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200
    return json.loads(body)


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _session_dir(root, sid) -> str:
    return os.path.join(root, sid)


def _no_leftovers(sess_dir):
    """No staging or backup folder of this render may survive."""
    leftovers = [n for n in os.listdir(sess_dir)
                 if n.startswith(srvmod.STAGE_PREFIX)
                 or n.startswith(srvmod.BACKUP_PREFIX)]
    assert leftovers == [], leftovers


def _one_shot_failure(target_dir, target_name):
    """A replace that fails once, the way a transient disk error does.

    Only the rename *into* the given name fails: the copy that moves a file
    aside for the rollback is a rename into the backup folder and has to
    work, or the session could not be put back.
    """
    real_replace = os.replace
    state = {"fired": False}

    def failing(src, dst, *args, **kwargs):
        if (not state["fired"]
                and os.path.dirname(str(dst)) == target_dir
                and os.path.basename(str(dst)) == target_name):
            state["fired"] = True
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    return failing


def _always_fail(target_dir, target_name):
    """A replace that keeps failing: the rollback cannot use it either."""
    real_replace = os.replace

    def failing(src, dst, *args, **kwargs):
        if (os.path.dirname(str(dst)) == target_dir
                and os.path.basename(str(dst)) == target_name):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    return failing


# ------------------------------------------------------- end to end over HTTP

def test_a_successful_run_publishes_a_consistent_session(srv):
    """Everything the render produced has to be there, and it has to agree."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "export_stems": True})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)

    with wave.open(os.path.join(sess_dir, "mix.wav"), "rb") as w:
        assert w.getnframes() > 0
    with wave.open(os.path.join(sess_dir, "raw.wav"), "rb") as w:
        assert w.getnframes() > 0

    plan = json.loads(_read(os.path.join(sess_dir, "mixplan.json")))
    result = json.loads(_read(os.path.join(sess_dir, "result.json")))
    manifest = json.loads(_read(os.path.join(sess_dir, "session.json")))
    assert plan["style"]["id"] == "rock"
    assert result["style"]["id"] == "rock"
    assert manifest["has_mix"] is True
    assert os.path.getsize(os.path.join(sess_dir, "report.md")) > 0
    assert os.listdir(os.path.join(sess_dir, "processed"))
    _no_leftovers(sess_dir)


def test_the_published_plan_points_at_the_original_stems(srv):
    """The staging folder must never leak into the plan the user reads."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "pop_2020s"})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)
    plan = json.loads(_read(os.path.join(sess_dir, "mixplan.json")))
    assert plan["source"]["stems_dir"] == os.path.join(sess_dir, "stems")
    assert srvmod.STAGE_PREFIX not in json.dumps(plan)


def test_a_failed_publication_keeps_the_previous_render(srv, monkeypatch):
    """The failure the ticket is about: a broken write must not eat the mix."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock"})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)

    keep = {}
    for name in ("mix.wav", "raw.wav", "mixplan.json", "report.md",
                 "result.json", "session.json"):
        keep[name] = _read(os.path.join(sess_dir, name))

    # fail on a mid-publication step: the plan cannot be moved into place
    monkeypatch.setattr(srvmod.os, "replace",
                        _one_shot_failure(sess_dir, "mixplan.json"))
    code, body = _render(base, sid, {"style": "pop_2020s"})
    assert code == 500, (code, body[:300])

    for name, blob in keep.items():
        assert _read(os.path.join(sess_dir, name)) == blob, name
    # the previous mix was not rotated away by the failed render
    assert not os.path.exists(os.path.join(sess_dir, "mix_prev.wav"))
    _no_leftovers(sess_dir)
    # and the session still reports the render it did publish
    assert _state(base, sid)["has_mix"] is True


def test_a_failed_first_render_publishes_nothing(srv, monkeypatch):
    """With no earlier render there is nothing to fall back to: nothing shows."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    sess_dir = _session_dir(root, sid)

    monkeypatch.setattr(srvmod.os, "replace",
                        _one_shot_failure(sess_dir, "mix.wav"))
    code, body = _render(base, sid, {"style": "rock"})
    assert code == 500, (code, body[:300])

    assert not os.path.exists(os.path.join(sess_dir, "mix.wav"))
    assert not os.path.exists(os.path.join(sess_dir, "raw.wav"))
    assert not os.path.exists(os.path.join(sess_dir, "result.json"))
    assert not os.path.exists(os.path.join(sess_dir, "mixplan.json"))
    _no_leftovers(sess_dir)
    assert _state(base, sid)["has_mix"] is False


def test_the_previous_mix_rotates_once_per_successful_run(srv):
    """Two styles, two mixes: the first one has to stay comparable."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    sess_dir = _session_dir(root, sid)

    code, body = _render(base, sid, {"style": "rock"})
    assert code == 200, body[:300]
    first_mix = _read(os.path.join(sess_dir, "mix.wav"))
    assert not os.path.exists(os.path.join(sess_dir, "mix_prev.wav"))

    code, body = _render(base, sid, {"style": "pop_2020s"})
    assert code == 200, body[:300]
    second_mix = _read(os.path.join(sess_dir, "mix.wav"))
    assert _read(os.path.join(sess_dir, "mix_prev.wav")) == first_mix
    assert second_mix != first_mix
    # the label of the rotated mix travels with it
    assert _read(os.path.join(sess_dir, "prev_style.txt")) == b"rock"
    _no_leftovers(sess_dir)


def test_the_export_cache_survives_a_failed_render(srv, monkeypatch):
    """A failed export must not prune the cache the last good render left."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "export_stems": True})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)
    processed = os.path.join(sess_dir, "processed")
    before = sorted(os.listdir(processed))
    assert before

    monkeypatch.setattr(srvmod.os, "replace",
                        _one_shot_failure(sess_dir, "mixplan.json"))
    code, body = _render(base, sid, {"style": "pop_2020s",
                                     "export_stems": True})
    assert code == 500, (code, body[:300])
    assert sorted(os.listdir(processed)) == before
    _no_leftovers(sess_dir)


def test_a_render_without_export_leaves_no_stale_exports_in_the_result(srv):
    """export=false keeps the old cache on disk but must not advertise it."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "export_stems": True})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)
    processed = os.path.join(sess_dir, "processed")
    before = sorted(os.listdir(processed))
    assert before

    code, body = _render(base, sid, {"style": "rock", "export_stems": False})
    assert code == 200, body[:300]
    result = json.loads(_read(os.path.join(sess_dir, "result.json")))
    assert result["exported"] == []
    assert result.get("export_summary") is None
    # the previous cache is still on disk: that is the documented behaviour
    assert sorted(os.listdir(processed)) == before


# ----------------------------------------------- the publication, on its own

def _old_session(tmp_path) -> Session:
    """A session that already holds one complete render."""
    root = str(tmp_path)
    sess = Session(root, sid="s1")
    sess.dir = os.path.join(root, "s1")
    sess.stems_dir = os.path.join(sess.dir, "stems")
    os.makedirs(sess.stems_dir, exist_ok=True)
    sess.mix_path = os.path.join(sess.dir, "mix.wav")
    sess.raw_path = os.path.join(sess.dir, "raw.wav")
    with open(sess.mix_path, "wb") as fh:
        fh.write(b"OLD-MIX")
    with open(sess.raw_path, "wb") as fh:
        fh.write(b"OLD-RAW")
    with open(os.path.join(sess.dir, "mixplan.json"), "wb") as fh:
        fh.write(b'{"old": true}')
    with open(os.path.join(sess.dir, "report.md"), "wb") as fh:
        fh.write(b"OLD-REPORT")
    with open(os.path.join(sess.dir, "result.json"), "wb") as fh:
        fh.write(b'{"old": true}')
    sess.controls = {"style": "rock"}
    sess.result = {"old": True}
    sess.save()
    return sess


def _stage(sess, mix=b"NEW-MIX", raw=b"NEW-RAW", plan=b'{"new": true}',
           report=b"NEW-REPORT", processed=None) -> str:
    """A staged render, as the child would leave it."""
    stage = srvmod._new_staging_dir(sess.dir)
    with open(os.path.join(stage, "mix.wav"), "wb") as fh:
        fh.write(mix)
    with open(os.path.join(stage, "raw.wav"), "wb") as fh:
        fh.write(raw)
    with open(os.path.join(stage, "mixplan.json"), "wb") as fh:
        fh.write(plan)
    with open(os.path.join(stage, "report.md"), "wb") as fh:
        fh.write(report)
    if processed is not None:
        out = os.path.join(stage, "processed")
        os.makedirs(out, exist_ok=True)
        for name, blob in processed.items():
            with open(os.path.join(out, name), "wb") as fh:
                fh.write(blob)
    return stage


def _publish(sess, stage, **kwargs):
    options = dict(prev_style="rock", reverb_amount=1.0)
    options.update(kwargs)
    return _publish_render(
        sess, stage, {"style": {"id": "pop_2020s"}, "exported": []},
        {"style": "pop_2020s"}, **options
    )


def test_a_mid_publication_failure_rolls_the_session_back(tmp_path,
                                                          monkeypatch):
    """A rename that fails halfway must leave the old bytes in place."""
    sess = _old_session(tmp_path)
    stage = _stage(sess)
    old_mix = _read(sess.mix_path)
    old_result = _read(os.path.join(sess.dir, "result.json"))
    old_manifest = _read(os.path.join(sess.dir, "session.json"))

    real_replace = os.replace
    seen = {"n": 0}

    def failing_replace(src, dst, *args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 4:      # the fourth rename of the publication
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(srvmod.os, "replace", failing_replace)
    with pytest.raises(PublishError):
        _publish(sess, stage)

    assert _read(sess.mix_path) == old_mix
    assert _read(os.path.join(sess.dir, "raw.wav")) == b"OLD-RAW"
    assert _read(os.path.join(sess.dir, "mixplan.json")) == b'{"old": true}'
    assert _read(os.path.join(sess.dir, "report.md")) == b"OLD-REPORT"
    assert _read(os.path.join(sess.dir, "result.json")) == old_result
    assert _read(os.path.join(sess.dir, "session.json")) == old_manifest
    assert not os.path.exists(os.path.join(sess.dir, "mix_prev.wav"))
    # the session still describes the render it did publish
    assert sess.controls == {"style": "rock"}
    assert sess.result == {"old": True}
    _no_leftovers(sess.dir)


def test_a_rollback_that_cannot_finish_keeps_the_backups(tmp_path,
                                                         monkeypatch):
    """When the disk keeps refusing, the old files must stay recoverable."""
    sess = _old_session(tmp_path)
    stage = _stage(sess)

    # this one keeps failing, so the rollback cannot use it either
    monkeypatch.setattr(srvmod.os, "replace",
                        _always_fail(sess.dir, "mixplan.json"))
    with pytest.raises(PublishError) as excinfo:
        _publish(sess, stage)
    message = str(excinfo.value)
    assert "rollback did not complete" in message
    assert srvmod.BACKUP_PREFIX in message

    # the file that could not be put back is still in the backup folder,
    # where a person (or the next run) can get it
    backups = [n for n in os.listdir(sess.dir)
               if n.startswith(srvmod.BACKUP_PREFIX)]
    assert len(backups) == 1
    kept = os.path.join(sess.dir, backups[0], "mixplan.json")
    assert _read(kept) == b'{"old": true}'
    # the mix that could be put back was put back
    assert _read(sess.mix_path) == b"OLD-MIX"


def test_a_failed_first_publication_leaves_no_output(tmp_path, monkeypatch):
    """Nothing existed before: a failed publication must not invent something."""
    sess = _old_session(tmp_path)
    for name in ("mix.wav", "raw.wav", "mixplan.json", "report.md",
                 "result.json"):
        os.unlink(os.path.join(sess.dir, name))
    sess.mix_path = None
    sess.raw_path = None
    stage = _stage(sess)

    monkeypatch.setattr(srvmod.os, "replace",
                        _one_shot_failure(sess.dir, "mix.wav"))
    with pytest.raises(PublishError):
        _publish(sess, stage)

    assert not os.path.exists(os.path.join(sess.dir, "mix.wav"))
    assert not os.path.exists(os.path.join(sess.dir, "raw.wav"))
    assert not os.path.exists(os.path.join(sess.dir, "mixplan.json"))
    assert not os.path.exists(os.path.join(sess.dir, "result.json"))
    _no_leftovers(sess.dir)


def test_a_successful_publication_rotates_the_previous_mix_once(tmp_path):
    """The old mix becomes the previous mix, and only once."""
    sess = _old_session(tmp_path)
    old_mix = _read(sess.mix_path)
    stage = _stage(sess)
    _publish(sess, stage)

    assert _read(sess.mix_path) == b"NEW-MIX"
    assert _read(os.path.join(sess.dir, "mix_prev.wav")) == old_mix
    assert _read(os.path.join(sess.dir, "raw.wav")) == b"NEW-RAW"
    assert json.loads(_read(os.path.join(sess.dir, "result.json")))["style"] == \
        {"id": "pop_2020s"}
    manifest = json.loads(_read(os.path.join(sess.dir, "session.json")))
    assert manifest["has_mix"] is True
    assert manifest["controls"] == {"style": "pop_2020s"}
    _no_leftovers(sess.dir)


def test_a_failed_publication_keeps_the_export_cache(tmp_path, monkeypatch):
    """The generated export cache is only replaced by a successful render."""
    sess = _old_session(tmp_path)
    published = os.path.join(sess.dir, "processed")
    os.makedirs(published, exist_ok=True)
    with open(os.path.join(published, "kick.wav"), "wb") as fh:
        fh.write(b"OLD-EXPORT")
    stage = _stage(sess, processed={"kick.wav": b"NEW-EXPORT"})

    monkeypatch.setattr(srvmod.os, "replace",
                        _one_shot_failure(sess.dir, "mixplan.json"))
    with pytest.raises(PublishError):
        _publish(sess, stage)
    assert _read(os.path.join(published, "kick.wav")) == b"OLD-EXPORT"

    # and a successful publication replaces it
    stage = _stage(sess, processed={"kick.wav": b"NEW-EXPORT"})
    _publish(sess, stage)
    assert _read(os.path.join(published, "kick.wav")) == b"NEW-EXPORT"


def test_a_partial_write_is_never_publishable(tmp_path):
    """A writer that dies half way must not leave a file that looks done."""
    target = os.path.join(str(tmp_path), "mix.wav")

    def half_written(path):
        with open(path, "wb") as fh:
            fh.write(b"17 bytes")
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError):
        _staged_write(target, half_written)
    assert not os.path.exists(target)
    assert os.listdir(str(tmp_path)) == []


def test_the_export_cache_is_pruned_only_of_what_the_server_generated(tmp_path):
    """Generated files go; anything else in the folder stays."""
    sess = _old_session(tmp_path)
    published = os.path.join(sess.dir, "processed")
    os.makedirs(published, exist_ok=True)
    with open(os.path.join(published, "stale.wav"), "wb") as fh:
        fh.write(b"STALE")
    with open(os.path.join(published, "keep.bin"), "wb") as fh:
        fh.write(b"KEEP")
    stage = _stage(sess, processed={"kick.wav": b"NEW-EXPORT"})
    _publish(sess, stage)
    assert not os.path.exists(os.path.join(published, "stale.wav"))
    assert _read(os.path.join(published, "kick.wav")) == b"NEW-EXPORT"
    assert _read(os.path.join(published, "keep.bin")) == b"KEEP"


def test_a_persistence_failure_is_not_a_silent_success(tmp_path, monkeypatch):
    """A manifest that cannot be written is a failed render, not a quiet one."""
    sess = _old_session(tmp_path)
    stage = _stage(sess)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sess, "write_manifest", boom)
    with pytest.raises(PublishError):
        _publish(sess, stage)
    # nothing was published, and the session still describes the old render
    assert _read(sess.mix_path) == b"OLD-MIX"
    assert _read(os.path.join(sess.dir, "mixplan.json")) == b'{"old": true}'
    assert sess.controls == {"style": "rock"}
    assert sess.result == {"old": True}


# -------------------------------------------------- cancel and a dead child

class _FakeConn:
    """One end of the pipe between the parent and the fake child."""

    def __init__(self, inbox, ctx=None):
        self._inbox = inbox
        self._ctx = ctx
        self._child_done = False

    def send(self, msg):
        self._inbox.append(msg)

    def poll(self, timeout=0.0):
        # the fake child runs when the parent first waits for it, which is
        # after the parent has reset its own state
        ctx = self._ctx
        if ctx is not None and ctx.job is not None and not ctx.started:
            ctx.started = True
            payload = ctx.job[1][0]
            try:
                ctx.script(payload, self)
            except BaseException:  # noqa: BLE001 - the child died mid-render
                self._child_done = True
                return True
            self._child_done = True
        return bool(self._inbox) or self._child_done

    def recv(self):
        if not self._inbox:
            raise EOFError("the child died")
        return self._inbox.pop(0)

    def close(self):
        pass


class _FakeProc:
    def __init__(self):
        self.alive = False

    def start(self):
        self.alive = True

    def join(self, timeout=None):
        self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False


class _FakeCtx:
    """A multiprocessing context whose child runs inside this process."""

    def __init__(self, script):
        self.script = script
        self.job = None
        self.started = False

    def Pipe(self, duplex=False):
        inbox = []
        return _FakeConn(inbox, self), _FakeConn(inbox)

    def Process(self, target, args, daemon=True):
        self.job = (target, args)
        return _FakeProc()


class _FakeHandler:
    """Just enough of Handler for _run_isolated."""

    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, code, obj):
        self.status = code
        self.body = obj
        return obj

    def _error(self, code, msg):
        self.status = code
        self.body = {"error": msg}
        return self.body


def _staging_worker(payload, conn, mix=b"FAKE-MIX"):
    """A child that stages a render and reports it as done."""
    stage = payload["stage_dir"]
    conn.send({"progress": {"stage": "rendering", "done": 0, "total": 0}})
    with open(os.path.join(stage, "mix.wav"), "wb") as fh:
        fh.write(mix)
    with open(os.path.join(stage, "raw.wav"), "wb") as fh:
        fh.write(b"FAKE-RAW")
    with open(os.path.join(stage, "mixplan.json"), "wb") as fh:
        fh.write(b'{"fake": true}')
    with open(os.path.join(stage, "report.md"), "wb") as fh:
        fh.write(b"FAKE-REPORT")
    conn.send({"ok": True, "payload": {"style": {"id": "pop_2020s"},
                                       "prev_style": "rock",
                                       "exported": []}})


def test_a_cancel_before_the_commit_does_not_publish(tmp_path, monkeypatch):
    """A cancel that arrives while the child works must not publish anything."""
    sess = _old_session(tmp_path)
    old_mix = _read(sess.mix_path)

    def script(payload, conn):
        sess.cancel_requested = True     # the user hit cancel mid-render
        _staging_worker(payload, conn)

    monkeypatch.setattr(srvmod.multiprocessing, "get_context",
                        lambda name: _FakeCtx(script))
    handler = _FakeHandler()
    srvmod.Handler._run_isolated(handler, sess, {"sess_dir": sess.dir},
                                 {"style": "pop_2020s"})

    assert handler.status == 499, handler.body
    assert _read(sess.mix_path) == old_mix
    assert not os.path.exists(os.path.join(sess.dir, "mix_prev.wav"))
    _no_leftovers(sess.dir)


def test_a_render_that_fails_in_the_child_leaves_the_previous_render(tmp_path,
                                                                   monkeypatch):
    """A render that reports an error has published nothing at all."""
    sess = _old_session(tmp_path)
    old_mix = _read(sess.mix_path)

    def script(payload, conn):
        conn.send({"ok": False, "error": "PlanError: every stem is muted",
                   "kind": "PlanError"})

    monkeypatch.setattr(srvmod.multiprocessing, "get_context",
                        lambda name: _FakeCtx(script))
    handler = _FakeHandler()
    srvmod.Handler._run_isolated(handler, sess, {"sess_dir": sess.dir},
                                 {"style": "pop_2020s"})

    assert handler.status == 400, handler.body
    assert _read(sess.mix_path) == old_mix
    assert _read(os.path.join(sess.dir, "mixplan.json")) == b'{"old": true}'
    assert sess.controls == {"style": "rock"}
    _no_leftovers(sess.dir)


def test_a_child_that_dies_mid_render_publishes_nothing(tmp_path, monkeypatch):
    """The child owns no part of the commit: its death cannot corrupt a mix."""
    sess = _old_session(tmp_path)
    old_mix = _read(sess.mix_path)

    def script(payload, conn):
        with open(os.path.join(payload["stage_dir"], "mix.wav"), "wb") as fh:
            fh.write(b"17 bytes")
        raise RuntimeError("the child died")

    monkeypatch.setattr(srvmod.multiprocessing, "get_context",
                        lambda name: _FakeCtx(script))
    handler = _FakeHandler()
    srvmod.Handler._run_isolated(handler, sess, {"sess_dir": sess.dir},
                                 {"style": "pop_2020s"})

    assert handler.status == 500, handler.body
    assert _read(sess.mix_path) == old_mix
    _no_leftovers(sess.dir)


def test_a_successful_render_publishes_through_the_parent(tmp_path,
                                                          monkeypatch):
    """The happy path of the same route, without a real child."""
    sess = _old_session(tmp_path)
    old_mix = _read(sess.mix_path)

    def script(payload, conn):
        _staging_worker(payload, conn)

    monkeypatch.setattr(srvmod.multiprocessing, "get_context",
                        lambda name: _FakeCtx(script))
    handler = _FakeHandler()
    srvmod.Handler._run_isolated(handler, sess, {"sess_dir": sess.dir},
                                 {"style": "pop_2020s"})

    assert handler.status == 200, handler.body
    assert _read(sess.mix_path) == b"FAKE-MIX"
    assert _read(os.path.join(sess.dir, "mix_prev.wav")) == old_mix
    assert sess.controls == {"style": "pop_2020s"}
    _no_leftovers(sess.dir)


# ------------------------------------------------- paths that are not ours

def test_a_linked_export_folder_is_refused(tmp_path):
    """A processed folder that points outside the session is never written."""
    sess = _old_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "01_Drums.wav").write_bytes(b"OUTSIDE-DRUMS")
    (outside / "keep.txt").write_bytes(b"OUTSIDE-KEEP")
    os.symlink(str(outside), os.path.join(sess.dir, "processed"))
    stage = _stage(sess, processed={"01_Drums.wav": b"NEW-EXPORT"})

    with pytest.raises(PublishError):
        _publish(sess, stage)

    # nothing outside the session was touched
    assert (outside / "01_Drums.wav").read_bytes() == b"OUTSIDE-DRUMS"
    assert (outside / "keep.txt").read_bytes() == b"OUTSIDE-KEEP"
    # and the session still holds the result it had
    assert _read(sess.mix_path) == b"OLD-MIX"


def test_a_folder_where_an_export_file_belongs_is_refused(tmp_path):
    """A folder is the user's; a render must not replace it with a file."""
    sess = _old_session(tmp_path)
    processed = os.path.join(sess.dir, "processed")
    os.makedirs(processed)
    folder = os.path.join(processed, "01_Drums.wav")
    os.makedirs(folder)
    with open(os.path.join(folder, "sentinel.txt"), "wb") as fh:
        fh.write(b"SENTINEL")
    stage = _stage(sess, processed={"01_Drums.wav": b"NEW-EXPORT"})

    with pytest.raises(PublishError):
        _publish(sess, stage)

    assert os.path.isdir(folder)
    assert _read(os.path.join(folder, "sentinel.txt")) == b"SENTINEL"
    assert _read(sess.mix_path) == b"OLD-MIX"


def test_a_linked_output_is_refused(tmp_path):
    """A link in the way of an output is not written through."""
    sess = _old_session(tmp_path)
    outside = tmp_path / "outside-mix.wav"
    outside.write_bytes(b"OUTSIDE-MIX")
    os.unlink(sess.mix_path)
    os.symlink(str(outside), sess.mix_path)
    stage = _stage(sess)

    with pytest.raises(PublishError):
        _publish(sess, stage)
    assert outside.read_bytes() == b"OUTSIDE-MIX"
    assert _read(os.path.join(sess.dir, "raw.wav")) == b"OLD-RAW"


def test_a_dangling_link_output_is_refused(tmp_path):
    """A link to nowhere is not followed, and not quietly deleted."""
    sess = _old_session(tmp_path)
    os.unlink(sess.mix_path)
    os.symlink(str(tmp_path / "nowhere.wav"), sess.mix_path)
    stage = _stage(sess)

    with pytest.raises(PublishError):
        _publish(sess, stage)
    assert os.path.islink(sess.mix_path)
    assert not (tmp_path / "nowhere.wav").exists()


def test_a_render_without_a_comparison_keeps_the_old_set(tmp_path):
    """A render that expected a comparison but has none must not publish."""
    sess = _old_session(tmp_path)
    comparison = os.path.join(sess.dir, "reference_matched.wav")
    with open(comparison, "wb") as fh:
        fh.write(b"OLD-COMPARISON")
    stage = _stage(sess)

    with pytest.raises(PublishError):
        _publish(sess, stage, expect_reference=True)
    # the old comparison was matched to the old mix: it stays with it
    assert _read(comparison) == b"OLD-COMPARISON"
    assert _read(sess.mix_path) == b"OLD-MIX"

    # a render without a reference leaves the unused comparison alone
    stage = _stage(sess)
    _publish(sess, stage, expect_reference=False)
    assert _read(comparison) == b"OLD-COMPARISON"
    assert _read(sess.mix_path) == b"NEW-MIX"


def test_a_failed_comparison_write_fails_the_render(tmp_path, monkeypatch):
    """A comparison that cannot be written is not published as if it were."""
    sess_dir = tmp_path / "sess"
    stems = sess_dir / "stems"
    stems.mkdir(parents=True)
    (stems / "kick.wav").write_bytes(_wav_bytes(60.0))
    (stems / "lead.wav").write_bytes(_wav_bytes(440.0))
    reference = sess_dir / "reference.wav"
    reference.write_bytes(_wav_bytes(220.0))
    stage = srvmod._new_staging_dir(str(sess_dir))

    payload = {
        "sess_dir": str(sess_dir),
        "stage_dir": stage,
        "stems_dir": str(stems),
        "roles": {},
        "style": "pop_2020s",
        "intensity": 1.0,
        "balance_preserve": None,
        "source_balance_profile": "auto",
        "blend": 0.7,
        "reference_path": str(reference),
        "trims": {},
        "muted": [],
        "pans": {},
        "export_stems": False,
        "target_lufs": None,
        "true_peak_db": None,
        "reverb_amount": None,
    }

    class _Conn:
        def __init__(self):
            self.messages = []

        def send(self, msg):
            self.messages.append(msg)

        def close(self):
            pass

    # the reference analysis is not what this test is about
    monkeypatch.setattr(srvmod.refmod, "analyse", lambda path: None)
    real_write = srvmod.render.write_wav24

    def failing_write(path, audio, sr):
        if "reference_matched" in str(path):
            raise OSError(28, "No space left on device")
        return real_write(path, audio, sr)

    monkeypatch.setattr(srvmod.render, "write_wav24", failing_write)

    conn = _Conn()
    srvmod._worker(payload, conn)
    assert conn.messages
    final = conn.messages[-1]
    assert final["ok"] is False, final
    # nothing that looks like a comparison was left behind
    assert not os.path.exists(os.path.join(stage, "reference_matched.wav"))
    assert not [n for n in os.listdir(stage) if n.endswith(".part")]


def test_a_state_change_during_the_commit_cannot_save_new_controls(srv,
                                                                    monkeypatch):
    """A request that arrives mid-commit must not write the new controls."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "target_lufs": -20.0})
    assert code == 200, body[:300]
    sess_dir = _session_dir(root, sid)
    assert json.loads(_read(os.path.join(sess_dir, "session.json")))["controls"] \
        ["target_lufs"] == -20.0

    # pause the publication right before the plan is replaced
    reached = threading.Event()
    release = threading.Event()
    real_replace = os.replace
    state = {"paused": False}

    def pausing_replace(src, dst, *args, **kwargs):
        if (not state["paused"]
                and os.path.dirname(str(dst)) == sess_dir
                and os.path.basename(str(dst)) == "mixplan.json"):
            state["paused"] = True
            reached.set()
            release.wait(60)
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(srvmod.os, "replace", pausing_replace)

    render_result = {}

    def run_render():
        render_result["code"], render_result["body"] = _render(
            base, sid, {"style": "pop_2020s", "target_lufs": -17.0}
        )

    pan_result = {}

    def run_pan():
        code, body, _ = _req(
            base, f"/api/session/{sid}/pan",
            data=json.dumps({"track_id": "kick", "pan": 0.5}).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )
        pan_result["code"] = code
        pan_result["body"] = body

    t = threading.Thread(target=run_render)
    t.start()
    assert reached.wait(120), "the publication never reached the plan"

    p = threading.Thread(target=run_pan)
    p.start()
    p.join(30)

    # the edit is refused, not queued into the commit
    assert pan_result.get("code") == 409, pan_result
    manifest = json.loads(_read(os.path.join(sess_dir, "session.json")))
    assert manifest["controls"]["target_lufs"] == -20.0

    release.set()
    t.join(120)
    p.join(120)

    assert render_result["code"] == 500, render_result["body"][:200]
    # the controls on disk are the ones the session still holds: the failed
    # render's controls were never exposed
    manifest = json.loads(_read(os.path.join(sess_dir, "session.json")))
    assert manifest["controls"]["target_lufs"] == -20.0
    _no_leftovers(sess_dir)


def test_a_rollback_that_cannot_restore_keeps_the_only_copy(tmp_path,
                                                            monkeypatch):
    """When the current mix cannot be put back, its only copy stays."""
    sess = _old_session(tmp_path)
    with open(os.path.join(sess.dir, "mix_prev.wav"), "wb") as fh:
        fh.write(b"OLDER-PREVIOUS")
    old_current = _read(sess.mix_path)
    stage = _stage(sess)

    real_replace = os.replace
    state = {"mix_published": False}

    def failing_replace(src, dst, *args, **kwargs):
        parent = os.path.dirname(str(dst))
        name = os.path.basename(str(dst))
        # the manifest cannot be published, and the current mix cannot be
        # put back afterwards
        if parent == sess.dir and name == "session.json":
            raise OSError(28, "No space left on device")
        if parent == sess.dir and name == "mix.wav":
            if state["mix_published"]:
                raise OSError(28, "No space left on device")
            state["mix_published"] = True
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(srvmod.os, "replace", failing_replace)
    with pytest.raises(PublishError) as excinfo:
        _publish(sess, stage)

    message = str(excinfo.value)
    assert "rollback did not complete" in message
    # the mix that was current is still there: its only copy was not deleted
    # to make the rollback look complete
    kept = os.path.join(sess.dir, "mix_prev.wav")
    assert _read(kept) == old_current
    # and the older previous is still in the backup folder
    backups = [n for n in os.listdir(sess.dir)
               if n.startswith(srvmod.BACKUP_PREFIX)]
    assert len(backups) == 1
    assert _read(os.path.join(sess.dir, backups[0], "mix_prev.wav")) \
        == b"OLDER-PREVIOUS"
    # the error names where the only copy is
    assert kept in message


def test_an_expected_comparison_keeps_the_old_set_when_it_is_missing(tmp_path):
    """A comparison that is a folder is never removed by a publication."""
    sess = _old_session(tmp_path)
    comparison = os.path.join(sess.dir, "reference_matched.wav")
    os.makedirs(comparison)
    with open(os.path.join(comparison, "sentinel.txt"), "wb") as fh:
        fh.write(b"SENTINEL")
    stage = _stage(sess)

    with pytest.raises(PublishError):
        _publish(sess, stage, expect_reference=True)

    assert os.path.isdir(comparison)
    assert _read(os.path.join(comparison, "sentinel.txt")) == b"SENTINEL"
    assert _read(sess.mix_path) == b"OLD-MIX"


def test_a_download_during_a_commit_gets_the_published_comparison(srv,
                                                                   monkeypatch):
    """The comparison and the download have to be the same moment."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body, _ = _req(base, f"/api/session/{sid}/reference",
                         data=_wav_bytes(220.0), method="POST",
                         headers={"X-Filename": "reference.wav"})
    assert code == 200, body[:200]
    sess_dir = _session_dir(root, sid)

    reached = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def pausing_replace(src, dst, *args, **kwargs):
        if (os.path.dirname(str(dst)) == sess_dir
                and os.path.basename(str(dst)) == "reference_matched.wav"):
            reached.set()
            release.wait(60)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(srvmod.os, "replace", pausing_replace)

    render_result = {}

    def run_render():
        render_result["code"], render_result["body"] = _render(
            base, sid, {"style": "pop_2020s"}
        )

    t = threading.Thread(target=run_render)
    t.start()
    assert reached.wait(120), "the publication never reached the comparison"

    download = {}

    def run_download():
        code, body, _ = _req(base, f"/api/session/{sid}/audio/reference")
        download["code"] = code
        download["body"] = body

    d = threading.Thread(target=run_download)
    d.start()
    # the download waits for the commit instead of being served the old choice
    d.join(1.0)
    assert not download, "the download was served during the commit"

    release.set()
    t.join(180)
    d.join(60)

    assert render_result["code"] == 200, render_result["body"][:200]
    assert download["code"] == 200, download.get("body", b"")[:200]
    matched = os.path.join(sess_dir, "reference_matched.wav")
    assert os.path.exists(matched)
    # the matched comparison, not the raw reference
    assert download["body"] == _read(matched)


def test_an_edit_during_a_successful_commit_is_refused(srv, monkeypatch):
    """A refused edit cannot turn the answer of a good render into null."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    sess_dir = _session_dir(root, sid)

    reached = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def pausing_replace(src, dst, *args, **kwargs):
        if (os.path.dirname(str(dst)) == sess_dir
                and os.path.basename(str(dst)) == "mixplan.json"):
            reached.set()
            release.wait(60)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(srvmod.os, "replace", pausing_replace)

    render_result = {}

    def run_render():
        render_result["code"], render_result["body"] = _render(
            base, sid, {"style": "rock", "target_lufs": -17.0}
        )

    t = threading.Thread(target=run_render)
    t.start()
    assert reached.wait(120), "the publication never reached the plan"

    pan_result = {}

    def run_pan():
        code, body, _ = _req(
            base, f"/api/session/{sid}/pan",
            data=json.dumps({"track_id": "kick", "pan": 0.5}).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )
        pan_result["code"] = code

    p = threading.Thread(target=run_pan)
    p.start()
    p.join(30)
    assert pan_result.get("code") == 409, pan_result

    release.set()
    t.join(180)
    p.join(60)

    assert render_result["code"] == 200, render_result["body"][:200]
    answer = json.loads(render_result["body"])
    assert answer["style"]["id"] == "rock"
    assert answer["target_lufs"] == -17.0
    # the session still shows the render that was published
    assert _state(base, sid)["has_mix"] is True
    assert _state(base, sid)["controls"]["target_lufs"] == -17.0
