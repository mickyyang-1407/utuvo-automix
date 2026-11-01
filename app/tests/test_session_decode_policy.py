"""A GUI session keeps the decoder it was made with.

``float32_v2`` is only for new imports.  A session saved before R47 was
imported, role-checked, drawn and rendered with ``float32_v1``; reopening it
and pressing Run must not quietly decode its mono files 3.01 dB louder than
the mix it already holds.
"""

from __future__ import annotations

import io
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from automix import analyze
from automix import server as srvmod
from automix.server import Handler, Session

needs_ffmpeg = pytest.mark.skipif(not analyze._ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("sessions"))
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
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _audio(channels: int, freq: float, fmt: str) -> bytes:
    t = np.arange(48000 * 2) / 48000
    cols = [0.3 * np.sin(2 * np.pi * freq * t)] * channels
    buf = io.BytesIO()
    sf.write(buf, np.stack(cols, axis=1), 48000, subtype="PCM_16", format=fmt)
    return buf.getvalue()


def _session_with_stems(base) -> str:
    code, body = _req(base, "/api/session", data=b"", method="POST")
    assert code == 200
    sid = json.loads(body)["id"]
    for name, data in (("vocal_lead.flac", _audio(1, 440.0, "FLAC")),
                       ("bass.wav", _audio(2, 55.0, "WAV"))):
        code, _ = _req(base, f"/api/session/{sid}/stem", data=data, method="POST",
                       headers={"X-Filename": name, "Content-Type": "application/octet-stream"})
        assert code == 200
    return sid


def _drop_policy_from_manifest(root, sid, value=None):
    path = os.path.join(root, sid, Session.MANIFEST)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("decode_policy", None)
    if value is not None:
        data["decode_policy"] = value
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _restart(root, sid):
    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    assert srvmod._load_sessions(root) >= 1


def _run(base, sid) -> dict:
    code, body = _req(base, f"/api/session/{sid}/run", data=json.dumps({"style": "pop_2020s"}).encode(),
                      method="POST", headers={"Content-Type": "application/json"})
    assert code == 200, body
    code, body = _req(base, f"/api/session/{sid}/plan")
    assert code == 200, body
    return json.loads(body)


# ---------------------------------------------------------------- the manifest

def test_a_new_session_uses_the_current_default_and_saves_it(tmp_path):
    sess = Session(str(tmp_path))
    assert sess.decode_policy == analyze.DEFAULT_DECODE_POLICY == "float32_v2"
    assert sess.manifest_data()["decode_policy"] == "float32_v2"
    sess.save()
    assert Session.load(str(tmp_path), sess.id).decode_policy == "float32_v2"


@pytest.mark.parametrize("saved", [None, "bogus", 1, True, ["float32_v2"], "FLOAT32_V2"])
def test_a_session_without_a_valid_policy_keeps_float32_v1(tmp_path, saved):
    sess = Session(str(tmp_path))
    sess.save()
    _drop_policy_from_manifest(str(tmp_path), sess.id, saved)
    assert Session.load(str(tmp_path), sess.id).decode_policy == "float32_v1"


@pytest.mark.parametrize("policy", list(analyze.DECODE_POLICIES))
def test_a_recorded_policy_is_kept_as_recorded(tmp_path, policy):
    sess = Session(str(tmp_path))
    sess.decode_policy = policy
    sess.save()
    assert Session.load(str(tmp_path), sess.id).decode_policy == policy


# ------------------------------------------------------- every load in the GUI

@needs_ffmpeg
def test_roles_and_peaks_decode_with_the_sessions_policy(srv, monkeypatch):
    base, root = srv
    sid = _session_with_stems(base)
    _drop_policy_from_manifest(root, sid)
    _restart(root, sid)
    seen = []
    loads, files = analyze.load_stems, analyze.load_stem_file

    def spy_stems(stem_dir, decode_policy=analyze.DEFAULT_DECODE_POLICY):
        seen.append(decode_policy)
        return loads(stem_dir, decode_policy)

    def spy_file(path, decode_policy=analyze.DEFAULT_DECODE_POLICY):
        seen.append(decode_policy)
        return files(path, decode_policy)

    monkeypatch.setattr(analyze, "load_stems", spy_stems)
    monkeypatch.setattr(analyze, "load_stem_file", spy_file)
    assert _req(base, f"/api/session/{sid}/roles")[0] == 200
    sess = srvmod._sessions[sid]
    srvmod._peaks(os.path.join(sess.stems_dir, "vocal_lead.flac"), sess)
    assert seen and set(seen) == {"float32_v1"}


@needs_ffmpeg
def test_an_old_session_renders_with_float32_v1_and_a_new_one_with_v2(srv):
    base, root = srv
    old = _session_with_stems(base)
    _drop_policy_from_manifest(root, old)
    _restart(root, old)
    assert _run(base, old)["source"]["decode_policy"] == "float32_v1"
    # the render publishes the manifest again; the session keeps its policy
    _restart(root, old)
    assert srvmod._sessions[old].decode_policy == "float32_v1"

    new = _session_with_stems(base)
    assert _run(base, new)["source"]["decode_policy"] == "float32_v2"
    _restart(root, new)
    assert srvmod._sessions[new].decode_policy == "float32_v2"
