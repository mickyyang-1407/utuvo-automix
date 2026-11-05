"""Tests for the local web UI server.

The server is a thin shell over the engine, so these tests are mostly about
the shell being honest: styles actually load, uploads land in the session and
nowhere else, a render produces the same numbers the CLI would, and the audio
endpoints serve bytes a browser can actually seek in.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from automix.server import Handler, _safe_name


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
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _wav_bytes(freq=220.0, dur=1.0, sr=48000, amp=0.3) -> bytes:
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    pcm = (np.stack([mono, mono], axis=1) * 32767).astype("<i2")
    import io

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
    # HTTP headers are latin-1, so a raw non-ASCII name is percent-encoded the
    # way the browser client does it.  Names that are already ASCII or already
    # encoded are passed through untouched.
    import urllib.parse

    if any(ord(ch) > 127 for ch in name):
        name = urllib.parse.quote(name)

    return _req(
        base,
        f"/api/session/{sid}/stem",
        data=data,
        method="POST",
        headers={
            "X-Filename": name,
            "Content-Type": "application/octet-stream",
        },
    )


# ------------------------------------------------------------------ static/api

def test_index_is_served(srv):
    base, _ = srv
    code, body, headers = _req(base, "/")
    assert code == 200
    assert "text/html" in headers.get("Content-Type", "")
    assert b"<html" in body.lower()


def test_styles_endpoint_lists_every_preset(srv):
    base, _ = srv
    code, body, _ = _req(base, "/api/styles")
    assert code == 200
    styles = json.loads(body)["styles"]
    ids = {s["id"] for s in styles}
    assert {"pop_2020s", "rock", "lofi", "trap", "acoustic", "cinematic"} <= ids
    assert all(s["name"] for s in styles)


def test_unknown_session_is_404(srv):
    base, _ = srv
    code, body, _ = _req(base, "/api/session/deadbeef/state")
    assert code == 404
    assert "unknown session" in json.loads(body)["error"]


# -------------------------------------------------------------------- uploads

def test_upload_then_state(srv):
    base, _ = srv
    sid = _new_session(base)
    code, body, _ = _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    assert code == 200
    assert json.loads(body)["stems"] == ["kick.wav"]

    code, body, _ = _req(base, f"/api/session/{sid}/state")
    state = json.loads(body)
    assert state["stems"] == ["kick.wav"]
    assert state["has_mix"] is False


def test_upload_rejects_unsupported_extension(srv):
    base, _ = srv
    sid = _new_session(base)
    code, body, _ = _upload(base, sid, "notes.txt", b"hello")
    assert code == 400
    assert "unsupported" in json.loads(body)["error"]


def test_path_traversal_stays_inside_the_session(srv):
    base, root = srv
    sid = _new_session(base)
    code, _, _ = _upload(base, sid, "../../escape.wav", _wav_bytes())
    assert code == 200
    # it must exist inside the session's stems dir, and nowhere above it
    assert os.path.exists(os.path.join(root, sid, "stems", "escape.wav"))
    assert not os.path.exists(os.path.join(root, "escape.wav"))
    assert not os.path.exists(os.path.join(os.path.dirname(root), "escape.wav"))


def test_empty_upload_is_rejected(srv):
    base, _ = srv
    sid = _new_session(base)
    code, body, _ = _upload(base, sid, "kick.wav", b"")
    assert code == 400
    assert "empty" in json.loads(body)["error"]


def test_safe_name_is_a_basename():
    assert _safe_name("a/b/kick.wav") == "kick.wav"
    assert _safe_name("..%2F..%2Fx.wav") == "x.wav"
    with pytest.raises(ValueError):
        _safe_name(".hidden")


# ----------------------------------------------------------------------- run

def test_run_without_stems_is_rejected(srv):
    base, _ = srv
    sid = _new_session(base)
    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "lofi"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 400
    assert "no stems" in json.loads(body)["error"]


def test_full_render_round_trip(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "pop_2020s", "intensity": 1.0}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200, body[:300]
    result = json.loads(body)

    for key in ("lufs", "true_peak_db", "crest_db", "width", "correlation", "tracks"):
        assert key in result
    assert abs(result["lufs"] - result["target_lufs"]) < 1.0
    assert result["true_peak_db"] <= result["ceiling_db"] + 1e-6
    assert {t["role"] for t in result["tracks"]} == {"kick", "bass", "lead_vocal"}

    # the audio, the plan and the receipt must all be fetchable
    for kind in ("mix", "raw"):
        code, data, headers = _req(base, f"/api/session/{sid}/audio/{kind}")
        assert code == 200
        assert data[:4] == b"RIFF"
        assert "audio/wav" in headers.get("Content-Type", "")

    code, plan_body, _ = _req(base, f"/api/session/{sid}/plan")
    assert code == 200
    plan = json.loads(plan_body)
    assert plan["style"]["id"] == "pop_2020s"
    assert len(plan["tracks"]) == 3

    code, report_body, _ = _req(base, f"/api/session/{sid}/report")
    assert code == 200
    assert b"LUFS" in report_body


def test_range_request_is_supported(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "lofi"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    code, body, headers = _req(
        base, f"/api/session/{sid}/audio/mix", headers={"Range": "bytes=0-99"}
    )
    assert code == 206
    assert len(body) == 100
    assert headers.get("Content-Range", "").startswith("bytes 0-99/")


@pytest.fixture
def range_source(srv):
    """A tiny isolated source file: transport tests never need a render."""
    base, root = srv
    sid = _new_session(base)
    payload = _wav_bytes(dur=.01, sr=8000)
    code, _, _ = _upload(base, sid, "range.wav", payload)
    assert code == 200
    return base, f"/api/session/{sid}/audio/stem/range.wav", payload, os.path.join(root, sid, "stems", "range.wav")


def test_suffix_range_returns_exact_last_bytes(range_source):
    base, path, payload, source = range_source
    code, body, headers = _req(base, path, headers={"Range": "bytes=-16"})
    assert code == 206
    assert body == payload[-16:]
    assert headers["Content-Range"] == f"bytes {len(payload)-16}-{len(payload)-1}/{len(payload)}"
    assert headers["Content-Length"] == "16"
    assert headers["Accept-Ranges"] == "bytes"
    with open(source, "rb") as handle:
        assert handle.read() == payload


def test_manual_reverb_http_restore_invalid_inputs_and_wet_to_dry_exports(srv):
    from automix.server import Session
    base,root=srv;sid=_new_session(base)
    assert _upload(base,sid,'vocal_lead.wav',_wav_bytes(440,dur=.3,sr=8000))[0]==200
    assert _upload(base,sid,'fx.wav',_wav_bytes(230,dur=.3,sr=8000))[0]==200
    url=f'/api/session/{sid}'
    def run(options):
        return _req(base,url+'/run',json.dumps(options).encode(),'POST',{'Content-Type':'application/json'})
    code,body,_=run({'reverb_amount':.6,'export_stems':True})
    assert code==200,body
    assert '_reverb.wav' in {e['name'] for e in json.loads(body)['exported']}
    before_mix=_req(base,url+'/audio/mix')[1]
    manifest_path=os.path.join(root,sid,'session.json')
    with open(manifest_path,'rb') as handle:
        before_manifest=handle.read()
    for value in (True,False,float('nan'),float('inf'),-.1,1.1,'0.5',[],{}):
        code,body,_=run({'reverb_amount':value})
        assert code==400 and 'reverb amount' in json.loads(body)['error']
        assert json.loads(_req(base,url+'/state')[1])['reverb_amount']==.6
        assert _req(base,url+'/audio/mix')[1]==before_mix
        with open(manifest_path,'rb') as handle:
            assert handle.read()==before_manifest
    assert Session.load(root,sid).reverb_amount==.6
    code,body,_=run({'reverb_amount':0,'export_stems':True})
    assert code==200,body
    result=json.loads(body)
    assert '_reverb.wav' not in {e['name'] for e in result['exported']}
    assert 'fx.wav' in {e['name'] for e in result['exported']}
    assert _req(base,url+'/processed/file/_reverb.wav')[0]==404
    assert not os.path.exists(os.path.join(root,sid,'processed','_reverb.wav'))
    assert _req(base,url+'/audio/stem/fx.wav')[0]==200
    assert json.loads(_req(base,url+'/state')[1])['reverb_amount']==0
    assert Session.load(root,sid).reverb_amount==0
    assert json.loads(_req(base,url+'/plan')[1])['reverb']['manual_amount']==0
    # An omitted value resumes the saved amount; a failed render cannot replace it.
    code,_,_=run({'style':'not_a_style','reverb_amount':.8})
    assert code==400 and json.loads(_req(base,url+'/state')[1])['reverb_amount']==0
    code,body,_=run({'export_stems':True})
    assert code==200,body
    assert json.loads(_req(base,url+'/plan')[1])['reverb']['manual_amount']==0


@pytest.mark.parametrize("value,start,end", [
    pytest.param("bytes=4-8", 4, 8, id="normal-inclusive"),
    pytest.param("bytes=100-", 100, None, id="open-ended"),
    pytest.param("bytes=-1", -1, None, id="last-byte"),
    pytest.param("bytes=-999999", 0, None, id="oversized-suffix"),
    pytest.param("bytes=4-999999", 4, None, id="end-clamped"),
    pytest.param("bytes=0004-0008", 4, 8, id="leading-zeros"),
    pytest.param("\tByTeS=4-8 \t", 4, 8, id="case-and-outer-ows"),
    pytest.param("bytes= \t4-8 \t", 4, 8, id="range-set-ows"),
    pytest.param("bytes=4-" + "9"*5000, 4, None, id="huge-end"),
    pytest.param("bytes=-" + "9"*5000, 0, None, id="huge-suffix"),
    pytest.param("bytes=" + "0"*5000 + "4-8", 4, 8, id="long-zero-start"),
    pytest.param("bytes=-" + "0"*5000 + "16", -16, None, id="long-zero-suffix"),
])
def test_single_ranges_have_exact_body_and_headers(range_source,value,start,end):
    base, path, payload, source = range_source
    start = start if start >= 0 else len(payload) + start
    end = end if end is not None else len(payload) - 1
    code, body, headers = _req(base, path, headers={"Range": value})
    assert code == 206
    assert body == payload[start:end+1]
    assert headers["Content-Range"] == f"bytes {start}-{end}/{len(payload)}"
    assert int(headers["Content-Length"]) == len(body)
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "audio/wav"
    with open(source, "rb") as handle:
        assert handle.read() == payload


@pytest.mark.parametrize("value", [
    pytest.param("bytes=999-", id="start-beyond-end"),
    pytest.param("bytes=999-999", id="closed-beyond-end"),
    pytest.param("bytes=-0", id="zero-suffix"),
    pytest.param("bytes=" + "9"*5000 + "-", id="huge-start"),
    pytest.param("bytes=" + "9"*5000 + "-" + "9"*5000, id="huge-closed"),
    pytest.param("bytes=-" + "0"*5000, id="long-zero-suffix"),
])
def test_unsatisfiable_ranges_are_416_json(range_source,value):
    base, path, payload, source = range_source
    code, body, headers = _req(base, path, headers={"Range": value})
    assert code == 416
    assert json.loads(body) == {"error": "bad range"}
    assert headers["Content-Range"] == f"bytes */{len(payload)}"
    assert int(headers["Content-Length"]) == len(body)
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Accept-Ranges"] == "bytes"
    with open(source, "rb") as handle:
        assert handle.read() == payload


@pytest.mark.parametrize("value", [
    pytest.param("items=1-2", id="unknown-unit"),
    pytest.param("bytes=bogus", id="non-numeric"),
    pytest.param("bytes=1", id="missing-hyphen"),
    pytest.param("bytes=-", id="missing-both-numbers"),
    pytest.param("bytes=4-2", id="reversed"),
    pytest.param("bytes=+1-2", id="signed-number"),
    pytest.param("bytes=1--2", id="negative-end"),
    pytest.param("bytes=0-1,4-5", id="unsupported-multipart"),
    pytest.param("bytes=1.0-2", id="non-integer"),
    pytest.param("bytes=²-4", id="non-ascii-digit"),
    pytest.param("bytes =1-2", id="invalid-unit-whitespace"),
    pytest.param("bytes=1 - 2", id="invalid-internal-whitespace"),
    pytest.param("bytes=" + "9"*5000 + "-" + "8"*5000, id="huge-reversed"),
])
def test_ignored_ranges_are_normal_full_200(range_source,value):
    base, path, payload, source = range_source
    code, body, headers = _req(base, path, headers={"Range": value})
    assert code == 200 and body == payload
    assert "Content-Range" not in headers
    assert headers["Content-Length"] == str(len(payload))
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "audio/wav"
    with open(source, "rb") as handle:
        assert handle.read() == payload


@pytest.mark.parametrize("value", [None,"bytes=0-0","bytes=-1","bytes=-0","bytes=999-","items=0-1"])
def test_empty_file_ignores_range_and_returns_empty_200(srv,value):
    base, root = srv
    sid = _new_session(base)
    source = os.path.join(root,sid,"stems","empty.wav")
    with open(source,"wb"):
        pass
    headers = {"Range":value} if value is not None else {}
    code,body,headers = _req(base,f"/api/session/{sid}/audio/stem/empty.wav",headers=headers)
    assert code == 200 and body == b""
    assert headers["Content-Length"] == "0"
    assert headers["Accept-Ranges"] == "bytes"
    assert "Content-Range" not in headers
    assert os.path.getsize(source) == 0


def test_uploading_again_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "lofi"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert json.loads(body)["has_mix"] is True

    _upload(base, sid, "pad.wav", _wav_bytes(220.0))
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert json.loads(body)["has_mix"] is False


def test_reference_upload_is_used(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    ref = _wav_bytes(110.0, dur=1.5, amp=0.12)
    code, _, _ = _req(
        base,
        f"/api/session/{sid}/reference",
        data=ref,
        method="POST",
        headers={"X-Filename": "ref.wav", "Content-Type": "application/octet-stream"},
    )
    assert code == 200

    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "cinematic", "blend": 0.8}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200
    result = json.loads(body)
    assert result["reference_passes"] == 2
    assert "reference" in result["style"]
    assert len(result["style"]["reference"]["tone"]) == 31

def test_remove_stem(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))

    code, body, _ = _req(
        base,
        f"/api/session/{sid}/remove",
        data=json.dumps({"name": "bass.wav"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200
    assert json.loads(body)["stems"] == ["kick.wav"]


def test_ab_raw_sum_is_loudness_matched(srv, tmp_path):
    """The 'before' must be as loud as the 'after', or the A/B is a level trick."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))

    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "pop_2020s"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200

    from automix import analyze, dsp

    def lufs_of(kind, tmp_path):
        # NB: the served files are 24-bit, so decode them with the project's
        # reader.  Reading them as int16 (an easy mistake) still returns
        # *a* number, and two equally-wrong numbers can pass a delta test.
        code, data, _ = _req(base, f"/api/session/{sid}/audio/{kind}")
        assert code == 200
        path = tmp_path / f"{kind}.wav"
        path.write_bytes(data)
        audio, sr = analyze.load_stem_file(str(path))
        assert audio.shape[1] == 2
        return dsp.loudness_lufs(audio, sr)

    mix_lufs = lufs_of("mix", tmp_path)
    raw_lufs = lufs_of("raw", tmp_path)
    assert abs(mix_lufs - raw_lufs) < 0.6, f"mix {mix_lufs:.2f} raw {raw_lufs:.2f}"

def test_utf8_filename_round_trips(srv):
    """HTTP headers are latin-1; a CJK stem name must survive the trip."""
    base, _ = srv
    sid = _new_session(base)
    from urllib.parse import quote

    cjk = "音軌 02.wav"
    code, body, _ = _upload(base, sid, quote(cjk), _wav_bytes(330.0))
    assert code == 200
    assert json.loads(body)["stems"] == [cjk]

    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert json.loads(body)["stems"] == [cjk]

def test_roles_endpoint_reports_detection_and_source(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    _upload(base, sid, "Audio 1.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    roles = json.loads(body)["roles"]
    assert roles["kick.wav"]["role"] == "kick"
    assert roles["kick.wav"]["source"] == "keyword"
    assert roles["Audio 1.wav"]["source"] in ("spectral", "keyword")


def test_role_edits_reuse_automatic_analysis_and_keep_complete_snapshots(srv, monkeypatch):
    from automix import server as srvmod

    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    original_load = srvmod.analyze.load_stems
    loads = []

    def counted_load(*args, **kwargs):
        loads.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(srvmod.analyze, "load_stems", counted_load)
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    initial = json.loads(body)
    assert initial["role_revision"] == 0 and len(initial["roles"]) == 2
    for revision, name, role in [(1, "bass.wav", "keys"), (2, "vocal_lead.wav", "backing_vocal")]:
        code, body, _ = _req(base, f"/api/session/{sid}/roles",
                             data=json.dumps({"roles": {name: role}}).encode(), method="POST")
        assert code == 200
        snapshot = json.loads(body)
        assert snapshot["role_revision"] == revision
        assert snapshot["stems"] == initial["stems"]
        assert snapshot["roles"][name]["role"] == role
        assert snapshot["roles"][name]["source"] == "user"
        for filename, detected in initial["roles"].items():
            current = snapshot["roles"][filename]
            assert set(current) == set(detected)
            assert current["metrics"] == detected["metrics"]
            assert current["detected"] == detected["detected"]
            assert current["id"] == detected["id"]
        if revision == 1:
            assert snapshot["roles"]["vocal_lead.wav"] == initial["roles"]["vocal_lead.wav"]
    code, _, _ = _req(base, f"/api/session/{sid}/roles",
                      data=json.dumps({"roles": {"nope.wav": "bass"}}).encode(), method="POST")
    assert code == 400
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    final = json.loads(body)
    assert code == 200 and final["role_revision"] == 2
    assert final["roles"]["bass.wav"]["role"] == "keys"
    assert final["roles"]["vocal_lead.wav"]["role"] == "backing_vocal"
    assert loads == [1], "unchanged source was loaded again for a role-only edit/read"


@pytest.mark.parametrize("change", ["replace", "same_size_preserved_mtime", "rename",
                                    "add", "remove", "metadata", "decode_policy"])
def test_role_analysis_refreshes_for_source_identity_changes(srv, monkeypatch, change):
    from automix import server as srvmod

    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    original_load = srvmod.analyze.load_stems
    loads = []

    def counted_load(*args, **kwargs):
        loads.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(srvmod.analyze, "load_stems", counted_load)
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    before = json.loads(body)
    path = os.path.join(root, sid, "stems", "bass.wav")
    info = os.stat(path)
    if change == "replace":
        replacement = path + ".replacement"
        with open(replacement, "wb") as f:
            f.write(_wav_bytes(55.0, amp=.15))
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.replace(replacement, path)
        assert os.stat(path).st_size == info.st_size
        assert os.stat(path).st_ino != info.st_ino
    elif change == "same_size_preserved_mtime":
        with open(path, "wb") as f:
            f.write(_wav_bytes(55.0, amp=.15))
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        now = os.stat(path)
        assert (now.st_ino, now.st_size, now.st_mtime_ns) == (info.st_ino, info.st_size, info.st_mtime_ns)
        assert now.st_ctime_ns != info.st_ctime_ns
    elif change == "rename":
        os.rename(path, os.path.join(os.path.dirname(path), "kick.wav"))
    elif change == "add":
        assert _upload(base, sid, "kick.wav", _wav_bytes(60.0))[0] == 200
    elif change == "remove":
        assert _req(base, f"/api/session/{sid}/remove",
                    data=json.dumps({"name": "bass.wav"}).encode(), method="POST")[0] == 200
    elif change == "metadata":
        os.chmod(path, info.st_mode ^ 0o100)
        assert os.stat(path).st_ctime_ns != info.st_ctime_ns
    else:
        sess = srvmod._sessions[sid]
        sess.decode_policy = next(p for p in srvmod.analyze.DECODE_POLICIES if p != sess.decode_policy)
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    after = json.loads(body)
    expected = {"bass.wav", "vocal_lead.wav"}
    if change in {"rename", "remove"}:
        expected.remove("bass.wav")
    if change in {"rename", "add"}:
        expected.add("kick.wav")
    assert set(after["stems"]) == set(after["roles"]) == expected
    assert after["role_revision"] == 0
    if change in {"replace", "same_size_preserved_mtime"}:
        assert after["roles"]["bass.wav"]["metrics"] != before["roles"]["bass.wav"]["metrics"]
    if change in {"rename", "add"}:
        assert after["roles"]["kick.wav"]["detected"] == "kick"
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200 and json.loads(body) == after
    assert loads == [1, 1], "changed source must refresh once, then reuse the new snapshot"


def test_role_analysis_failure_does_not_cache_partial_detection(srv, monkeypatch):
    from automix import server as srvmod

    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    original_metrics = srvmod.analyze.quick_metrics
    failed = []

    def fail_once(stem):
        if not failed:
            failed.append(1)
            raise srvmod.analyze.StemError("temporary metric failure")
        return original_metrics(stem)

    monkeypatch.setattr(srvmod.analyze, "quick_metrics", fail_once)
    code, _, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 400
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    assert set(json.loads(body)["roles"]) == {"bass.wav", "vocal_lead.wav"}


def test_role_override_wins_and_survives_a_render(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "Audio 1.wav", _wav_bytes(440.0))
    _upload(base, sid, "pad.wav", _wav_bytes(220.0))

    # force the unknown stem to be a bass
    code, body, _ = _req(
        base,
        f"/api/session/{sid}/roles",
        data=json.dumps({"roles": {"Audio 1.wav": "bass"}}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200
    roles = json.loads(body)["roles"]
    assert roles["Audio 1.wav"]["role"] == "bass"
    assert roles["Audio 1.wav"]["source"] == "user"
    assert roles["Audio 1.wav"]["detected"] != "user"

    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "pop_2020s"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200
    got = {t["id"]: t["role"] for t in json.loads(body)["tracks"]}
    assert got["Audio_1"] == "bass"


def test_roles_list_endpoint(srv):
    base, _ = srv
    code, body, _ = _req(base, "/api/roles")
    assert code == 200
    roles = json.loads(body)["roles"]
    assert "kick" in roles and "lead_vocal" in roles and "other" in roles

def test_a_running_render_can_be_cancelled(srv):
    """A three-minute session takes ~30 s; bailing out must actually stop it."""
    base, _ = srv
    sid = _new_session(base)
    # a small successful mix first: the controls it saved are what the
    # cancelled render below has to leave alone
    _upload(base, sid, "small.wav", _wav_bytes(55.0))
    _upload(base, sid, "small2.wav", _wav_bytes(440.0))
    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "pop_2020s", "balance_preserve": 0.35,
                         "target_lufs": -14.0, "blend": 0.25}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200, body[:200]
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200
    before = json.loads(body)["controls"]
    assert before["balance_preserve"] == 0.35
    assert before["target_lufs"] == -14.0
    # ~40 s of audio so the render is comfortably longer than the test's patience
    _upload(base, sid, "pad.wav", _wav_bytes(220.0, dur=40.0))
    _upload(base, sid, "bass.wav", _wav_bytes(55.0, dur=40.0))

    result = {}

    def run():
        result["code"], result["body"], _ = _req(
            base,
            f"/api/session/{sid}/run",
            data=json.dumps({"style": "cinematic"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    t = threading.Thread(target=run)
    t.start()
    try:
        # give the child a moment to start, then cancel
        for _ in range(40):
            code, body, _ = _req(base, f"/api/session/{sid}/state")
            if json.loads(body)["stems"]:
                break
            time.sleep(0.05)
        time.sleep(0.4)
        code, body, _ = _req(base, f"/api/session/{sid}/cancel", data=b"", method="POST")
        assert code == 200
    finally:
        t.join(180)

    assert result["code"] == 499, (result["code"], result["body"][:200])
    assert "cancelled" in json.loads(result["body"])["error"]
    # a cancelled render is not a successful one: the controls of the last
    # successful mix are still the ones this session holds
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200
    assert json.loads(body)["controls"] == before

def test_progress_is_reported_during_a_render(srv):
    """A 30 s wait with no feedback reads as a freeze; stages must be visible."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "pad.wav", _wav_bytes(220.0, dur=40.0))

    seen = []
    result = {}

    def run():
        result["code"], result["body"], _ = _req(
            base,
            f"/api/session/{sid}/run",
            data=json.dumps({"style": "cinematic"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    t = threading.Thread(target=run)
    t.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            code, body, _ = _req(base, f"/api/session/{sid}/progress")
            if code == 200:
                p = json.loads(body)
                if p.get("progress"):
                    seen.append(p["progress"].get("stage"))
                if not p.get("running") and seen:
                    break
            time.sleep(0.05)
    finally:
        t.join(180)

    assert result["code"] == 200, result["body"][:200]
    assert seen, "no progress was ever reported"
    assert any(st in ("loading", "roles", "measuring", "planning", "rendering", "writing") for st in seen)
    # once finished the endpoint reports idle, and the last stage must have run
    code, body, _ = _req(base, f"/api/session/{sid}/progress")
    assert json.loads(body)["running"] is False

def test_uploads_are_rejected_while_a_render_runs(srv):
    """Changing stems mid-render would have the child working on moved files."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "pad.wav", _wav_bytes(220.0, dur=40.0))

    result = {}

    def run():
        result["code"], _, _ = _req(
            base,
            f"/api/session/{sid}/run",
            data=json.dumps({"style": "cinematic"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    t = threading.Thread(target=run)
    t.start()
    try:
        time.sleep(0.5)
        code, body, _ = _upload(base, sid, "kick.wav", _wav_bytes(60.0))
        assert code == 409, (code, body[:200])
        assert "render is running" in json.loads(body)["error"]
    finally:
        t.join(180)

def test_reference_ab_is_loudness_matched(srv, tmp_path):
    """A/B against a reference must compare sound, not level."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))
    # a deliberately quiet reference
    _req(
        base,
        f"/api/session/{sid}/reference",
        data=_wav_bytes(110.0, dur=2.0, amp=0.05),
        method="POST",
        headers={"X-Filename": "ref.wav", "Content-Type": "application/octet-stream"},
    )
    code, body, _ = _req(
        base,
        f"/api/session/{sid}/run",
        data=json.dumps({"style": "acoustic"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    assert code == 200
    mix_lufs = json.loads(body)["lufs"]

    from automix import analyze, dsp

    def lufs_of(kind):
        code, data, _ = _req(base, f"/api/session/{sid}/audio/{kind}")
        assert code == 200
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(data)
        audio, sr = analyze.load_stem_file(str(p))
        return dsp.loudness_lufs(audio, sr)

    assert abs(lufs_of("reference") - mix_lufs) < 1.0
    assert abs(lufs_of("raw") - mix_lufs) < 1.0

def test_session_pruning_removes_old_and_excess(tmp_path):
    """A UI left open all day must not fill the disk with 400 MB sessions."""
    from automix.server import _prune

    root = tmp_path / "sessions"
    root.mkdir()
    for i in range(5):
        d = root / f"s{i}"
        d.mkdir()
        (d / "x").write_text("x")
    old = time.time() - 13 * 3600
    os.utime(root / "s0", (old, old))
    os.utime(root / "s1", (old, old))

    assert _prune(str(root), ttl_h=12, keep=40) == 2
    assert sorted(p.name for p in root.iterdir()) == ["s2", "s3", "s4"]

    assert _prune(str(root), ttl_h=100, keep=2) == 1
    assert sorted(p.name for p in root.iterdir()) == ["s3", "s4"]

def test_roles_endpoint_includes_preflight_metrics(srv):
    """A clipped or empty stem must be visible before the user hits mix."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    # a deliberately clipped stem
    import numpy as np

    n = int(1.0 * 48000)
    t = np.arange(n) / 48000
    hot = np.clip(3.0 * np.sin(2 * np.pi * 440.0 * t), -1, 1)
    pcm = (np.stack([hot, hot], 1) * 32767).astype("<i2")
    import io
    import wave as _wave

    buf = io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())
    _upload(base, sid, "gtr.wav", buf.getvalue())

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    roles = json.loads(body)["roles"]
    assert roles["gtr.wav"]["metrics"]["clipped_ratio"] > 0.1
    assert roles["kick.wav"]["metrics"]["lufs"] > -40


def _preflight_wav_bytes(
    sr=48000,
    total_s=46.0,
    residue_s=0.25,
    residue_amp=0.003,
    music_start_s=36.0,
    music_amp=0.2,
    freq=440.0,
    seed=7,
) -> bytes:
    """A stereo WAV whose head is a weak residue and whose music starts late.

    The residue sits above the -60 dB floor, so the active span *starts* with
    it instead of with the music: the same thing a decoded AAC file does when
    a few near-silent frames still carry a little noise.
    """
    import io

    n = int(total_s * sr)
    audio = np.zeros((n, 2))
    m = int(residue_s * sr)
    if m:
        rng = np.random.default_rng(seed)
        audio[:m] = residue_amp * rng.uniform(-1.0, 1.0, size=(m, 2))
    s = int(music_start_s * sr)
    if s < n:
        t = np.arange(n - s) / sr
        audio[s:] = music_amp * np.sin(2 * np.pi * freq * t)[:, None]
    pcm = (audio * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_preflight_lufs_measures_the_whole_active_span_not_its_first_thirty_seconds(srv):
    """Late music must not be reported as very quiet.

    The pre-flight numbers are a sample of the active span.  When the head of
    the file carries nothing but a weak residue -- above the -60 dB floor, so
    the active span starts there -- a reading taken from the first 30 seconds
    of that span is silence, and the UI would label a normally loud stem as
    "very quiet" before the user ever heard it.
    """
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "vocal_lead.wav", _preflight_wav_bytes())

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    metrics = json.loads(body)["roles"]["vocal_lead.wav"]["metrics"]
    assert metrics["silent"] is False
    assert metrics["lufs"] > -30, "late music was reported as very quiet"
    assert metrics["lufs"] < -6, "the reading is not the music's own level"
    # the whole-span numbers stay whole-span
    assert metrics["active_s"] == pytest.approx(46.0, abs=0.05)
    assert metrics["peak_db"] == pytest.approx(-13.98, abs=0.1)
    assert metrics["clipped_ratio"] == 0.0


def test_preflight_keeps_a_genuinely_quiet_stem_quiet(srv):
    """A stem that really is quiet must still read quiet."""
    base, _ = srv
    sid = _new_session(base)
    _upload(
        base,
        sid,
        "vocal_lead.wav",
        _preflight_wav_bytes(total_s=40.0, music_start_s=0.0, residue_amp=0.0, music_amp=0.002),
    )

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    metrics = json.loads(body)["roles"]["vocal_lead.wav"]["metrics"]
    assert metrics["silent"] is False
    assert metrics["lufs"] < -30


def test_preflight_measures_a_short_one_shot_momentarily(srv):
    """A 150 ms stem is shorter than a loudness block: it is not silent.

    Integrated loudness has no 400 ms block to gate here and would read the
    -70 floor, so a one-shot has to be measured momentarily.
    """
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "fx.wav", _wav_bytes(1000.0, dur=0.15, amp=0.3))

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    metrics = json.loads(body)["roles"]["fx.wav"]["metrics"]
    assert metrics["silent"] is False
    assert metrics["lufs"] > -30
    assert metrics["active_s"] == pytest.approx(0.15, abs=0.01)


def test_preflight_reports_a_silent_stem_as_silent(srv):
    """Silence stays silence: the reading must not invent a level."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "drums_full.wav", _wav_bytes(60.0, dur=1.0, amp=0.0))

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    metrics = json.loads(body)["roles"]["drums_full.wav"]["metrics"]
    assert metrics["silent"] is True
    assert metrics["lufs"] == -70.0
    assert metrics["active_s"] == 0.0
    assert metrics["peak_db"] == -120.0


def test_a_single_stem_can_be_auditioned(srv):
    """Answering "is this really the bass?" should not need a render."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "音軌 02.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/audio/stem/bass.wav")
    assert code == 200 and body[:4] == b"RIFF"

    import urllib.parse

    quoted = urllib.parse.quote("音軌 02.wav")
    code, body, _ = _req(base, f"/api/session/{sid}/audio/stem/{quoted}")
    assert code == 200 and body[:4] == b"RIFF"

    code, _, _ = _req(base, f"/api/session/{sid}/audio/stem/nope.wav")
    assert code == 404

def test_manual_trim_shifts_the_gain_and_is_recorded(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    before = {t["id"]: t["gain_db"] for t in json.loads(body)["tracks"]}

    code, body, _ = _req(base, f"/api/session/{sid}/trim",
                         data=json.dumps({"trims": {"vocal_lead": 1.5}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    assert json.loads(body)["trims"] == {"vocal_lead": 1.5}

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    tracks = {t["id"]: t for t in json.loads(body)["tracks"]}
    assert abs((tracks["vocal_lead"]["gain_db"] - before["vocal_lead"]) - 1.5) < 0.02
    assert tracks["vocal_lead"]["trim_db"] == 1.5
    assert any("manual trim" in e for e in tracks["vocal_lead"]["evidence"])


def test_trim_rejects_absurd_values(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    code, body, _ = _req(base, f"/api/session/{sid}/trim",
                         data=json.dumps({"trims": {"bass": 40}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 400
    assert "beyond" in json.loads(body)["error"]


def test_trimming_invalidates_the_previous_render(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    code, _, _ = _req(base, f"/api/session/{sid}/run",
                      data=json.dumps({"style": "pop_2020s"}).encode(),
                      method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    code, _, _ = _req(base, f"/api/session/{sid}/trim",
                      data=json.dumps({"trims": {"bass": -1.0}}).encode(),
                      method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    code, body, _ = _req(base, f"/api/session/{sid}/result")
    assert code == 404

def test_only_one_render_runs_at_a_time(srv):
    """Two concurrent renders would double the peak memory for no benefit."""
    base, _ = srv
    a = _new_session(base)
    b = _new_session(base)
    _upload(base, a, "pad.wav", _wav_bytes(220.0, dur=40.0))
    _upload(base, b, "bass.wav", _wav_bytes(55.0))

    first = {}

    def run_a():
        first["code"], first["body"], _ = _req(
            base, f"/api/session/{a}/run",
            data=json.dumps({"style": "cinematic"}).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )

    t = threading.Thread(target=run_a)
    t.start()
    try:
        # wait for A to actually take the slot
        deadline = time.time() + 30
        started = False
        while time.time() < deadline:
            code, body, _ = _req(base, f"/api/session/{a}/progress")
            if json.loads(body).get("progress"):
                started = True
                break
            time.sleep(0.05)
        assert started

        code, body, _ = _req(
            base, f"/api/session/{b}/run",
            data=json.dumps({"style": "pop_2020s"}).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )
        assert code == 429, (code, body[:200])
        assert "another render" in json.loads(body)["error"]
    finally:
        t.join(180)
    assert first["code"] == 200

    # and once the slot is free, B works
    code, body, _ = _req(
        base, f"/api/session/{b}/run",
        data=json.dumps({"style": "pop_2020s"}).encode(),
        method="POST", headers={"Content-Type": "application/json"},
    )
    assert code == 200

def test_a_muted_track_is_left_out_of_the_mix(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/mute",
                         data=json.dumps({"track_id": "bass", "muted": True}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    assert json.loads(body)["muted"] == ["bass"]

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    ids = [t["id"] for t in json.loads(body)["tracks"]]
    assert "bass" not in ids and "vocal_lead" in ids

    # unmute restores it
    _req(base, f"/api/session/{sid}/mute",
         data=json.dumps({"track_id": "bass", "muted": False}).encode(),
         method="POST", headers={"Content-Type": "application/json"})
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert "bass" in [t["id"] for t in json.loads(body)["tracks"]]


def test_muting_everything_is_a_clear_error(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _req(base, f"/api/session/{sid}/mute",
         data=json.dumps({"track_id": "bass", "muted": True}).encode(),
         method="POST", headers={"Content-Type": "application/json"})
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 500 or code == 400
    assert "muted" in json.loads(body)["error"]

def test_run_accepts_an_explicit_loudness_target(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s", "target_lufs": -14.0}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    r = json.loads(body)
    assert abs(r["lufs"] - (-14.0)) < 0.5
    assert abs(r["target_lufs"] - (-14.0)) < 0.01


def test_an_absurd_target_is_rejected(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s", "target_lufs": -40}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 500 or code == 400
    assert "outside the supported" in json.loads(body)["error"]

def test_peaks_endpoint_returns_an_envelope_and_sections(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))
    _req(base, f"/api/session/{sid}/run",
         data=json.dumps({"style": "pop_2020s"}).encode(),
         method="POST", headers={"Content-Type": "application/json"})

    code, body, _ = _req(base, f"/api/session/{sid}/peaks?kind=mix")
    assert code == 200
    d = json.loads(body)
    assert len(d["peaks"]) == 600
    assert all(0.0 <= p <= 1.01 for p in d["peaks"])
    assert d["duration_s"] > 0

    code, _, _ = _req(base, f"/api/session/{sid}/peaks?kind=raw")
    assert code == 200
    code, _, _ = _req(base, f"/api/session/{sid}/peaks?kind=reference")
    assert code == 404  # no reference uploaded

def test_a_session_survives_a_server_restart(srv):
    """The audio never left the disk; the id -> Session map did."""
    from automix import server as srvmod

    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "kick.wav", _wav_bytes(60.0))
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _req(base, f"/api/session/{sid}/trim",
         data=json.dumps({"trims": {"bass": 1.5}}).encode(),
         method="POST", headers={"Content-Type": "application/json"})
    _req(base, f"/api/session/{sid}/mute",
         data=json.dumps({"track_id": "kick", "muted": True}).encode(),
         method="POST", headers={"Content-Type": "application/json"})
    _req(base, f"/api/session/{sid}/roles",
         data=json.dumps({"roles": {"bass.wav": "sub_808"}}).encode(),
         method="POST", headers={"Content-Type": "application/json"})

    # simulate a restart: forget every live session, then restore from disk
    root = srvmod._sessions[sid].dir.rsplit("/", 1)[0]
    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    assert srvmod._load_sessions(root) >= 1

    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200
    st = json.loads(body)
    assert st["muted"] == ["kick"]
    assert set(st["stems"]) == {"kick.wav", "bass.wav"}

    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert json.loads(body)["roles"]["bass.wav"]["role"] == "sub_808"
    assert json.loads(body)["roles"]["bass.wav"]["source"] == "user"

def test_the_previous_style_is_kept_for_ab(srv):
    """Run pop, then rock: the pop mix must still be playable."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    first = json.loads(body)
    assert first.get("prev_style") is None

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "lofi"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    second = json.loads(body)
    assert second["prev_style"] == "pop_2020s"
    assert abs(second["lufs"] - first["lufs"]) > 0.5

    code, prev, _ = _req(base, f"/api/session/{sid}/audio/prev")
    assert code == 200 and prev[:4] == b"RIFF"
    code, cur, _ = _req(base, f"/api/session/{sid}/audio/mix")
    assert code == 200 and cur[:4] == b"RIFF"
    assert prev != cur

    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert json.loads(body)["has_prev"] is True

def test_pan_can_be_overridden_and_is_recorded(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "guitar_l.wav", _wav_bytes(220.0))
    _upload(base, sid, "keys.wav", _wav_bytes(330.0))

    code, body, _ = _req(base, f"/api/session/{sid}/pan",
                         data=json.dumps({"track_id": "guitar_l", "pan": 0.4}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    assert json.loads(body)["pans"]["guitar_l"] == 0.4

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    track = next(t for t in json.loads(body)["tracks"] if t["id"] == "guitar_l")
    assert abs(track["pan"] - 0.4) < 0.01
    assert any("manual pan" in e for e in track["evidence"])


def test_pan_rejects_out_of_range(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "keys.wav", _wav_bytes(330.0))
    code, body, _ = _req(base, f"/api/session/{sid}/pan",
                         data=json.dumps({"track_id": "keys", "pan": 3}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 400
    assert "between -1 and 1" in json.loads(body)["error"]


def test_pan_survives_a_restart(srv):
    from automix import server as srvmod

    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "keys.wav", _wav_bytes(330.0))
    _req(base, f"/api/session/{sid}/pan",
         data=json.dumps({"track_id": "keys", "pan": -0.3}).encode(),
         method="POST", headers={"Content-Type": "application/json"})

    root = srvmod._sessions[sid].dir.rsplit("/", 1)[0]
    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    srvmod._load_sessions(root)
    assert srvmod._sessions[sid].pans == {"keys": -0.3}

def test_a_session_has_a_stem_limit(srv, monkeypatch):
    from automix import server as srvmod

    monkeypatch.setattr(srvmod, "MAX_STEMS", 3)
    base, _ = srv
    sid = _new_session(base)
    for i in range(3):
        code, _, _ = _upload(base, sid, f"s{i}.wav", _wav_bytes(100.0 + i))
        assert code == 200
    code, body, _ = _upload(base, sid, "one_too_many.wav", _wav_bytes(440.0))
    assert code == 413
    assert "at most 3 stems" in json.loads(body)["error"]

    # replacing an existing stem is still allowed at the limit
    code, _, _ = _upload(base, sid, "s0.wav", _wav_bytes(120.0))
    assert code == 200


def test_a_session_has_a_total_size_limit(srv, monkeypatch):
    from automix import server as srvmod

    monkeypatch.setattr(srvmod, "MAX_SESSION_BYTES", 100_000)
    base, _ = srv
    sid = _new_session(base)
    code, body, _ = _upload(base, sid, "big.wav", _wav_bytes(220.0, dur=2.0))
    assert code == 413
    assert "GB" in json.loads(body)["error"]

def test_a_bad_stem_is_a_client_error_not_a_server_error(srv):
    """Uploading garbage is a 4xx; a 500 would blame the server."""
    base, _ = srv
    sid = _new_session(base)
    code, _, _ = _upload(base, sid, "broken.wav", b"this is not audio at all")
    assert code == 200  # stored; the failure happens when it is decoded
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 400, (code, body[:200])
    assert "StemError" in json.loads(body)["error"]

def test_run_reports_the_checks_for_the_ui(srv):
    """The receipt's verdict table should not require downloading the file."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s"}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    verdicts = json.loads(body)["verdicts"]
    checks = {v["check"]: v for v in verdicts}
    assert "loudness" in checks and "true peak" in checks
    assert all(set(v) == {"check", "measured", "verdict"} for v in verdicts)
    assert checks["loudness"]["verdict"] == "ok"

def test_stems_can_be_exported_from_the_ui_path(srv):
    """The GUI needs the same export the CLI has, with a size to warn about."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "bass.wav", _wav_bytes(55.0))
    _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))

    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps({"style": "pop_2020s", "export_stems": True}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    result = json.loads(body)
    exported = result["exported"]
    names = {e["name"] for e in exported}
    assert "bass.wav" in names and "vocal_lead.wav" in names
    assert "_reverb.wav" in names
    assert "README.txt" in names
    assert "_master_chain.json" in names
    assert sum(e["bytes"] for e in exported) > 100_000  # the size warning has real data

    code, data, _ = _req(base, f"/api/session/{sid}/processed/file/bass.wav")
    assert code == 200 and data[:4] == b"RIFF"
    code, note, _ = _req(base, f"/api/session/{sid}/processed/file/README.txt")
    assert code == 200 and "does NOT reproduce" in note.decode("utf-8")

    code, chain, _ = _req(base, f"/api/session/{sid}/processed/file/_master_chain.json")
    assert code == 200
    master = json.loads(chain)["master"]
    assert master["target_lufs"] is not None
    assert "loudness_gain_db" in master

    code, _, _ = _req(base, f"/api/session/{sid}/processed/file/nope.wav")
    assert code == 404

def test_a_user_can_override_a_role_by_name_or_track_id(srv):
    """The GUI sends the file name; other clients send the bare track id."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "keys.wav", _wav_bytes(220.0))

    def set_role(payload):
        code, raw, _ = _req(base, f"/api/session/{sid}/roles",
                             data=json.dumps(payload).encode(), method="POST",
                             headers={"Content-Type": "application/json"})
        return code, json.loads(raw)

    code, body = set_role({"roles": {"keys.wav": "lead_vocal"}})
    assert code == 200
    assert body["roles"]["keys.wav"]["role"] == "lead_vocal"
    assert body["roles"]["keys.wav"]["source"] == "user"

    code, body = set_role({"roles": {"keys": "bass"}})
    assert code == 200
    assert body["roles"]["keys.wav"]["role"] == "bass"

    code, body = set_role({"roles": {"nope": "bass"}})
    assert code == 400
    assert "no such stem" in body["error"]


def test_the_export_refuses_to_write_more_than_the_limit():
    """The export writes one 24-bit stereo WAV per track: it adds up fast.

    The server has to say so *before* writing, not after, and it has to say
    it as a user error (4xx), because the input was the problem.
    """
    import automix.server as server

    assert server.EXPORT_LIMIT_BYTES > 0
    plan = {"source": {"sample_rate": 48000, "duration_s": 60.0}}
    # one minute of four tracks at 48 kHz, 24-bit stereo: about 82 MiB
    est = server._export_size_bytes(plan, 4)
    assert 80 * 1024**2 < est < 85 * 1024**2, est
    # twelve hours of sixteen tracks is over the limit
    long = {"source": {"sample_rate": 48000, "duration_s": 12 * 3600}}
    assert server._export_size_bytes(long, 16) > server.EXPORT_LIMIT_BYTES


def test_a_file_name_with_a_newline_is_cleaned_up():
    """A newline in a file name is legal on disk and useless everywhere else.

    It reaches the plan, the report and the UI, so the server replaces it
    instead of storing it as-is.
    """
    from automix.server import _safe_name

    assert _safe_name("kick.wav") == "kick.wav"
    assert _safe_name("kick%0Aodd.wav") == "kick_odd.wav"
    assert "\n" not in _safe_name("kick%0Aodd.wav")
    # a path in the name is stripped, not obeyed
    assert _safe_name("../../etc/passwd.wav") == "passwd.wav"
    # and a name without a supported extension is refused
    try:
        _safe_name("passwd")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError")


def test_source_profile_http_proof_override_replace_and_restore(srv):
    """Real HTTP/child renderer, with one genuine mono upload and sparse role."""
    import io
    from automix import analyze
    base, root = srv
    sid = _new_session(base)
    sr = 8000
    first = _wav_bytes(freq=730, dur=9, sr=sr, amp=.13)
    second = _wav_bytes(freq=237, dur=9, sr=sr, amp=.09)
    assert _upload(base, sid, 'vocal_lead.wav', first)[0] == 200
    assert _upload(base, sid, 'other.wav', second)[0] == 200
    # Reconstruct from the actual quantized uploads, not a tone-distance proxy.
    def decoded(body):
        with wave.open(io.BytesIO(body), 'rb') as f:
            return np.frombuffer(f.readframes(f.getnframes()), '<i2').reshape(-1, 2) / 32768
    def wav(x):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as f:
            f.setnchannels(2); f.setsampwidth(2); f.setframerate(sr)
            f.writeframes(np.rint(x * 32767).astype('<i2').tobytes())
        return stream.getvalue()
    ref = wav((decoded(first) + decoded(second)) * .7)
    code, _, _ = _req(base, f'/api/session/{sid}/reference', ref, 'POST', {'X-Filename': 'same.wav'})
    assert code == 200
    def run(options):
        code, body, _ = _req(base, f'/api/session/{sid}/run', json.dumps(options).encode(), 'POST')
        assert code == 200, body
        return json.loads(body)
    result = run({'blend': 0})
    assert result['source_balance']['origin'] == 'proof'
    assert result['style']['balance_preserve'] == 1
    assert result['style']['id'] == 'pop_2020s'
    assert result['source_balance']['proof']['active_coverage_s'] == 9
    code, body, _ = _req(base, f'/api/session/{sid}/report')
    assert code == 200 and b'coherent_same_content_reconstruction' in body
    result = run({'source_balance_profile': 'balanced_stems', 'balance_preserve': .5,
                  'style': 'rock', 'blend': 0})
    assert result['style']['balance_preserve'] == .5
    assert result['source_balance']['origin'] == 'explicit'
    from automix.server import Session
    restored = Session.load(root, sid)
    assert restored.result['source_balance']['requested_balance_preserve'] == .5
    assert restored.result['style']['id'] == 'rock'
    code, state, _ = _req(base, f'/api/session/{sid}/state')
    state = json.loads(state)
    assert code == 200 and state['source_balance'] == result['source_balance']
    assert state['style']['id'] == 'rock'
    # Identical filename with changed audio cannot retain the old proof.
    unrelated = _wav_bytes(freq=981, dur=9, sr=sr, amp=.1)
    assert _req(base, f'/api/session/{sid}/reference', unrelated, 'POST', {'X-Filename': 'same.wav'})[0] == 200
    result = run({'blend': 0})
    assert result['source_balance']['balance_preserve'] == .5
    assert not result['source_balance']['proof']['passed']
    for malformed in ({'source_balance_profile': 'four_stems'}, {'balance_preserve': float('nan')},
                      {'balance_preserve': '0.5'}, {'balance_preserve': True}, {'balance_preserve': 1.5}):
        code, body, _ = _req(base, f'/api/session/{sid}/run', json.dumps(malformed).encode(), 'POST')
        assert code == 400 and b'error' in body


def test_reference_balance_http_and_session_restore(srv):
    import io
    from automix.server import Session
    base,root=srv;sid=_new_session(base);sr=8000
    first=_wav_bytes(freq=730,dur=12,sr=sr,amp=.16)
    second=_wav_bytes(freq=237,dur=12,sr=sr,amp=.04)
    assert _upload(base,sid,'vocal_lead.wav',first)[0]==200
    assert _upload(base,sid,'other.wav',second)[0]==200
    def decoded(body):
        with wave.open(io.BytesIO(body),'rb') as f:
            return np.frombuffer(f.readframes(f.getnframes()),'<i2').reshape(-1,2)/32768
    def wav(x):
        buf=io.BytesIO()
        with wave.open(buf,'wb') as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(sr)
            f.writeframes(np.rint(x*32767).astype('<i2').tobytes())
        return buf.getvalue()
    original=(decoded(first)*10**(-6/20)+decoded(second)*10**(6/20))*.7
    assert _req(base,f'/api/session/{sid}/reference',wav(original),'POST',{'X-Filename':'rough.wav'})[0]==200
    def run(options):
        code,body,_=_req(base,f'/api/session/{sid}/run',json.dumps(options).encode(),'POST')
        assert code==200,body
        return json.loads(body)
    result=run({'source_balance_profile':'reference_balance','blend':0})
    b=result['source_balance']
    assert b['origin']=='reference_fit' and b['resolved']=='reference_balance' and b['balance_preserve']==1
    assert b['proof']['common_gain_applied'] is False
    assert b['proof']['relative_gain_db']['vocal_lead']==pytest.approx(-6,abs=.02)
    code,body,_=_req(base,f'/api/session/{sid}/state')
    assert code==200 and json.loads(body)['source_balance']==b
    restored=Session.load(root,sid)
    assert restored.result['source_balance']==b
    code,body,_=_req(base,f'/api/session/{sid}/report')
    assert code==200 and b'raw-source gain-only reconstruction passed' in body
    manual=run({'source_balance_profile':'reference_balance','balance_preserve':.5,'blend':0})
    assert manual['source_balance']['origin']=='explicit'
    assert manual['style']['balance_preserve']==.5
    assert 'relative_gain_db' not in manual['source_balance']['proof']
    # Same filename with unrelated audio must lose the accepted fit.
    wrong=_wav_bytes(freq=981,dur=12,sr=sr,amp=.1)
    assert _req(base,f'/api/session/{sid}/reference',wrong,'POST',{'X-Filename':'rough.wav'})[0]==200
    fallback=run({'source_balance_profile':'reference_balance','blend':0})
    assert fallback['source_balance']['balance_preserve']==.5
    assert fallback['source_balance']['resolved']=='unknown'


# ------------------------------------- the controls of the last successful mix

def _two_stems(base, sid):
    """Two small synthetic stems: a real render, fast enough to repeat."""
    assert _upload(base, sid, "bass.wav", _wav_bytes(55.0))[0] == 200
    assert _upload(base, sid, "vocal_lead.wav", _wav_bytes(440.0))[0] == 200


def _render(base, sid, options):
    code, body, _ = _req(base, f"/api/session/{sid}/run",
                         data=json.dumps(options).encode(), method="POST",
                         headers={"Content-Type": "application/json"})
    return code, body


def _state(base, sid):
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200, body
    return json.loads(body)


def _restart(base, sid, root):
    """Forget every live session and rebuild them from their manifests."""
    from automix import server as srvmod

    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    assert srvmod._load_sessions(root) >= 1


def test_a_successful_run_saves_the_controls_it_was_given(srv):
    """Reopening a session must show the choices, not what the engine derived."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {
        "style": "rock",
        "intensity": 0.8,
        "source_balance_profile": "balanced_stems",
        "balance_preserve": 0.35,
        "blend": 0.25,
        "target_lufs": -20.0,      # valid through the API, not one of the three
        "export_stems": True,
        "reverb_amount": 0.5,
    })
    assert code == 200, body[:300]
    controls = _state(base, sid)["controls"]
    assert controls == {
        "style": "rock",
        "intensity": 0.8,
        "source_balance_profile": "balanced_stems",
        "balance_preserve": 0.35,
        "blend": 0.25,
        "target_lufs": -20.0,
        "export_stems": True,
        "reverb_amount": 0.5,
    }
    # the manifest carries the same thing, so a restarted server restores it
    assert Session.load(root, sid).controls == controls
    _restart(base, sid, root)
    assert _state(base, sid)["controls"] == controls


def test_an_unset_k_and_an_unset_target_stay_unset(srv):
    """A fitted number is a result; saving it would claim the user chose it."""
    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    # balanced_stems resolves the keep amount to 1.00 all by itself, and with
    # no reference the requested blend has no effect either
    code, body = _render(base, sid, {"style": "pop_2020s",
                                     "source_balance_profile": "balanced_stems",
                                     "blend": 0.25})
    assert code == 200, body[:300]
    result = json.loads(body)
    assert result["style"]["balance_preserve"] == 1   # the engine's own number
    controls = _state(base, sid)["controls"]
    assert controls["balance_preserve"] is None        # the user never asked
    assert controls["target_lufs"] is None             # "follow the style"
    assert controls["blend"] == 0.25                   # the request is kept
    # an explicit request is saved as the request
    code, body = _render(base, sid, {"style": "pop_2020s",
                                     "source_balance_profile": "balanced_stems",
                                     "balance_preserve": 0.35,
                                     "target_lufs": -14.0,
                                     "blend": 0.0})
    assert code == 200, body[:300]
    controls = _state(base, sid)["controls"]
    assert controls["balance_preserve"] == 0.35
    assert controls["target_lufs"] == -14.0
    assert controls["blend"] == 0.0


def test_a_session_saved_before_the_controls_still_opens(srv):
    """A session from the old server has no snapshot: say so, do not guess."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    manifest = os.path.join(root, sid, "session.json")
    with open(manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("controls", None)
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    assert Session.load(root, sid).controls is None
    _restart(base, sid, root)
    assert _state(base, sid)["controls"] is None


def test_a_damaged_snapshot_does_not_stop_the_session(srv):
    """Unusable saved metadata falls back conservatively instead of crashing."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    manifest = os.path.join(root, sid, "session.json")
    with open(manifest, encoding="utf-8") as fh:
        good = json.load(fh)
    for saved in ("not a dict",
                  {"style": 17},
                  {"style": "rock", "balance_preserve": "abc", "target_lufs": "NaN",
                   "intensity": 5, "blend": None, "export_stems": "yes"}):
        data = dict(good)
        data["controls"] = saved
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        loaded = Session.load(root, sid)
        assert loaded is not None
        # repair3: a value we cannot vouch for drops the whole snapshot, so the
        # UI reports the session as one whose choices cannot be restored
        # instead of restoring half of it and calling that a full restore
        assert loaded.controls is None
        _restart(base, sid, root)
        assert _state(base, sid)["controls"] == loaded.controls


def test_corrupt_saved_controls_are_dropped_instead_of_claimed(srv):
    """A saved value we cannot vouch for must not come back as a choice."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "balance_preserve": 0.35,
                                     "target_lufs": -20.0, "blend": 0.25})
    assert code == 200, body[:300]
    manifest = os.path.join(root, sid, "session.json")
    with open(manifest, encoding="utf-8") as fh:
        good = json.load(fh)
    assert good["controls"]["target_lufs"] == -20.0
    corrupt = [
        {"target_lufs": -40.0},          # outside the planner's -24..-6 range
        {"target_lufs": 999.0},
        {"source_balance_profile": "bogus"},
        {"style": "bogus_style"},        # not one of the bundled presets
        {"intensity": "0.5"},            # a string is not a number
        {"balance_preserve": "abc"},
        {"blend": 1.5},
        {"export_stems": "yes"},
    ]
    for saved in corrupt:
        data = dict(good)
        data["controls"] = dict(good["controls"], **saved)
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        loaded = Session.load(root, sid)
        assert loaded is not None
        assert loaded.controls is None, saved
        _restart(base, sid, root)
        assert _state(base, sid)["controls"] is None, saved
    # a sound snapshot still comes back, so the check is not "always drop"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(good, fh, ensure_ascii=False, indent=2)
    assert Session.load(root, sid).controls == good["controls"]


def test_the_state_snapshot_of_roles_and_revision_is_one_moment(srv, monkeypatch):
    """The state answer must not mix a role map from before with a revision from after.

    The gate sits after the session lookup and before the state snapshot, so
    this test cannot pass on a runtime that only locks the lookup.
    """
    from automix import server as srvmod
    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)

    lookup_done = threading.Event()
    release_lookup = threading.Event()
    original_session = srvmod.Handler._session

    def gated_session(self, session_id):
        session = original_session(self, session_id)
        if self.path.endswith("/state"):
            lookup_done.set()
            release_lookup.wait(10)
        return session

    monkeypatch.setattr(srvmod.Handler, "_session", gated_session)

    done = threading.Event()
    seen = {}

    def read():
        code, body, _ = _req(base, f"/api/session/{sid}/state")
        seen["code"] = code
        seen["body"] = json.loads(body)
        done.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert lookup_done.wait(5), "the state read never got past the session lookup"
        with srvmod._lock:
            release_lookup.set()
            # the reader is past the lookup now, so only the snapshot lock can
            # still hold it back
            assert not done.wait(0.5), "the state read finished without the snapshot lock"
    finally:
        release_lookup.set()
        reader.join(10)
    assert seen["code"] == 200
    assert seen["body"]["manual_roles"] == {}
    assert seen["body"]["role_revision"] == 0
    # and after a committed change the two agree
    code, body, _ = _req(base, f"/api/session/{sid}/roles",
                         data=json.dumps({"roles": {"bass.wav": "keys"}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    state = _state(base, sid)
    assert state["manual_roles"] == {"bass.wav": "keys"}
    assert state["role_revision"] == 1


def test_the_role_revision_orders_role_snapshots(srv):
    """An accepted manual role change advances the revision; a rejected one does not."""
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert code == 200
    assert json.loads(body)["role_revision"] == 0        # a read does not advance it
    code, body, _ = _req(base, f"/api/session/{sid}/roles",
                         data=json.dumps({"roles": {"bass.wav": "sub_808"}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    accepted = json.loads(body)
    assert accepted["role_revision"] == 1
    assert accepted["roles"]["bass.wav"]["role"] == "sub_808"
    # a rejected change advances nothing
    code, body, _ = _req(base, f"/api/session/{sid}/roles",
                         data=json.dumps({"roles": {"nope.wav": "sub_808"}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 400
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    assert json.loads(body)["role_revision"] == 1
    # the roles and the revision travel together, and survive a restart
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert json.loads(body)["role_revision"] == 1
    _restart(base, sid, root)
    code, body, _ = _req(base, f"/api/session/{sid}/roles")
    restarted = json.loads(body)
    assert restarted["role_revision"] == 1
    assert restarted["roles"]["bass.wav"]["role"] == "sub_808"
    # the next accepted change advances it again
    code, body, _ = _req(base, f"/api/session/{sid}/roles",
                         data=json.dumps({"roles": {"bass.wav": "keys"}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200
    assert json.loads(body)["role_revision"] == 2


def test_a_legacy_fractional_reverb_state_is_preserved(srv):
    """The session's own reverb amount is known even without the snapshot."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    manifest = os.path.join(root, sid, "session.json")
    with open(manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    data["reverb_amount"] = 0.333
    data.pop("controls", None)
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    loaded = Session.load(root, sid)
    assert loaded is not None
    assert loaded.reverb_amount == 0.333
    assert loaded.controls is None
    _restart(base, sid, root)
    state = _state(base, sid)
    assert state["reverb_amount"] == 0.333
    assert state["controls"] is None


def test_the_state_carries_the_manual_roles(srv):
    """The UI must not wait for the slow detection call to show the choices."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body, _ = _req(base, f"/api/session/{sid}/roles",
                         data=json.dumps({"roles": {"bass.wav": "sub_808"}}).encode(),
                         method="POST", headers={"Content-Type": "application/json"})
    assert code == 200, body[:200]
    assert _state(base, sid)["manual_roles"] == {"bass.wav": "sub_808"}
    _restart(base, sid, root)
    assert _state(base, sid)["manual_roles"] == {"bass.wav": "sub_808"}
    assert Session.load(root, sid).roles == {"bass.wav": "sub_808"}


def test_a_failed_run_leaves_the_last_successful_controls_alone(srv):
    """Rejected and failed requests must not change the last successful controls."""
    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "rock", "balance_preserve": 0.35,
                                     "target_lufs": -20.0, "blend": 0.25})
    assert code == 200, body[:300]
    before = _state(base, sid)["controls"]
    # rejected before the render starts
    assert _render(base, sid, {"style": "no_such_style"})[0] == 400
    assert _render(base, sid, {"balance_preserve": 1.5})[0] == 400
    # rejected by the planner inside the render (an impossible target)
    assert _render(base, sid, {"style": "rock", "target_lufs": -40})[0] == 400
    assert _state(base, sid)["controls"] == before


# ------------------------------------------------------- R118 group faders

def test_group_faders_reach_the_plan_and_are_kept_with_the_render(srv):
    """R118: the few faders after a mix (vocal, drums, bass, other, space).
    A run with them writes buses.<group>.user_trim_db, the controls snapshot
    keeps them (only when set), a restart restores them, and a run without
    the field keeps the current render's faders."""
    from automix.server import Session
    base, root = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "pop_2020s", "group_trims": {"vocal": 2.0, "fx": -3.0}})
    assert code == 200, body[:300]
    code, plan_body, _ = _req(base, f"/api/session/{sid}/plan")
    assert code == 200
    plan = json.loads(plan_body)
    assert plan["buses"]["vocal"]["user_trim_db"] == 2.0
    assert plan["buses"]["fx"]["user_trim_db"] == -3.0
    controls = _state(base, sid)["controls"]
    assert controls["group_trims"] == {"vocal": 2.0, "fx": -3.0}
    assert Session.load(root, sid).controls == controls
    _restart(base, sid, root)
    assert _state(base, sid)["controls"]["group_trims"] == {"vocal": 2.0, "fx": -3.0}
    # a run that does not mention them keeps them
    code, body = _render(base, sid, {"style": "pop_2020s"})
    assert code == 200, body[:300]
    assert _state(base, sid)["controls"]["group_trims"] == {"vocal": 2.0, "fx": -3.0}
    # all back to 0: the snapshot has no faders again
    code, body = _render(base, sid, {"style": "pop_2020s", "group_trims": {"vocal": 0, "fx": 0}})
    assert code == 200, body[:300]
    assert "group_trims" not in _state(base, sid)["controls"]
    plan = json.loads(_req(base, f"/api/session/{sid}/plan")[1])
    assert all("user_trim_db" not in b for b in plan["buses"].values())


@pytest.mark.parametrize("bad", [{"vocal": 13}, {"master": 1}, [1, 2], {"vocal": "loud"}])
def test_a_bad_group_fader_is_refused_before_rendering(srv, bad):
    base, _ = srv
    sid = _new_session(base)
    _two_stems(base, sid)
    code, body = _render(base, sid, {"style": "pop_2020s", "group_trims": bad})
    assert code == 400, body[:300]
    assert not _state(base, sid)["has_mix"]


def test_a_damaged_saved_fader_drops_the_snapshot():
    from automix.server import clean_controls
    ok = {"style": "rock", "group_trims": {"vocal": 1.5}}
    assert clean_controls(ok)["group_trims"] == {"vocal": 1.5}
    assert "group_trims" not in clean_controls({"style": "rock"})
    for bad in ({"vocal": 99}, {"nope": 1}, "loud"):
        assert clean_controls({"style": "rock", "group_trims": bad}) is None
