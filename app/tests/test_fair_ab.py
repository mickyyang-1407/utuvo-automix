"""The level-matched comparison pair (R42).

The comparison between the finished mix and the untreated sum is only
meaningful when both are the same loudness and both are *one constant gain*
away from their source.  The old code gain matched the sum and then ran it
through the limiter, so the level of the "before" moved over time and part
of what a listener heard was the limiter.

These tests are the numerical side of that claim:

* each view is exactly its input times one constant (no time varying gain);
* the two views end at the same loudness;
* a high crest untreated sum lowers *both* views, never only the raw one;
* a low target needs no headroom reduction;
* silence, a too short input and a nonfinite input are reported honestly
  instead of being published as a match;
* a render that claims a matched pair must deliver it, and a failure in the
  middle of that publication leaves the previous render alone;
* an old session cannot claim a matched pair it never had.

Everything here is synthetic: isolated arrays and task private temporary
folders.  No real session, no real audio, no network outside 127.0.0.1.
"""

from __future__ import annotations

import io
import json
import os
import struct
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from automix import dsp
from automix import server as srvmod
from automix.server import (
    Handler,
    PublishError,
    Session,
    _monitor_plan,
    _publish_render,
    _raw_sum_audio,
    _write_wav_float32,
)

SR = 48000
CEILING = -1.0  # the ceiling the renderer honours


# ------------------------------------------------------------------ helpers

def _tone(dur=3.0, amp=0.2, freq=440.0, sr=SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([mono, mono], axis=1)


def _crest(dur=3.0, burst_s=0.05, burst_amp=0.95, quiet_amp=0.01,
           freq=440.0, sr=SR) -> np.ndarray:
    """A signal with a high crest factor: one loud burst, then near silence."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    mono = quiet_amp * np.sin(2 * np.pi * freq * t)
    nb = int(burst_s * sr)
    mono[:nb] = burst_amp * np.sin(2 * np.pi * freq * t[:nb])
    return np.stack([mono, mono], axis=1)


def _at_lufs(x: np.ndarray, target: float, sr: int = SR) -> np.ndarray:
    """The same signal, scaled to a given loudness."""
    cur = dsp.loudness_lufs(x, sr)
    return x * dsp.db_to_lin(target - cur)


def _gain_and_spread(out: np.ndarray, inp: np.ndarray) -> tuple[float, float]:
    """The constant gain between two signals, and how constant it is.

    A limiter (or any time varying gain) shows up here as a spread; a single
    gain shows up as float rounding only.
    """
    mask = np.abs(inp) > 1e-4
    ratio = out[mask] / inp[mask]
    mean = float(np.mean(ratio))
    return mean, float(np.max(np.abs(ratio - mean)))


def _read_float_wav(path: str) -> tuple[np.ndarray, int]:
    """Read back a float WAV, and say what it claims to be."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a RIFF/WAVE"
    pos = 12
    fmt = None
    payload = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    assert fmt is not None and payload is not None, "no fmt/data chunk"
    tag, n_ch, sr = fmt[0], fmt[1], fmt[2]
    bits = fmt[5]
    assert tag == 3, f"expected IEEE float (3), got {tag}"
    assert bits == 32, f"expected 32 bits, got {bits}"
    arr = np.frombuffer(payload, dtype="<f4").reshape(-1, n_ch)
    return arr.astype(np.float64), sr


def _wav_bytes(freq=220.0, dur=4.0, sr=SR, amp=0.3) -> bytes:
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


# ------------------------------------------------- the plan, on its own

def test_each_view_is_exactly_one_constant_gain():
    """No limiter, no compressor: one number multiplies the whole signal."""
    mix = _at_lufs(_tone(), -9.0)
    raw = _at_lufs(_crest(), -9.0)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "matched"

    mix_out = mix * dsp.db_to_lin(plan["mix_gain_db"])
    raw_out = raw * dsp.db_to_lin(plan["raw_gain_db"])

    gain, spread = _gain_and_spread(mix_out, mix)
    assert abs(gain - dsp.db_to_lin(plan["mix_gain_db"])) < 1e-9
    assert spread < 1e-9, f"the mix view is not one gain: spread {spread}"

    gain, spread = _gain_and_spread(raw_out, raw)
    assert abs(gain - dsp.db_to_lin(plan["raw_gain_db"])) < 1e-9
    assert spread < 1e-9, f"the raw view is not one gain: spread {spread}"


def test_the_two_views_end_at_the_same_loudness():
    """Otherwise the listener just picks the louder one."""
    mix = _at_lufs(_tone(), -9.0)
    raw = _at_lufs(_crest(), -6.0)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "matched"

    mix_lufs = dsp.loudness_lufs(mix * dsp.db_to_lin(plan["mix_gain_db"]), SR)
    raw_lufs = dsp.loudness_lufs(raw * dsp.db_to_lin(plan["raw_gain_db"]), SR)
    assert abs(mix_lufs - raw_lufs) < 0.05, f"{mix_lufs:.3f} vs {raw_lufs:.3f}"
    assert abs(mix_lufs - plan["level_lufs"]) < 0.05


def test_a_high_crest_sum_lowers_both_views():
    """The -9 case: the untreated sum cannot reach the mix, so both go down.

    Lowering only the raw view would make the comparison a level trick in
    the other direction: the mix would sound louder than the export.
    """
    mix = _at_lufs(_tone(), -9.0)
    raw = _at_lufs(_crest(), -9.0)
    mix_lufs = dsp.loudness_lufs(mix, SR)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "matched"

    assert plan["level_lufs"] < mix_lufs - 0.5, "the level was not pulled down"
    assert plan["mix_gain_db"] < 0, "the mix view must be quieter too"
    assert plan["raw_gain_db"] < 0, "the raw view must be quieter"
    # both views stay under the ceiling
    assert plan["mix_peak_db"] <= CEILING + 1e-6
    assert plan["raw_peak_db"] <= CEILING + 1e-6
    # and the raw keeps its margin under it
    assert plan["raw_peak_db"] <= CEILING - 0.2 + 1e-6


def test_a_low_target_needs_no_headroom_reduction():
    """The -17 case: the sum is turned down, so it fits under the ceiling."""
    mix = _at_lufs(_tone(), -17.0)
    raw = _at_lufs(_tone(freq=220.0), -9.0)
    mix_lufs = dsp.loudness_lufs(mix, SR)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "matched"

    assert abs(plan["level_lufs"] - mix_lufs) < 0.05, "the level was reduced"
    assert plan["mix_gain_db"] <= 0.0
    assert plan["raw_gain_db"] < 0.0, "the sum has to be turned down"
    assert plan["raw_peak_db"] <= CEILING + 1e-6
    assert plan["headroom_db"] >= 0.0


def test_a_silent_side_cannot_be_matched():
    """No level, no match: the pair says so instead of pretending."""
    mix = _at_lufs(_tone(), -9.0)
    raw = np.zeros_like(mix)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "unavailable"
    assert "raw" in plan["unavailable"]
    assert plan["level_lufs"] is None
    assert plan["mix_gain_db"] is None
    assert plan["raw_gain_db"] is None


def test_a_short_side_cannot_be_matched():
    """Shorter than a gating block: there is no loudness to match."""
    mix = _at_lufs(_tone(), -9.0)
    raw = _tone(dur=0.1)
    plan = _monitor_plan(mix, SR, raw, SR, CEILING)
    assert plan["status"] == "unavailable"
    assert plan["level_lufs"] is None


def test_a_nonfinite_input_is_rejected():
    """A corrupt signal is never published as audio."""
    mix = _at_lufs(_tone(), -9.0)
    broken = _at_lufs(_tone(), -9.0).copy()
    broken[100, 0] = np.nan
    with pytest.raises(ValueError):
        _monitor_plan(mix, SR, broken, SR, CEILING)

    inf = _at_lufs(_tone(), -9.0).copy()
    inf[100, 1] = np.inf
    with pytest.raises(ValueError):
        _monitor_plan(mix, SR, inf, SR, CEILING)


def test_the_raw_sum_keeps_the_native_timeline():
    """The sum is the stems as they are: same rate, same length, no gain."""
    class _Stem:
        def __init__(self, audio, sr=SR):
            self.audio = audio
            self.sr = sr

    short = _tone(dur=1.0)
    long = _tone(dur=2.0, freq=220.0)
    acc, sr = _raw_sum_audio([_Stem(short), _Stem(long)])
    assert sr == SR
    assert acc.shape == long.shape
    # the shorter stem is padded with silence, not stretched
    assert np.allclose(acc[: len(short)], short + long[: len(short)])
    assert np.allclose(acc[len(short):], long[len(short):])


def test_the_monitoring_files_are_float(tmp_path):
    """Float32 keeps the gain exact; the reader has to agree."""
    x = _tone(dur=0.5)
    path = str(tmp_path / "probe.wav")
    _write_wav_float32(path, x, SR)
    back, sr = _read_float_wav(path)
    assert sr == SR
    assert back.shape == x.shape
    assert np.max(np.abs(back - x)) < 1e-6


# ------------------------------------------------- the publication, on its own

def _session_with_a_render(tmp_path) -> Session:
    """A session that already holds one complete render."""
    root = str(tmp_path)
    sess = Session(root, sid="s1")
    sess.dir = os.path.join(root, "s1")
    sess.stems_dir = os.path.join(sess.dir, "stems")
    os.makedirs(sess.stems_dir, exist_ok=True)
    sess.mix_path = os.path.join(sess.dir, "mix.wav")
    sess.raw_path = os.path.join(sess.dir, "raw.wav")
    for name, blob in (("mix.wav", b"OLD-MIX"), ("raw.wav", b"OLD-RAW"),
                       ("mixplan.json", b'{"old": true}'),
                       ("report.md", b"OLD-REPORT"),
                       ("result.json", b'{"old": true}')):
        with open(os.path.join(sess.dir, name), "wb") as fh:
            fh.write(blob)
    sess.controls = {"style": "rock"}
    sess.result = {"old": True}
    sess.save()
    return sess


def _stage(sess, with_pair=True, result=None) -> str:
    """A staged render, as the child would leave it."""
    stage = sess.dir + "/.stage-probe"
    os.makedirs(stage, exist_ok=True)
    for name, blob in (("mix.wav", b"NEW-MIX"), ("raw.wav", b"NEW-RAW"),
                       ("mixplan.json", b'{"new": true}'),
                       ("report.md", b"NEW-REPORT")):
        with open(os.path.join(stage, name), "wb") as fh:
            fh.write(blob)
    if with_pair:
        for name in ("monitor_mix.wav", "monitor_raw.wav"):
            with open(os.path.join(stage, name), "wb") as fh:
                fh.write(b"FLOAT-PAIR")
    return stage


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _publish(sess, stage, result=None):
    payload = result if result is not None else {
        "style": {"id": "pop_2020s"},
        "exported": [],
        "monitoring": {"status": "matched", "level_lufs": -12.0},
    }
    return _publish_render(sess, stage, payload, {"style": "pop_2020s"},
                           prev_style="rock", reverb_amount=1.0)


MATCHED = {"status": "matched", "level_lufs": -12.0}


def test_a_claimed_pair_that_is_missing_fails_the_publication(tmp_path):
    """A render that says it is level matched must deliver the pair."""
    sess = _session_with_a_render(tmp_path)
    stage = _stage(sess, with_pair=False)
    with pytest.raises(PublishError):
        _publish(sess, stage)
    # the previous render is untouched
    assert _read(sess.mix_path) == b"OLD-MIX"
    assert _read(os.path.join(sess.dir, "raw.wav")) == b"OLD-RAW"
    assert _read(os.path.join(sess.dir, "result.json")) == b'{"old": true}'
    assert sess.result == {"old": True}


def test_a_failure_in_the_middle_of_the_pair_keeps_the_previous_one(tmp_path,
                                                                    monkeypatch):
    """The pair is published with the render, so it rolls back with it."""
    sess = _session_with_a_render(tmp_path)
    for name, blob in (("monitor_mix.wav", b"OLD-MON-MIX"),
                       ("monitor_raw.wav", b"OLD-MON-RAW")):
        with open(os.path.join(sess.dir, name), "wb") as fh:
            fh.write(blob)
    stage = _stage(sess, with_pair=True)

    real_replace = os.replace

    def failing(src, dst, *args, **kwargs):
        if os.path.basename(str(dst)) == "monitor_raw.wav":
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(PublishError):
        _publish(sess, stage)

    assert _read(sess.mix_path) == b"OLD-MIX"
    assert _read(os.path.join(sess.dir, "monitor_mix.wav")) == b"OLD-MON-MIX"
    assert _read(os.path.join(sess.dir, "monitor_raw.wav")) == b"OLD-MON-RAW"
    assert _read(os.path.join(sess.dir, "result.json")) == b'{"old": true}'


def test_a_render_without_a_claim_may_publish_without_the_pair(tmp_path):
    """An old style result has no pair to deliver; that is not a failure."""
    sess = _session_with_a_render(tmp_path)
    stage = _stage(sess, with_pair=False)
    _publish(sess, stage, result={"style": {"id": "pop_2020s"}, "exported": []})
    assert _read(sess.mix_path) == b"NEW-MIX"
    assert sess.monitor_mix_path is None
    assert sess.monitor_raw_path is None


def test_a_published_pair_is_restored_with_its_session(tmp_path):
    """A reopened project keeps the pair it published."""
    sess = _session_with_a_render(tmp_path)
    stage = _stage(sess, with_pair=True)
    _publish(sess, stage)

    reopened = Session.load(sess.dir.rsplit("/s1", 1)[0], "s1")
    assert reopened is not None
    assert reopened.monitor_mix_path == os.path.join(reopened.dir, "monitor_mix.wav")
    assert reopened.monitor_raw_path == os.path.join(reopened.dir, "monitor_raw.wav")
    assert reopened.result["monitoring"]["status"] == "matched"


def test_an_old_session_cannot_claim_a_matched_pair(tmp_path):
    """A session from before the pair existed says nothing about matching."""
    sess = _session_with_a_render(tmp_path)
    reopened = Session.load(sess.dir.rsplit("/s1", 1)[0], "s1")
    assert reopened is not None
    assert reopened.monitor_mix_path is None
    assert reopened.monitor_raw_path is None
    assert "monitoring" not in (reopened.result or {})


# ------------------------------------------------- over HTTP, end to end

@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("fair_ab_sessions"))
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
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _new_session(base) -> str:
    code, body = _req(base, "/api/session", data=b"", method="POST")
    assert code == 200
    return json.loads(body)["id"]


def _upload(base, sid, name, data):
    return _req(base, f"/api/session/{sid}/stem", data=data, method="POST",
                 headers={"X-Filename": name,
                          "Content-Type": "application/octet-stream"})


def test_a_render_publishes_a_level_matched_pair(srv, tmp_path):
    """The whole claim, over HTTP, on synthesized stems."""
    base, root = srv
    sid = _new_session(base)
    assert _upload(base, sid, "kick.wav", _wav_bytes(60.0))[0] == 200
    assert _upload(base, sid, "bass.wav", _wav_bytes(220.0))[0] == 200

    code, body = _req(base, f"/api/session/{sid}/run",
                      data=json.dumps({"style": "pop_2020s"}).encode(),
                      method="POST",
                      headers={"Content-Type": "application/json"})
    assert code == 200, body[:300]
    result = json.loads(body)
    mon = result["monitoring"]
    assert mon["status"] == "matched", mon
    assert mon["level_lufs"] is not None
    assert mon["export_lufs"] == result["lufs"]
    assert mon["mix_gain_db"] is not None and mon["raw_gain_db"] is not None

    sess_dir = os.path.join(root, sid)
    for name in ("monitor_mix.wav", "monitor_raw.wav"):
        assert os.path.exists(os.path.join(sess_dir, name)), name

    # the two views are the same loudness
    views = {}
    for kind in ("monitor/mix", "monitor/raw"):
        code, data = _req(base, f"/api/session/{sid}/audio/{kind}")
        assert code == 200, kind
        path = tmp_path / (kind.replace("/", "_") + ".wav")
        path.write_bytes(data)
        arr, sr = _read_float_wav(str(path))
        views[kind] = dsp.loudness_lufs(arr, sr)
    assert abs(views["monitor/mix"] - views["monitor/raw"]) < 0.1, views

    # the mix view is the delivered mix times one constant
    code, data = _req(base, f"/api/session/{sid}/audio/mix")
    assert code == 200
    export = tmp_path / "export_mix.wav"
    export.write_bytes(data)
    from automix import analyze
    mix_in, mix_sr = analyze.load_stem_file(str(export))
    code, data = _req(base, f"/api/session/{sid}/audio/monitor/mix")
    mon_mix = tmp_path / "monitor_mix.wav"
    mon_mix.write_bytes(data)
    mon_arr, _ = _read_float_wav(str(mon_mix))
    gain, spread = _gain_and_spread(mon_arr, mix_in.astype(np.float64))
    assert spread < 1e-4, f"the monitoring mix is not one gain: {spread}"
    assert abs(dsp.lin_to_db(gain) - mon["mix_gain_db"]) < 0.01

    # the export is still the export: a different file, and the download
    # route is not the monitoring route
    code, data = _req(base, f"/api/session/{sid}/download/mix")
    assert code == 200
    assert data[:4] == b"RIFF"
    assert data[:4] != b"\x00\x00\x00\x00"
    export_bytes = export.read_bytes()
    assert data == export_bytes, "the download is not the mix"

    # the peaks follow the audible view
    for kind in ("monitor_mix", "monitor_raw"):
        code, data = _req(base, f"/api/session/{sid}/peaks?kind={kind}")
        assert code == 200, kind
        assert json.loads(data)["peaks"]

    # the served result still says the pair is level matched
    code, data = _req(base, f"/api/session/{sid}/result")
    assert code == 200
    assert json.loads(data)["monitoring"]["status"] == "matched"


def test_an_invalidated_render_stops_serving_the_pair(srv):
    """The pair belongs to a render: when the render is gone, so is the pair."""
    base, root = srv
    sid = _new_session(base)
    assert _upload(base, sid, "kick.wav", _wav_bytes(60.0))[0] == 200
    assert _upload(base, sid, "bass.wav", _wav_bytes(220.0))[0] == 200
    code, body = _req(base, f"/api/session/{sid}/run",
                      data=json.dumps({"style": "pop_2020s"}).encode(),
                      method="POST",
                      headers={"Content-Type": "application/json"})
    assert code == 200, body[:300]
    result = json.loads(body)
    track = result["tracks"][0]["id"]

    # the pair is there while the render is
    assert _req(base, f"/api/session/{sid}/audio/monitor/mix")[0] == 200
    assert _req(base, f"/api/session/{sid}/audio/monitor/raw")[0] == 200
    assert _req(base, f"/api/session/{sid}/peaks?kind=monitor_raw")[0] == 200

    # a pan invalidates the render
    code, body = _req(base, f"/api/session/{sid}/pan",
                      data=json.dumps({"track_id": track, "pan": 0.4}).encode(),
                      method="POST",
                      headers={"Content-Type": "application/json"})
    assert code == 200, body[:200]

    # the export is gone...
    assert _req(base, f"/api/session/{sid}/audio/mix")[0] == 404
    # ...and the comparison with it: a cached path is not a current one
    assert _req(base, f"/api/session/{sid}/audio/monitor/mix")[0] == 404
    assert _req(base, f"/api/session/{sid}/audio/monitor/raw")[0] == 404
    assert _req(base, f"/api/session/{sid}/peaks?kind=monitor_raw")[0] == 404
    assert _req(base, f"/api/session/{sid}/peaks?kind=monitor_mix")[0] == 404


def test_a_pair_invalidated_before_the_open_is_not_served(srv, monkeypatch):
    """The validity check and the open happen in one critical section.

    The sequence is modelled exactly: the request is on its way and has not
    opened anything, the edit lands, and only then does the file get
    opened.  Because the path is resolved inside the same lock that opens
    the file, the request sees the invalidated session and says "not ready"
    instead of serving a pair that belongs to a render that is gone.
    """
    base, root = srv
    sid = _new_session(base)
    assert _upload(base, sid, "kick.wav", _wav_bytes(60.0))[0] == 200
    assert _upload(base, sid, "bass.wav", _wav_bytes(220.0))[0] == 200
    code, body = _req(base, f"/api/session/{sid}/run",
                      data=json.dumps({"style": "pop_2020s"}).encode(),
                      method="POST",
                      headers={"Content-Type": "application/json"})
    assert code == 200, body[:300]
    track = json.loads(body)["tracks"][0]["id"]
    # the pair is there while the render is
    assert _req(base, f"/api/session/{sid}/audio/monitor/mix")[0] == 200

    entered = threading.Event()
    release = threading.Event()
    original = srvmod.Handler._serve_file

    def delayed(self, path, ctype, *args, **kwargs):
        # the request is on its way and has not opened anything yet
        entered.set()
        release.wait(20)
        return original(self, path, ctype, *args, **kwargs)

    monkeypatch.setattr(srvmod.Handler, "_serve_file", delayed)
    outcome = {}

    def fetch():
        outcome["code"] = _req(
            base, f"/api/session/{sid}/audio/monitor/mix")[0]

    thread = threading.Thread(target=fetch)
    thread.start()
    assert entered.wait(20), "the request never reached the file"
    # the edit lands while the request is on its way
    code, body = _req(base, f"/api/session/{sid}/pan",
                      data=json.dumps({"track_id": track, "pan": 0.3}).encode(),
                      method="POST",
                      headers={"Content-Type": "application/json"})
    assert code == 200, body[:200]
    release.set()
    thread.join(30)
    assert outcome.get("code") == 404, outcome


def test_a_missing_pair_file_is_not_advertised_as_matched(srv):
    """A result that claims a pair the session does not have is downgraded."""
    base, root = srv
    sid = "pairmissing"
    sess_dir = os.path.join(root, sid)
    os.makedirs(os.path.join(sess_dir, "stems"), exist_ok=True)
    with open(os.path.join(sess_dir, "mix.wav"), "wb") as fh:
        fh.write(_wav_bytes())
    with open(os.path.join(sess_dir, "monitor_mix.wav"), "wb") as fh:
        fh.write(_wav_bytes())
    # monitor_raw.wav is missing on purpose
    result = {
        "style": {"id": "rock"},
        "lufs": -9.0,
        "monitoring": {"status": "matched", "level_lufs": -12.0,
                       "mix_gain_db": -1.0, "raw_gain_db": -2.0},
    }
    with open(os.path.join(sess_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    with open(os.path.join(sess_dir, "session.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": sid, "has_mix": True}, fh)

    sess = srvmod.Session.load(root, sid)
    assert sess is not None
    srvmod._sessions[sid] = sess

    # the served result must not claim a matched comparison
    code, body = _req(base, f"/api/session/{sid}/result")
    assert code == 200
    mon = json.loads(body)["monitoring"]
    assert mon["status"] != "matched", mon
    assert mon["reason"] == "file-incomplete", mon
    # and the route does not serve what is not there
    assert _req(base, f"/api/session/{sid}/audio/monitor/raw")[0] == 404
    assert _req(base, f"/api/session/{sid}/audio/monitor/mix")[0] == 404
    # the export is untouched
    assert _req(base, f"/api/session/{sid}/audio/mix")[0] == 200


def test_a_legacy_session_serves_no_monitoring_pair(srv, tmp_path):
    """An old session has no pair: the route says so instead of inventing one."""
    base, root = srv
    sid = _new_session(base)
    sess_dir = os.path.join(root, sid)
    os.makedirs(sess_dir, exist_ok=True)
    with open(os.path.join(sess_dir, "mix.wav"), "wb") as fh:
        fh.write(_wav_bytes())
    with open(os.path.join(sess_dir, "raw.wav"), "wb") as fh:
        fh.write(_wav_bytes())
    with open(os.path.join(sess_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({"style": {"id": "rock"}, "lufs": -9.0}, fh)

    code, _ = _req(base, f"/api/session/{sid}/audio/monitor/raw")
    assert code == 404, "an old session must not serve a monitoring pair"
    code, _ = _req(base, f"/api/session/{sid}/audio/monitor/mix")
    assert code == 404
