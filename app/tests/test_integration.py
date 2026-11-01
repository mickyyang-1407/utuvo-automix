"""End-to-end journey: the way a person actually uses this.

Unit tests prove each rule; wiring tests prove each field reaches the audio.
This one proves the *combination* works -- uploading, overriding roles, trimming,
panning, muting, a reference, a loudness target, rendering, downloading, and
resuming after a restart.  Feature interactions are where regressions hide.
"""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import threading

import pytest
from http.server import ThreadingHTTPServer

from automix import server as srvmod
from automix.server import Handler

SR = 48000


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    """A real server on a real socket; the journey is an HTTP conversation."""
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


def _wav(freq, dur=4.0, amp=0.3) -> bytes:
    n = int(dur * SR)
    t = np.arange(n) / SR
    mono = amp * np.sin(2 * np.pi * freq * t)
    pcm = (np.stack([mono, mono], 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _call(base, path, data=None, method="GET", headers=None):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _upload(base, sid, name, data):
    import urllib.parse

    return _call(
        base,
        f"/api/session/{sid}/stem",
        data=data,
        method="POST",
        headers={
            "X-Filename": urllib.parse.quote(name),
            "Content-Type": "application/octet-stream",
        },
    )


def _post_json(base, path, payload):
    return _call(
        base,
        path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def test_the_whole_journey(srv):
    base, _ = srv
    code, body, _ = _call(base, "/api/session", data=b"", method="POST")
    sid = json.loads(body)["id"]

    stems = {
        "drums_kick.wav": _wav(60.0),
        "bass.wav": _wav(55.0),
        "guitar_elec.wav": _wav(220.0),
        "vocal_lead.wav": _wav(440.0),
    }
    for name, data in stems.items():
        code, _, _ = _upload(base, sid, name, data)
        assert code == 200

    # 1. roles are detected before any render
    code, body, _ = _call(base, f"/api/session/{sid}/roles")
    roles = {n: v["role"] for n, v in json.loads(body)["roles"].items()}
    assert roles["drums_kick.wav"] == "kick"
    assert roles["vocal_lead.wav"] == "lead_vocal"

    # 2. the user disagrees about one role, nudges a level, pans a guitar, and
    #    drops the kick out of the mix
    code, body, _ = _post_json(
        base, f"/api/session/{sid}/roles", {"roles": {"guitar_elec.wav": "keys"}}
    )
    assert json.loads(body)["roles"]["guitar_elec.wav"]["role"] == "keys"

    code, _, _ = _post_json(base, f"/api/session/{sid}/trim", {"trims": {"vocal_lead": 1.5}})
    assert code == 200
    code, _, _ = _post_json(base, f"/api/session/{sid}/pan", {"track_id": "guitar_elec", "pan": 0.5})
    assert code == 200
    code, _, _ = _post_json(base, f"/api/session/{sid}/mute", {"track_id": "drums_kick", "muted": True})
    assert code == 200

    # 3. a reference and an explicit loudness target
    code, _, _ = _call(
        base,
        f"/api/session/{sid}/reference",
        data=_wav(110.0, dur=4.0, amp=0.15),
        method="POST",
        headers={"X-Filename": "reference.wav", "Content-Type": "application/octet-stream"},
    )
    assert code == 200

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/run",
        {"style": "pop_2020s", "target_lufs": -14.0, "blend": 0.5},
    )
    assert code == 200, body[:400]
    result = json.loads(body)

    # 4. everything the user asked for is visible in the result
    assert abs(result["lufs"] - (-14.0)) < 0.5
    ids = {t["id"] for t in result["tracks"]}
    assert "drums_kick" not in ids  # muted
    vocal = next(t for t in result["tracks"] if t["id"] == "vocal_lead")
    assert vocal["trim_db"] == 1.5
    guitar = next(t for t in result["tracks"] if t["id"] == "guitar_elec")
    assert abs(guitar["pan"] - 0.5) < 0.01
    assert guitar["role"] == "keys"  # the override stuck
    assert result["verdicts"], result["verdicts"]
    assert result["reference_passes"] == 2

    # 5. the receipt and the plan are consistent with the render
    code, body, _ = _call(base, f"/api/session/{sid}/report")
    assert code == 200 and "## Checks" in body.decode("utf-8")
    code, body, _ = _call(base, f"/api/session/{sid}/plan")
    plan = json.loads(body)
    assert plan["buses"]["master"]["target_lufs"] == -14.0

    # 6. the mix downloads as audio
    code, body, _ = _call(base, f"/api/session/{sid}/download/mix")
    assert code == 200 and body[:4] == b"RIFF"

    # 7. and the whole thing survives a server restart
    root = srvmod._sessions[sid].dir.rsplit("/", 1)[0]
    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    srvmod._load_sessions(root)

    code, body, _ = _call(base, f"/api/session/{sid}/state")
    state = json.loads(body)
    assert state["muted"] == ["drums_kick"]
    session = srvmod._sessions[sid]
    assert session.trims == {"vocal_lead": 1.5}
    assert session.pans == {"guitar_elec": 0.5}
    assert session.roles == {"guitar_elec.wav": "keys"}

def test_the_rest_of_the_gui(srv):
    """Everything the page can do that the main journey does not touch."""
    base, _ = srv
    code, body, _ = _call(base, "/api/session", data=b"", method="POST")
    sid = json.loads(body)["id"]

    for name, freq in (("drums_kick.wav", 60.0), ("vocal_lead.wav", 440.0)):
        code, _, _ = _upload(base, sid, name, _wav(freq))
        assert code == 200

    # the waveform the page draws
    code, body, _ = _call(base, f"/api/session/{sid}/peaks?kind=mix")
    assert code in (200, 404)  # nothing rendered yet: honest, not a crash

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/run",
        {"style": "pop_2020s", "export_stems": True},
    )
    assert code == 200, body[:400]
    result = json.loads(body)
    assert result["exported"], "the export checkbox has to produce files"

    code, body, _ = _call(base, f"/api/session/{sid}/peaks?kind=mix")
    assert code == 200
    peaks = json.loads(body)
    assert peaks.get("peaks"), peaks

    # auditioning a single stem
    code, body, _ = _call(base, f"/api/session/{sid}/audio/stem/vocal_lead.wav")
    assert code == 200 and body[:4] == b"RIFF"

    # the A/B button: the previous mix is kept
    code, _, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "lofi"})
    assert code == 200
    code, body, _ = _call(base, f"/api/session/{sid}/audio/prev")
    assert code == 200 and body[:4] == b"RIFF"

    # the processed stems download
    names = [e["name"] for e in result["exported"]]
    assert "README.txt" in names and "_master_chain.json" in names
    assert "_export_report.json" in names
    assert result["export_summary"]["comparison"]["reference_available"]
    code, metadata, headers = _call(base, f"/api/session/{sid}/processed/file/_export_report.json")
    assert code == 200
    assert json.loads(metadata)["files"] == result["export_summary"]["files"]
    for name in names:
        code, body, _ = _call(
            base, f"/api/session/{sid}/processed/file/{name}"
        )
        assert code == 200, name
        assert body, name

    # removing a stem drops it from the session
    code, body, _ = _post_json(base, f"/api/session/{sid}/remove", {"name": "drums_kick.wav"})
    assert code == 200
    assert "drums_kick.wav" not in json.loads(body)["stems"]

    # cancelling when nothing is running is not an error
    code, body, _ = _call(base, f"/api/session/{sid}/cancel", data=b"", method="POST")
    assert code == 200


def test_successful_rerender_prunes_only_stale_generated_exports(srv):
    from pathlib import Path
    base, root = srv
    code, body, _ = _call(base, '/api/session', data=b'', method='POST')
    assert code == 200
    sid=json.loads(body)['id']
    source_bytes={name:_wav(freq,dur=1) for name,freq in [('bass.wav',55),('vocal_lead.wav',440)]}
    for name,data in source_bytes.items():
        code,_,_=_upload(base,sid,name,data)
        assert code==200
    code,body,_=_post_json(base,f'/api/session/{sid}/run',{'style':'pop_2020s','export_stems':True})
    assert code==200,body
    first=json.loads(body)
    old_file=next(f['filename'] for f in first['export_summary']['files'] if f['source_id']=='bass')
    processed=Path(root)/sid/'processed'
    assert (processed/old_file).is_file()
    # Only ordinary generated child files are owned by the prune operation.
    keep=processed/'keep.bin';keep.write_bytes(b'unrelated')
    keep_dir=processed/'directory.wav';keep_dir.mkdir()
    linked=processed/'link.wav';linked.symlink_to(keep)
    (processed/'stale.json').write_text('{}')
    code,_,_=_post_json(base,f'/api/session/{sid}/mute',{'track_id':'bass','muted':True})
    assert code==200
    code,body,_=_post_json(base,f'/api/session/{sid}/run',{'style':'pop_2020s','export_stems':True})
    assert code==200,body
    current=json.loads(body)
    assert all(f['source_id']!='bass' for f in current['export_summary']['files'])
    assert old_file not in {f['name'] for f in current['exported']}
    assert not (processed/old_file).exists()
    assert not (processed/'stale.json').exists()
    assert keep.read_bytes()==b'unrelated' and keep_dir.is_dir() and linked.is_symlink()
    code,_,_=_call(base,f'/api/session/{sid}/processed/file/{old_file}')
    assert code==404
    # Muting removes an export, never the uploaded source audio.
    code,body,_=_call(base,f'/api/session/{sid}/audio/stem/bass.wav')
    assert code==200 and body==source_bytes['bass.wav']


def test_reference_tone_and_unmask_processing_evidence_reach_the_gui_payload(srv):
    base,_=srv
    code,body,_=_call(base,'/api/session',data=b'',method='POST')
    assert code==200
    sid=json.loads(body)['id'];t=np.arange(SR*4)/SR
    def encode(mono):
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as writer:
            writer.setnchannels(2);writer.setsampwidth(2);writer.setframerate(SR)
            writer.writeframes((np.column_stack([mono]*2)*32767).astype('<i2').tobytes())
        return buffer.getvalue()
    music=.2*np.sin(2*np.pi*110*t)+.08*np.sin(2*np.pi*3200*t)
    voice=(.15*np.sin(2*np.pi*660*t)+.06*np.sin(2*np.pi*2640*t))*((t>=1)&(t<3))
    for name,audio in [('guitar_elec.wav',music),('vocal_lead.wav',voice)]:
        code,_,_=_upload(base,sid,name,encode(audio));assert code==200
    settings={'style':'pop_2020s','balance_preserve':1,'export_stems':True}
    code,body,_=_post_json(base,f'/api/session/{sid}/run',settings)
    assert code==200,body
    first=json.loads(body)
    assert first['unmask']['targets']==['guitar_elec']
    effect=next(x for x in first['tracks'] if x['id']=='guitar_elec')['dynamic_eq']
    assert effect['duty_fraction']>0 and 0<effect['max_nominal_cut_db']<=1.5
    code,reference,_=_call(base,f'/api/session/{sid}/audio/mix');assert code==200
    code,_,_=_call(base,f'/api/session/{sid}/reference',data=reference,method='POST',
                   headers={'X-Filename':'self.wav','Content-Type':'application/octet-stream'})
    assert code==200
    code,body,_=_post_json(base,f'/api/session/{sid}/run',dict(settings,blend=1))
    assert code==200,body
    result=json.loads(body);tone=result['reference_tone']
    assert tone['band_count']>=2 and tone['after_db']<.1
    assert result['unmask']['targets']==['guitar_elec']
    code,plan,_=_call(base,f'/api/session/{sid}/plan');assert code==200
    plan=json.loads(plan)
    assert plan['style']['reference']['tone_comparison']['indices']==tone['indices']
    assert any(x.get('dynamic_eq') for x in plan['tracks'])
    code,html,_=_call(base,'/');assert code==200
    assert '不是音質分數' in html.decode() and '人聲讓位' in html.decode()
    # A silent reference is input failure with an actionable4xx response.
    code,_,_=_call(base,f'/api/session/{sid}/reference',data=encode(np.zeros_like(t)),method='POST',
                   headers={'X-Filename':'silent.wav','Content-Type':'application/octet-stream'})
    assert code==200
    code,body,_=_post_json(base,f'/api/session/{sid}/run',settings)
    assert code==400 and 'silent' in body.decode()
