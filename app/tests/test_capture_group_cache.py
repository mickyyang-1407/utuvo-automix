"""Saving the same capture-group declaration must not throw the mix away.

The web app posts the capture groups before every run.  When the user posts
the declaration they already made -- unchanged -- the server used to clear
the cached mix and the result, so the session answered "no mix" and the
result endpoint answered 404 while every file was still on disk.

These tests drive the real HTTP API of a real server, with their own tiny
synthesised stems: no string matching, no mirrored implementation.
"""

from __future__ import annotations

import hashlib
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

from automix.server import Handler

SR = 48000
PAIR = ("guitar_01_L.wav", "guitar_01_R.wav")


# ------------------------------------------------------------------ the server


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


# ------------------------------------------------------------------- helpers


def _req(base, path, data=None, method="GET", headers=None):
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _post_json(base, path, payload):
    return _req(
        base,
        path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def _wav_bytes(freq=220.0, dur=1.0, sr=SR, amp=0.3) -> bytes:
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
    return _req(
        base,
        f"/api/session/{sid}/stem",
        data=data,
        method="POST",
        headers={"X-Filename": name, "Content-Type": "application/octet-stream"},
    )


def _state(base, sid) -> dict:
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200, body
    return json.loads(body)


def _result(base, sid):
    code, body, _ = _req(base, f"/api/session/{sid}/result")
    return code, body


def _mix_bytes(base, sid) -> bytes:
    code, body, _ = _req(base, f"/api/session/{sid}/audio/mix")
    assert code == 200, body[:200]
    return body


def _files_under(root, sid) -> dict:
    """Every file of the session, with its digest: nothing may move."""
    digests = {}
    base = os.path.join(root, sid)
    for folder, _dirs, names in os.walk(base):
        for name in names:
            path = os.path.join(folder, name)
            with open(path, "rb") as fh:
                digests[os.path.relpath(path, base)] = hashlib.sha256(fh.read()).hexdigest()
    return digests


def _session_with_a_mix(base, groups, extra=None):
    """A session with three of our own stems, the given groups and a mix."""
    sid = _new_session(base)
    pair = _wav_bytes(freq=220.0)
    for name, payload in (
        (PAIR[0], pair),
        (PAIR[1], pair),          # the same source, recorded by two mics
        ("bass.wav", _wav_bytes(freq=55.0)),
    ):
        code, body, _ = _upload(base, sid, name, payload)
        assert code == 200, body
    if extra is not None:
        extra(base, sid)
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": groups})
    assert code == 200, body
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 200, body[:400]
    return sid


def _groups(pan=0.0, members=None):
    return [
        {
            "id": "guitars",
            "members": list(members if members is not None else PAIR),
            "pan": pan,
        }
    ]


# ----------------------------------------------------------------- the counterexample


def test_saving_the_same_groups_keeps_the_finished_mix(srv):
    """The bug: posting the groups the user already declared threw the
    finished mix and the result away."""
    base, root = srv
    sid = _session_with_a_mix(base, _groups())

    before_state = _state(base, sid)
    assert before_state["has_mix"] is True
    before_mix = _mix_bytes(base, sid)
    before_result = _result(base, sid)
    assert before_result[0] == 200
    before_files = _files_under(root, sid)

    # exactly the declaration that is already stored
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": _groups()})
    assert code == 200, body

    after_state = _state(base, sid)
    assert after_state["has_mix"] is True
    assert after_state == before_state
    assert _result(base, sid) == before_result
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files


def test_the_same_declaration_by_track_id_keeps_the_finished_mix(srv):
    """The file name and the engine's track id are the same member."""
    base, root = srv
    sid = _session_with_a_mix(base, _groups())
    track_ids = _state(base, sid)["track_ids"]
    by_id = _groups(members=[track_ids[name] for name in PAIR])

    before_mix = _mix_bytes(base, sid)
    before_files = _files_under(root, sid)

    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": by_id})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is True
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files


def test_a_pan_of_zero_and_zero_point_zero_are_the_same_declaration(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _groups(pan=0))
    before_mix = _mix_bytes(base, sid)
    before_files = _files_under(root, sid)

    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": _groups(pan=0.0)})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is True
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files


# ------------------------------------------------------------- a real change still counts


def test_a_changed_pan_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _groups(pan=0.0))

    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": _groups(pan=0.3)})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_a_changed_member_list_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _groups())

    track_ids = _state(base, sid)["track_ids"]
    changed = _groups(members=[track_ids[PAIR[0]], track_ids["bass.wav"]])
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": changed})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_an_added_group_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _groups())

    added = _groups() + [{"id": "lows", "members": ["bass.wav"], "pan": 0.0}]
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": added})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_removed_groups_invalidate_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _groups())

    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": []})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


# ------------------------------------------------------------- refused requests


def test_an_invalid_payload_is_refused_and_keeps_the_state(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _groups())
    before_state = _state(base, sid)
    before_files = _files_under(root, sid)

    bad_payloads = [
        {"groups": "not-a-list"},
        {"groups": [{"id": "guitars", "members": "guitar_01_L.wav"}]},
        {"groups": [{"id": "guitars", "members": list(PAIR), "pan": 5.0}]},
        {"groups": [{"id": "", "members": list(PAIR)}]},
    ]
    for payload in bad_payloads:
        code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", payload)
        assert code == 400, (payload, body)
        assert _state(base, sid) == before_state
        assert _files_under(root, sid) == before_files
        assert _result(base, sid)[0] == 200


def test_saving_the_same_groups_touches_no_other_control(srv):
    """The manual pan the user set, the roles and the mix are all still there."""
    base, root = srv
    track_ids = {}

    def set_a_pan(base, sid):
        ids = _state(base, sid)["track_ids"]
        code, body, _ = _post_json(
            base, f"/api/session/{sid}/pan", {"track_id": ids["bass.wav"], "pan": 0.25}
        )
        assert code == 200, body

    sid = _session_with_a_mix(base, _groups(), extra=set_a_pan)
    before_state = _state(base, sid)
    assert before_state["pans"]
    before_mix = _mix_bytes(base, sid)
    before_files = _files_under(root, sid)

    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": _groups()})
    assert code == 200, body

    after_state = _state(base, sid)
    assert after_state["pans"] == before_state["pans"]
    assert after_state["muted"] == before_state["muted"]
    assert after_state["manual_roles"] == before_state["manual_roles"]
    assert after_state["controls"] == before_state["controls"]
    assert after_state["capture_groups"] == before_state["capture_groups"]
    assert after_state["has_mix"] is True
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files
