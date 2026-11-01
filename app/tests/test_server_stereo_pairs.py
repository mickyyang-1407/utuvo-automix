"""The stereo pair over the real HTTP API of a real server.

A declared stereo pair has to survive the whole way the server takes a
declaration: it is saved, it is restored, it is restored in the order the
user declared (the order IS the channel identity), it makes the previous mix
stale when it really changed, and it leaves that mix alone when it did not.
A request it cannot vouch for is refused with a reason and changes nothing.

These tests drive the real server on 127.0.0.1:0 with their own tiny
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

from automix import server as srvmod
from automix.server import Handler, Session

SR = 48000
LEFT = "left.wav"
RIGHT = "right.wav"
DRUMS = "drums.wav"
#: the server stores the engine's own track ids, not the file names: the
#: file name is an alias for the same member
LEFT_ID = "left"
RIGHT_ID = "right"


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


def _pair_groups(left=LEFT, right=RIGHT, pan=0.0, trim=0.0):
    return [
        {
            "id": "stereo",
            "members": [left, right],
            "mode": "stereo_pair",
            "pan": pan,
            "trim_db": trim,
        }
    ]


def _session_with_stems(base):
    """A session with two distinguishable channels and a third track."""
    sid = _new_session(base)
    for name, payload in (
        (LEFT, _wav_bytes(freq=220.0)),
        (RIGHT, _wav_bytes(freq=330.0)),
        (DRUMS, _wav_bytes(freq=90.0)),
    ):
        code, body, _ = _upload(base, sid, name, payload)
        assert code == 200, body
    # the two members are one role: that is the user's declaration, not a
    # guess the detector made
    code, body, _ = _post_json(
        base,
        f"/api/session/{sid}/roles",
        {"roles": {LEFT: "guitar_electric", RIGHT: "guitar_electric", DRUMS: "drums_full"}},
    )
    assert code == 200, body
    return sid


def _session_with_a_pair(base, groups=None):
    sid = _session_with_stems(base)
    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": groups if groups is not None else _pair_groups()}
    )
    assert code == 200, body
    return sid


def _session_with_a_mix(base, groups=None):
    sid = _session_with_a_pair(base, groups)
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 200, body[:400]
    return sid


# ------------------------------------------------------------------ save/load


def test_a_pair_is_saved_with_its_mode_and_its_trim(srv):
    base, _ = srv
    sid = _session_with_stems(base)
    groups = _pair_groups(pan=0.5, trim=2.5)
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": groups})
    assert code == 200, body
    stored = json.loads(body)["capture_groups"]
    assert len(stored) == 1
    assert stored[0]["mode"] == "stereo_pair"
    assert stored[0]["trim_db"] == 2.5
    assert stored[0]["members"] == [LEFT_ID, RIGHT_ID]
    assert float(stored[0]["pan"]) == 0.5

    restored = _state(base, sid)["capture_groups"]
    assert restored == stored


def test_the_member_order_is_the_channel_identity(srv):
    base, _ = srv
    sid = _session_with_stems(base)
    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(left=RIGHT, right=LEFT)}
    )
    assert code == 200, body
    stored = json.loads(body)["capture_groups"]
    assert stored[0]["members"] == [RIGHT_ID, LEFT_ID]
    assert _state(base, sid)["capture_groups"][0]["members"] == [RIGHT_ID, LEFT_ID]


def test_a_pair_survives_a_restart(srv):
    base, root = srv
    sid = _session_with_pair_and_mix = _session_with_a_pair(base, _pair_groups(pan=-0.25, trim=1.5))
    reloaded = Session.load(root, sid)
    assert reloaded is not None
    assert len(reloaded.capture_groups) == 1
    group = reloaded.capture_groups[0]
    assert group["mode"] == "stereo_pair"
    assert group["trim_db"] == 1.5
    assert group["members"] == [LEFT_ID, RIGHT_ID]
    assert float(group["pan"]) == -0.25


def test_a_pair_is_restored_as_a_pair_and_not_as_a_same_source_group(srv):
    """The mode is not dropped on the way in: a pair restored as a same-source
    group is a different declaration wearing the same name."""
    base, root = srv
    sid = _session_with_a_pair(base, _pair_groups())
    reloaded = Session.load(root, sid)
    assert reloaded is not None
    assert reloaded.capture_groups[0]["mode"] == "stereo_pair"
    assert reloaded.capture_groups_error is None


def test_a_saved_null_mode_is_dropped_and_reported():
    """A mode that is present but is not the one literal mode is never read
    as "no mode": the saved declaration is dropped and reported, so an
    ambiguous pair intent cannot come back as a collapsed same-source group.
    """
    groups, error = srvmod._clean_capture_groups(
        [{"id": "stereo", "members": [LEFT_ID, RIGHT_ID], "pan": 0.0, "mode": None}]
    )
    assert groups == []
    assert error and "could not be restored" in error


def test_a_saved_pair_with_a_null_trim_is_dropped_and_reported():
    groups, error = srvmod._clean_capture_groups(
        [{"id": "stereo", "members": [LEFT_ID, RIGHT_ID], "pan": 0.0,
          "mode": "stereo_pair", "trim_db": None}]
    )
    assert groups == []
    assert error and "could not be restored" in error


def test_a_same_source_group_is_still_saved_the_old_way(srv):
    base, root = srv
    sid = _session_with_stems(base)
    groups = [{"id": "room", "members": [LEFT, RIGHT], "pan": -0.2}]
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": groups})
    assert code == 200, body
    stored = json.loads(body)["capture_groups"]
    assert stored[0] == {"id": "room", "members": [LEFT_ID, RIGHT_ID], "pan": -0.2}
    reloaded = Session.load(root, sid)
    assert reloaded is not None
    assert reloaded.capture_groups[0] == {"id": "room", "members": [LEFT_ID, RIGHT_ID], "pan": -0.2}


# ---------------------------------------------------------------- idempotence


def test_saving_the_same_pair_keeps_the_finished_mix(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _pair_groups())

    before_state = _state(base, sid)
    assert before_state["has_mix"] is True
    before_mix = _mix_bytes(base, sid)
    before_files = _files_under(root, sid)
    assert _result(base, sid)[0] == 200

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups()}
    )
    assert code == 200, body

    assert _state(base, sid) == before_state
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files
    assert _result(base, sid)[0] == 200


def test_the_same_pair_by_track_id_keeps_the_finished_mix(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _pair_groups())
    before_mix = _mix_bytes(base, sid)
    before_files = _files_under(root, sid)
    ids = _state(base, sid)["track_ids"]

    code, body, _ = _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": _pair_groups(left=ids[LEFT], right=ids[RIGHT])},
    )
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is True
    assert _mix_bytes(base, sid) == before_mix
    assert _files_under(root, sid) == before_files


# ---------------------------------------------------- a real change still counts


def test_a_changed_pair_position_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _pair_groups(pan=0.0))

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(pan=0.5)}
    )
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_a_changed_shared_trim_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _pair_groups(trim=0.0))

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(trim=1.0)}
    )
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_a_swapped_pair_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _pair_groups())

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(left=RIGHT, right=LEFT)}
    )
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_a_pair_that_became_a_same_source_group_invalidates_the_previous_mix(srv):
    base, _ = srv
    sid = _session_with_a_mix(base, _pair_groups())

    groups = [{"id": "stereo", "members": [LEFT, RIGHT], "pan": 0.0}]
    code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", {"groups": groups})
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


def test_a_same_source_group_that_became_a_pair_invalidates_the_previous_mix(srv):
    base, _ = srv
    groups = [{"id": "stereo", "members": [LEFT, RIGHT], "pan": 0.0}]
    sid = _session_with_a_mix(base, groups)

    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups()}
    )
    assert code == 200, body

    assert _state(base, sid)["has_mix"] is False
    assert _result(base, sid)[0] == 404


# ------------------------------------------------------------ refused requests


def test_an_invalid_pair_is_refused_and_keeps_the_state(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _pair_groups())
    before_state = _state(base, sid)
    before_files = _files_under(root, sid)
    before_result = _result(base, sid)

    bad_payloads = [
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "mode": "stereo"}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "mode": None}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "mode": "stereo_pair", "trim_db": 30.0}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "mode": "stereo_pair", "trim_db": None}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT, DRUMS], "mode": "stereo_pair"}]},
        {"groups": [{"id": "p", "members": [LEFT, LEFT], "mode": "stereo_pair"}]},
        {"groups": [{"id": "p", "members": [LEFT], "mode": "stereo_pair"}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "trim_db": 1.0}]},
        {"groups": [{"id": "p", "members": [LEFT, RIGHT], "mode": "stereo_pair", "pan": 4.0}]},
    ]
    for payload in bad_payloads:
        code, body, _ = _post_json(base, f"/api/session/{sid}/capture_groups", payload)
        assert code == 400, (payload, body)
        assert _state(base, sid) == before_state
        assert _files_under(root, sid) == before_files
        assert _result(base, sid) == before_result


def test_a_busy_session_refuses_the_pair_and_keeps_everything(srv):
    base, root = srv
    sid = _session_with_a_mix(base, _pair_groups())
    before_state = _state(base, sid)
    before_files = _files_under(root, sid)
    before_result = _result(base, sid)

    sess = srvmod._sessions.get(sid)
    assert sess is not None
    sess.busy = True
    try:
        code, body, _ = _post_json(
            base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(trim=3.0)}
        )
        assert code == 409, body
    finally:
        sess.busy = False

    assert _state(base, sid) == before_state
    assert _files_under(root, sid) == before_files
    assert _result(base, sid) == before_result


def test_a_pair_that_cannot_be_planned_is_refused_by_the_run(srv):
    """The declaration is accepted, but the mix refuses a pair the engine
    cannot honour -- here a member that is not in the mix."""
    base, _ = srv
    sid = _session_with_stems(base)
    code, body, _ = _post_json(
        base, f"/api/session/{sid}/capture_groups", {"groups": _pair_groups(right="ghost.wav")}
    )
    assert code == 200, body
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 400, body
