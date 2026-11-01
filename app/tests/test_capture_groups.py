"""Tests for the declared capture groups.

A capture group is the user's explicit statement that several imported tracks
are the same physical source recorded by different microphones.  These tests
cover the three things the contract promises:

* what a group protects (the imported relationship, before the bus/master),
* what it refuses (an invalid declaration, before anything is rendered),
* what it must not change (every track that is not in a group).

Everything here is synthetic: short WAV files written into a tmp folder, no
audio from disk and no network.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from automix import (
    analyze,
    capture_groups as capmod,
    cli,
    dsp,
    pipeline,
    plan as planner,
    render,
    server as srvmod,
)

SR = 48000


# ------------------------------------------------------------------ fixtures

def _write(path: str, audio: np.ndarray, sr: int = SR) -> str:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1)


def _load(folder: str, roles: dict | None = None) -> list:
    stems = analyze.load_stems(folder)
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    for s in stems:
        if roles and s.name in roles:
            s.role = roles[s.name]
    return stems


def _preset() -> dict:
    return planner.load_preset("pop_2020s")


def _plan(stems, **kw) -> dict:
    return planner.build_plan(stems, _preset(), **kw)


def _track(plan: dict, tid: str) -> dict:
    return next(t for t in plan["tracks"] if t["id"] == tid)


def _source_of(stems, name: str) -> np.ndarray:
    return next(s.audio for s in stems if s.name == name)


def _room_files(folder) -> dict:
    """Two microphones of one source, a third of the same role, and drums."""
    n = int(1.5 * SR)
    t = np.arange(n) / SR
    src = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    _write(str(folder / "mic_a.wav"), _stereo(src))
    _write(str(folder / "mic_b.wav"), _stereo(src * 0.5))
    _write(str(folder / "mic_c.wav"), _stereo(0.25 * np.sin(2 * np.pi * 330 * t)))
    _write(str(folder / "drums.wav"), _stereo(0.2 * np.sin(2 * np.pi * 90 * t)))
    return {
        "mic_a.wav": "guitar_electric",
        "mic_b.wav": "guitar_electric",
        "mic_c.wav": "guitar_electric",
        "drums.wav": "drums_full",
    }


@pytest.fixture()
def room(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    roles = _room_files(d)
    return _load(str(d), roles)


# ------------------------------------------------- no groups: the old path

def test_a_plan_without_groups_carries_no_receipt(room):
    plan = _plan(room)
    assert "capture_groups" not in plan


def test_an_empty_group_list_is_the_default_path(room):
    plain = _plan(room)
    empty = _plan(room, capture_groups=[])
    assert empty["tracks"] == plain["tracks"]
    assert empty["source"] == plain["source"]
    assert "capture_groups" not in empty


def test_the_audio_is_identical_without_any_declaration(room):
    _, plain, _ = pipeline.mix(room, _preset())
    _, empty, _ = pipeline.mix(room, _preset(), capture_groups=[])
    assert plain.shape == empty.shape
    assert float(np.max(np.abs(plain - empty))) == 0.0


# ------------------------------------------------------- what a group does

def test_a_declared_member_is_protected(room):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.25}])
    for tid in ("mic_a", "mic_b"):
        t = _track(plan, tid)
        assert t["gain_db"] == 0.0
        assert t["hpf_hz"] == 20.0
        assert t["eq"] == []
        assert t.get("comp") is None
        assert t.get("deess") is None
        assert t.get("ride") is None
        assert t.get("dynamic_eq") is None
        assert t.get("level_match") is None
        assert t["saturation"] == 0.0
        assert t["width"] == 1.0
        assert t["sends"] == {"plate": 0.0}
        assert t["pan"] == -0.25


def test_the_protection_removes_a_dynamic_eq_that_was_already_planned():
    tracks = [
        {
            "id": "mic_a", "gain_db": 3.0, "trim_db": 0.0, "pan": 0.2, "hpf_hz": 90.0,
            "eq": [{"type": "bell", "freq": 3200.0, "gain_db": -2.0, "q": 1.0, "enabled": True}],
            "comp": {"ratio": 2.0}, "deess": {"amount": 0.5}, "ride": [[0.0, 1.0]],
            "dynamic_eq": {"mode": "whatever"}, "level_match": {"mode": "active_lufs_blend"},
            "saturation": 0.4, "width": 1.4, "sends": {"plate": 0.5},
        },
        {
            "id": "mic_b", "gain_db": -1.0, "pan": -0.2, "hpf_hz": 90.0, "eq": [],
            "saturation": 0.2, "width": 1.0, "sends": {"plate": 0.4},
        },
    ]
    facts = [
        {"id": "mic_a", "role": "guitar_electric", "sample_rate": 48000, "duration_s": 1.0,
         "dual_mono": True, "silent": False},
        {"id": "mic_b", "role": "guitar_electric", "sample_rate": 48000, "duration_s": 1.0,
         "dual_mono": True, "silent": False},
    ]
    groups = capmod.normalize([{"id": "g", "members": ["mic_a", "mic_b"]}], facts)
    receipt, _ = capmod.apply_protection(tracks, groups)
    t = tracks[0]
    assert t["gain_db"] == 0.0
    assert t["hpf_hz"] == 20.0
    assert t["eq"] == []
    assert "comp" not in t and "deess" not in t and "ride" not in t
    assert "dynamic_eq" not in t and "level_match" not in t
    assert t["saturation"] == 0.0 and t["width"] == 1.0
    assert t["sends"] == {"plate": 0.0}
    assert t["pan"] == 0.0
    assert receipt["groups"][0]["protected_tracks"] == ["mic_a", "mic_b"]


def test_the_protection_refuses_a_member_that_is_not_in_the_mix():
    tracks = [{"id": "mic_a", "gain_db": 0.0, "pan": 0.0, "hpf_hz": 20.0, "eq": [],
               "saturation": 0.0, "width": 1.0, "sends": {"plate": 0.0}}]
    facts = [
        {"id": "mic_a", "role": "guitar_electric", "sample_rate": 48000, "duration_s": 1.0,
         "dual_mono": True, "silent": False},
        {"id": "mic_b", "role": "guitar_electric", "sample_rate": 48000, "duration_s": 1.0,
         "dual_mono": True, "silent": False},
    ]
    groups = capmod.normalize([{"id": "g", "members": ["mic_a", "mic_b"]}], facts)
    # mic_b is not in the mix: the group is refused, never quietly reduced
    with pytest.raises(capmod.CaptureGroupError) as exc:
        capmod.apply_protection(tracks, groups)
    assert "never silently reduced" in str(exc.value)


def test_a_track_outside_the_group_keeps_its_numbers(room):
    plain = _plan(room)
    grouped = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"]}])
    for tid in ("mic_c", "drums"):
        assert _track(grouped, tid) == _track(plain, tid)


def test_a_same_role_track_outside_the_group_is_not_protected(room):
    grouped = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"]}])
    # mic_c has the same role as the members and is a different source: it
    # keeps its own ride, gain and pan
    outside = _track(grouped, "mic_c")
    assert outside.get("ride")
    assert outside["gain_db"] != 0.0
    assert outside["pan"] != _track(grouped, "mic_a")["pan"] or outside["pan"] == 0.0


def test_the_imported_arrays_are_not_touched(room):
    before = [s.audio.copy() for s in room]
    pipeline.mix(
        room,
        _preset(),
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}],
    )
    for stem, original in zip(room, before):
        assert np.array_equal(stem.audio, original)


def test_the_group_has_one_common_pan(room):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": 0.4}])
    assert _track(plan, "mic_a")["pan"] == 0.4
    assert _track(plan, "mic_b")["pan"] == 0.4


@pytest.mark.parametrize("pan", [-1.0, 0.0, 1.0])
def test_the_group_pan_may_be_an_extreme(room, pan):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": pan}])
    assert _track(plan, "mic_a")["pan"] == pan
    assert _track(plan, "mic_b")["pan"] == pan


def test_the_collected_member_is_the_imported_audio_panned(room):
    collect = {}
    pipeline.mix(
        room,
        _preset(),
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}],
        collect=collect,
    )
    imported = _source_of(room, "mic_a.wav")
    expected = render._pan(imported, -0.3)
    assert float(np.max(np.abs(collect["stems"]["mic_a"] - expected))) < 1e-12


def test_an_explicit_trim_is_applied_on_top_of_zero(room):
    collect = {}
    plan, _, _ = pipeline.mix(
        room,
        _preset(),
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": 0.2}],
        trims={"mic_a": 3.0},
        collect=collect,
    )
    assert _track(plan, "mic_a")["gain_db"] == 3.0
    assert _track(plan, "mic_a")["trim_db"] == 3.0
    imported = _source_of(room, "mic_a.wav")
    expected = render._pan(imported * dsp.db_to_lin(3.0), 0.2)
    assert float(np.max(np.abs(collect["stems"]["mic_a"] - expected))) < 1e-12
    # the trim breaks the imported relative level, and the receipt says so
    assert any("manual trim" in w for w in plan["warnings"])


def test_a_manual_pan_wins_over_the_group_pan(room):
    plan = _plan(
        room,
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}],
        pans={"mic_a": 0.5},
    )
    assert _track(plan, "mic_a")["pan"] == 0.5
    assert _track(plan, "mic_b")["pan"] == -0.3
    assert any("keeps its own pan" in w for w in plan["warnings"])


def test_a_delayed_and_inverted_member_is_not_repaired(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    src = 0.3 * np.sin(2 * np.pi * 220 * t)
    _write(str(d / "mic_a.wav"), _stereo(src))
    _write(str(d / "mic_b.wav"), _stereo(np.concatenate([np.zeros(120), src])[:n]))
    _write(str(d / "mic_c.wav"), _stereo(-src))
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric",
                           "mic_c.wav": "guitar_electric"})
    collect = {}
    pipeline.mix(
        stems,
        _preset(),
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b", "mic_c"]}],
        collect=collect,
    )
    a = collect["stems"]["mic_a"]
    b = collect["stems"]["mic_b"]
    c = collect["stems"]["mic_c"]
    # the delay is still there: nothing aligned it
    assert float(np.max(np.abs(b[:120]))) < 1e-12
    assert float(np.max(np.abs(b[120:] - a[:-120]))) < 1e-9
    # the polarity is still inverted: nothing flipped it
    assert float(np.max(np.abs(c + a))) < 1e-12


# ------------------------------------------------------------- the receipt

def test_the_receipt_says_what_was_protected_and_where_it_stops(room):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}])
    receipt = plan["capture_groups"]
    assert receipt["mode"] == "declared_capture_group"
    group = receipt["groups"][0]
    assert group["id"] == "room"
    assert group["members"] == ["mic_a", "mic_b"]
    assert group["protected_tracks"] == ["mic_a", "mic_b"]
    assert group["pan"] == -0.3
    assert "does not align time" in receipt["boundary"]
    planner.validate(plan)


def test_the_validator_refuses_a_receipt_that_claims_a_missing_track(room):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"]}])
    plan["capture_groups"]["groups"][0]["protected_tracks"] = ["not_in_this_mix"]
    with pytest.raises(planner.PlanError) as exc:
        planner.validate(plan)
    assert "not in this mix" in str(exc.value)


def test_a_saved_plan_replays_its_numbers_not_its_receipt(room):
    plan = _plan(room, capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}])
    edited = json.loads(json.dumps(plan))
    for t in edited["tracks"]:
        if t["id"] == "mic_a":
            t["gain_db"] = -6.0
    collect = {}
    render.render(room, edited, collect=collect)
    imported = _source_of(room, "mic_a.wav")
    expected = render._pan(imported * dsp.db_to_lin(-6.0), -0.3)
    assert float(np.max(np.abs(collect["stems"]["mic_a"] - expected))) < 1e-12


# ------------------------------------------------------------- validation

def _reject(stems, groups, needle=None):
    with pytest.raises(planner.PlanError) as exc:
        _plan(stems, capture_groups=groups)
    if needle:
        assert needle in str(exc.value)
    return str(exc.value)


def _reject_before_render(stems, groups, monkeypatch, needle=None):
    """A declaration that cannot be honoured must not reach the renderer.

    The renderer is replaced with a tripwire: if the mix gets that far, the
    declaration was not refused early enough.
    """
    entered = []

    def forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError("the renderer was reached for an unsupported declaration")

    monkeypatch.setattr(render, "render", forbidden)
    with pytest.raises(planner.PlanError) as exc:
        pipeline.mix(stems, _preset(), capture_groups=groups)
    assert not entered
    if needle:
        assert needle in str(exc.value)
    return str(exc.value)


def test_a_declaration_that_is_not_a_list_is_rejected(room):
    _reject(room, {"id": "room"}, "must be a list")


def test_a_group_that_is_not_an_object_is_rejected(room):
    _reject(room, ["room"], "must be an object")


def test_an_unknown_field_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_b"], "colour": "red"}], "unknown field")


def test_a_group_without_an_id_or_members_is_rejected(room):
    _reject(room, [{"members": ["mic_a", "mic_b"]}], "needs both")
    _reject(room, [{"id": "room"}], "needs both")


def test_an_empty_group_id_is_rejected(room):
    _reject(room, [{"id": "   ", "members": ["mic_a", "mic_b"]}], "nonempty")


def test_a_group_id_longer_than_64_characters_is_rejected(room):
    _reject(room, [{"id": "g" * 65, "members": ["mic_a", "mic_b"]}], "limit is 64")


def test_more_than_32_groups_are_rejected(room):
    groups = [{"id": f"g{i}", "members": ["mic_a", "mic_b"]} for i in range(33)]
    _reject(room, groups, "at most 32")


def test_a_boolean_pan_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_b"], "pan": True}], "finite number")


def test_a_nan_pan_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_b"], "pan": float("nan")}], "finite number")


def test_a_textual_pan_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_b"], "pan": "-0.2"}], "finite number")


def test_a_pan_outside_the_range_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_b"], "pan": 1.5}], "within -1.0..1.0")


def test_two_groups_with_the_same_id_are_rejected(room):
    _reject(
        room,
        [
            {"id": "room", "members": ["mic_a", "mic_b"]},
            {"id": "room", "members": ["mic_c", "drums"]},
        ],
        "declared twice",
    )


def test_a_member_listed_twice_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "mic_a"]}], "twice")


def test_a_track_in_two_groups_is_rejected(room):
    _reject(
        room,
        [
            {"id": "one", "members": ["mic_a", "mic_b"]},
            {"id": "two", "members": ["mic_a", "mic_c"]},
        ],
        "only be in one group",
    )


def test_an_unknown_member_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "nope"]}], "not one of the loaded tracks")


def test_a_group_with_one_member_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a"]}], "at least 2 members")


def test_a_group_with_more_than_128_members_is_rejected(room):
    members = [f"mic_{i}" for i in range(129)]
    _reject(room, [{"id": "room", "members": members}], "limit is 128")


def test_a_group_of_two_roles_is_rejected(room):
    _reject(room, [{"id": "room", "members": ["mic_a", "drums"]}], "mixes roles")


def test_a_stereo_member_is_rejected(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    mono = 0.3 * np.sin(2 * np.pi * 220 * t)
    _write(str(d / "mic_a.wav"), _stereo(mono))
    wide = np.stack([mono, -mono * 0.5], axis=1)   # a real stereo pair
    _write(str(d / "mic_b.wav"), wide)
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    _reject(stems, [{"id": "room", "members": ["mic_a", "mic_b"]}], "stereo file")


def test_members_imported_at_different_sample_rates_are_rejected(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    mono = 0.3 * np.sin(2 * np.pi * 220 * t)
    _write(str(d / "mic_a.wav"), _stereo(mono), sr=48000)
    _write(str(d / "mic_b.wav"), _stereo(mono), sr=44100)
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    _reject(stems, [{"id": "room", "members": ["mic_a", "mic_b"]}], "imported sample rates")


def test_members_that_differ_by_one_imported_sample_are_rejected(tmp_path, monkeypatch):
    """Sample precision, not milliseconds.

    Two imports that differ by a single sample are not the same take, and
    rounding them onto each other would let a mismatched pair into a group
    that claims to hold their relation together.
    """
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * 44100)
    t = np.arange(n) / 44100
    x = 0.2 * np.sin(2 * np.pi * 173 * t)
    _write(str(d / "Tom1 Top.wav"), _stereo(x), sr=44100)
    _write(str(d / "Tom1 Bottom.wav"), _stereo(x * 0.6), sr=44100)
    stems = _load(str(d), {"Tom1 Top.wav": "toms", "Tom1 Bottom.wav": "toms"})
    # the imported lengths differ by exactly one sample
    stems[1].source_duration = stems[0].source_duration + 1 / stems[0].sr
    _reject_before_render(
        stems,
        [{"id": "g", "members": ["Tom1_Top", "Tom1_Bottom"]}],
        monkeypatch,
        "imported durations",
    )


def test_a_pan_too_large_to_be_a_float_is_refused(tmp_path, monkeypatch):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    x = 0.2 * np.sin(2 * np.pi * 173 * t)
    _write(str(d / "mic_a.wav"), _stereo(x))
    _write(str(d / "mic_b.wav"), _stereo(x * 0.6))
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    # an integer too large to convert to a float must be a clear group
    # validation error, not an OverflowError from the middle of the engine
    _reject_before_render(
        stems,
        [{"id": "g", "members": ["mic_a", "mic_b"], "pan": 10 ** 400}],
        monkeypatch,
        "finite number",
    )


def test_a_saved_group_with_an_unusable_pan_is_dropped_with_an_error():
    """A saved state we cannot vouch for is dropped, never repaired."""
    cleaned, error = srvmod._clean_capture_groups(
        [{"id": "g", "members": ["a", "b"], "pan": 10 ** 400}]
    )
    assert cleaned == []
    assert error
    cleaned, error = srvmod._clean_capture_groups(
        [{"id": "g", "members": ["a", "b"], "pan": 2.0}]
    )
    assert cleaned == []
    assert error
    # a legitimate draft and a legal pan survive untouched
    for pan in (0.0, -1.0, 1.0):
        cleaned, error = srvmod._clean_capture_groups(
            [{"id": "g", "members": ["a"], "pan": pan}]
        )
        assert cleaned == [{"id": "g", "members": ["a"], "pan": pan}]
        assert error is None


def test_members_with_different_durations_are_rejected(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    mono = 0.3 * np.sin(2 * np.pi * 220 * t)
    _write(str(d / "mic_a.wav"), _stereo(mono))
    _write(str(d / "mic_b.wav"), _stereo(mono[: int(0.5 * SR)]))
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    _reject(stems, [{"id": "room", "members": ["mic_a", "mic_b"]}], "imported durations")


def test_a_group_of_silence_is_rejected(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    zeros = np.zeros((int(1.0 * SR), 2))
    _write(str(d / "mic_a.wav"), zeros)
    _write(str(d / "mic_b.wav"), zeros)
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    _reject(stems, [{"id": "room", "members": ["mic_a", "mic_b"]}], "reads as silent")


def test_a_member_the_planner_would_filter_is_refused(tmp_path):
    """A member that is not in the mix cannot be protected.

    Quiet audio is not the same thing as digital silence: a take far below
    the level floor still has real samples, but the planner drops it, so a
    group that claims to hold it would be a claim about audio that is not
    there.
    """
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * 44100)
    t = np.arange(n) / 44100
    x = (0.15 * np.sin(2 * np.pi * 173 * t) + 0.05 * np.sin(2 * np.pi * 1319 * t)) * np.exp(-3 * (t % 0.4))
    _write(str(d / "Tom1 Top.wav"), _stereo(x), sr=44100)
    _write(str(d / "Tom1 Bottom.wav"), _stereo(x * 1e-3), sr=44100)
    stems = _load(str(d), {"Tom1 Top.wav": "toms", "Tom1 Bottom.wav": "toms"})
    quiet = next(s for s in stems if s.name == "Tom1 Bottom.wav")
    # it is quiet, not empty: the samples are real
    assert int(np.count_nonzero(quiet.audio)) > 0
    assert quiet.metrics["silent"] is True
    _reject(stems, [{"id": "g", "members": ["Tom1_Top", "Tom1_Bottom"]}], "reads as silent")


def test_a_digitally_silent_member_is_refused(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    _write(str(d / "mic_a.wav"), _stereo(0.3 * np.sin(2 * np.pi * 220 * t)))
    _write(str(d / "mic_b.wav"), _stereo(np.zeros(n)))
    stems = _load(str(d), {"mic_a.wav": "guitar_electric", "mic_b.wav": "guitar_electric"})
    _reject(stems, [{"id": "room", "members": ["mic_a", "mic_b"]}], "reads as silent")


# --------------------------------------------------------- both reference passes

def test_both_reference_passes_receive_the_groups(room, tmp_path, monkeypatch):
    from automix import reference

    n = int(2.0 * SR)
    t = np.arange(n) / SR
    ref_path = _write(str(tmp_path / "ref.wav"), _stereo(0.2 * np.sin(2 * np.pi * 200 * t)))
    ref = reference.analyse(ref_path)

    seen = []
    real = planner.build_plan

    def spy(*args, **kwargs):
        seen.append(kwargs.get("capture_groups"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline.planner, "build_plan", spy)
    plan, _, report = pipeline.mix(
        room,
        _preset(),
        reference=ref,
        blend=0.7,
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}],
    )
    assert report["reference_passes"] == 2
    assert len(seen) == 2
    assert all(c is not None and c[0]["id"] == "room" for c in seen)
    # the plan that came out of the second pass is still protected
    assert _track(plan, "mic_a")["gain_db"] == 0.0
    assert _track(plan, "mic_a")["pan"] == -0.3


# ------------------------------------------------------------------- the CLI

def _cli_stems(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    _room_files(d)
    return d


def test_the_cli_reads_a_capture_groups_file(tmp_path):
    d = _cli_stems(tmp_path)
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps([{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.2}]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert cli.main([
        "--stems", str(d), "--style", "pop_2020s", "--out", str(out),
        "--capture-groups", str(groups), "--quiet",
    ]) == 0
    plan = json.loads((out / "mixplan.json").read_text(encoding="utf-8"))
    assert plan["capture_groups"]["groups"][0]["id"] == "room"
    assert plan["capture_groups"]["groups"][0]["pan"] == -0.2
    member = next(t for t in plan["tracks"] if t["id"] == "mic_a")
    assert member["gain_db"] == 0.0
    assert member["pan"] == -0.2
    # the receipt says what was protected and where the protection stops
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Capture groups" in report
    assert "room" in report


def test_the_cli_refuses_capture_groups_with_render_plan(tmp_path):
    groups = tmp_path / "groups.json"
    groups.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main([
            "--render-plan", str(tmp_path / "plan.json"),
            "--capture-groups", str(groups),
        ])


def test_the_cli_replays_a_saved_plan_without_fitting_groups(tmp_path):
    d = _cli_stems(tmp_path)
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps([{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.2}]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert cli.main([
        "--stems", str(d), "--style", "pop_2020s", "--out", str(out),
        "--capture-groups", str(groups), "--plan-only", "--quiet",
    ]) == 0
    plan_path = out / "mixplan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "capture_groups" in plan

    # the replay uses the saved numbers; it does not fit a group into them
    again = tmp_path / "again"
    assert cli.main([
        "--render-plan", str(plan_path), "--out", str(again), "--quiet",
    ]) == 0
    replayed = json.loads((again / "mixplan.json").read_text(encoding="utf-8"))
    assert [t["gain_db"] for t in replayed["tracks"]] == [t["gain_db"] for t in plan["tracks"]]
    assert [t["pan"] for t in replayed["tracks"]] == [t["pan"] for t in plan["tracks"]]


# ----------------------------------------------------------------- the server

@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("r25-sessions"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srvmod.Handler)
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
    return _req(
        base,
        f"/api/session/{sid}/stem",
        data=data,
        method="POST",
        headers={"X-Filename": name, "Content-Type": "application/octet-stream"},
    )


def _post_json(base, path, payload):
    return _req(
        base,
        path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def test_the_declared_groups_are_stored_and_survive_a_restart(srv):
    base, root = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))

    code, body, _ = _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": [{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.2}]},
    )
    assert code == 200
    assert json.loads(body)["capture_groups"][0]["id"] == "room"

    code, body, _ = _req(base, f"/api/session/{sid}/state")
    state = json.loads(body)
    assert state["capture_groups"][0]["members"] == ["mic_a", "mic_b"]
    assert state["capture_groups"][0]["pan"] == -0.2

    with srvmod._lock:
        srvmod._sessions.pop(sid, None)
    srvmod._load_sessions(root)
    assert srvmod._sessions[sid].capture_groups[0]["id"] == "room"


def test_a_damaged_group_state_is_not_guessed(tmp_path):
    root = tmp_path / "sessions"
    sess = srvmod.Session(str(root), sid="s1")
    os.makedirs(os.path.join(sess.dir, "stems"), exist_ok=True)
    with open(os.path.join(sess.dir, "session.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": "s1", "capture_groups": [{"id": "room", "members": "not a list"}]}, fh)
    loaded = srvmod.Session.load(str(root), "s1")
    assert loaded is not None
    assert loaded.capture_groups == []
    assert loaded.capture_groups_error


def test_a_draft_with_one_member_is_stored_but_the_mix_refuses(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))

    # the draft is kept: the user is still making the group
    code, body, _ = _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": [{"id": "room", "members": ["mic_a"]}]},
    )
    assert code == 200
    assert json.loads(body)["capture_groups"][0]["members"] == ["mic_a"]

    # but a mix refuses it, and says why
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 400
    assert "at least 2 members" in json.loads(body)["error"]


def test_the_draft_endpoint_refuses_an_unusable_pan(srv):
    """A draft with a pan nobody can vouch for is a 400, not a 500."""
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))
    for pan in (10 ** 400, float("nan"), 2.0, -2.0):
        code, body, _ = _post_json(
            base,
            f"/api/session/{sid}/capture_groups",
            {"groups": [{"id": "room", "members": ["mic_a", "mic_b"], "pan": pan}]},
        )
        assert code == 400, (pan, body)
        assert "pan" in json.loads(body)["error"]


def test_a_group_is_protected_in_a_real_render_and_says_so(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))
    _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": [{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.2}]},
    )
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 200
    result = json.loads(body)
    receipt = result["capture_groups"]
    assert receipt["groups"][0]["id"] == "room"
    assert receipt["groups"][0]["pan"] == -0.2
    member = next(t for t in result["tracks"] if t["id"] == "mic_a")
    assert member["gain_db"] == 0.0
    assert member["pan"] == -0.2


def test_a_member_may_be_named_by_its_file_name(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))
    code, body, _ = _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": [{"id": "room", "members": ["mic_a.wav", "mic_b.wav"], "pan": 0.1}]},
    )
    assert code == 200
    stored = json.loads(body)["capture_groups"][0]
    # the session keeps the engine's own track id, whichever form was used
    assert stored["members"] == ["mic_a", "mic_b"]


def test_the_state_answers_with_the_engines_track_ids(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic a.wav", _wav_bytes(220.0))
    code, body, _ = _req(base, f"/api/session/{sid}/state")
    assert code == 200
    ids = json.loads(body)["track_ids"]
    assert ids["mic a.wav"] == "mic_a"


def test_an_excluded_member_stops_the_mix_and_keeps_the_previous_output(srv):
    base, _ = srv
    sid = _new_session(base)
    _upload(base, sid, "mic_a.wav", _wav_bytes(220.0))
    _upload(base, sid, "mic_b.wav", _wav_bytes(220.0))
    _post_json(
        base,
        f"/api/session/{sid}/capture_groups",
        {"groups": [{"id": "room", "members": ["mic_a", "mic_b"]}]},
    )
    code, _, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 200

    mix_path = os.path.join(srvmod._sessions[sid].dir, "mix.wav")
    with open(mix_path, "rb") as fh:
        before = fh.read()

    # excluding a member makes the group impossible to protect
    code, _, _ = _post_json(base, f"/api/session/{sid}/mute", {"track_id": "mic_a", "muted": True})
    assert code == 200
    code, body, _ = _post_json(base, f"/api/session/{sid}/run", {"style": "pop_2020s"})
    assert code == 400
    assert "excluded" in json.loads(body)["error"]

    # the previous successful output is still the one on disk
    with open(mix_path, "rb") as fh:
        assert fh.read() == before


def test_the_payload_function_survives_adversarial_group_names():
    """A group name is user text, so it may be "__proto__" or "constructor".

    The function is taken out of the page and run in node, because the bug
    this guards against only shows up against the browser's own objects.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    page = Path(__file__).resolve().parents[1] / "web" / "index.html"
    html = page.read_text(encoding="utf-8")
    function = html.split("function captureGroupsPayload()", 1)[1].split(
        "// A group that cannot", 1
    )[0]
    js = (
        "let groupByTrack=Object.create(null),groupPans=Object.create(null);\n"
        "function captureGroupsPayload()" + function + "\n"
        "const results=[];\n"
        "for (const name of ['normal','__proto__','constructor','toString']) {\n"
        " groupByTrack=Object.create(null);groupPans=Object.create(null);\n"
        " groupByTrack.a=name;groupByTrack.b=name;groupPans[name]=-.375;\n"
        " try {const payload=captureGroupsPayload();"
        "results.push({name,pass:payload.length===1&&payload[0].id===name&&"
        "payload[0].pan===-.375&&payload[0].members.length===2,payload});}\n"
        " catch(e){results.push({name,pass:false,error:e.message});}\n"
        "}\n"
        "console.log(JSON.stringify({results,pass:results.every(r=>r.pass)}));"
    )
    proc = subprocess.run([node, "-e", js], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["pass"], receipt
