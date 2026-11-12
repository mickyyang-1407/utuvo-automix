"""A bus compressor's estimated threshold must stay inside the schema's range.

``plan.build_plan`` writes a bus compressor's ``threshold_db`` as an *estimate*:
the summed level of the bus's own tracks plus an offset.  The estimate can land
outside the range the shipped plan schema allows -- ``compressor.threshold_db``
is [-60, 0] -- and then the saved plan is rejected even though the renderer
never used that number: the renderer places the threshold from
``threshold_offset_db`` whenever it exists, which is exactly the case for an
estimated bus threshold.

So the estimate is bounded to the schema's range and nothing else moves.  These
tests pin that: the offset, the ratio, the timing and the rendered audio are
unchanged, an estimate that was already in range is left exactly as it was, and
the bound is the shipped schema's own range, read from the schema file rather
than hard-coded here.

All fixtures are synthetic: short sine tones, no recorded material.
"""

from __future__ import annotations

import copy
import json
import os
from unittest import mock

import numpy as np

from automix import analyze, plan as planner, render

SR = 24000
SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schemas", "mix_plan.schema.json"
)
BUS_GROUPS = ("drum", "bass", "music", "vocal", "fx")


# ------------------------------------------------------------------ the schema

def _schema_threshold_range() -> tuple[float, float]:
    """The legal compressor threshold range, read from the shipped schema."""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)
    spec = schema["$defs"]["compressor"]["properties"]["threshold_db"]
    low, high = float(spec["minimum"]), float(spec["maximum"])
    # every bus group points at the same compressor definition
    for name, body in schema["properties"]["buses"]["properties"].items():
        comp = (body.get("properties") or {}).get("comp") or {}
        ref = comp.get("$ref")
        if ref:
            assert ref == "#/$defs/compressor", name
        elif "properties" in comp:
            other = comp["properties"]["threshold_db"]
            assert (float(other["minimum"]), float(other["maximum"])) == (low, high), name
    return low, high


# ------------------------------------------------------------------- fixtures

def _stem(name: str, audio: np.ndarray, role: str):
    st = analyze.Stem(
        name + ".wav",
        name + ".wav",
        np.column_stack([audio, audio]),
        SR,
        role=role,
        role_source="user",
    )
    analyze.measure(st)
    return st


def _guitar(scale: float = 1.0) -> np.ndarray:
    t = np.arange(8 * SR) / SR
    return scale * (0.2 * np.sin(2 * np.pi * 110 * t) + 0.08 * np.sin(2 * np.pi * 3200 * t))


def _vocal() -> np.ndarray:
    t = np.arange(8 * SR) / SR
    return (0.15 * np.sin(2 * np.pi * 660 * t) + 0.06 * np.sin(2 * np.pi * 2640 * t)) * (
        (t >= 2) & (t < 4)
    )


def _build(stems, **kw):
    opts = dict(
        source_balance_profile="raw_tracks",
        balance_preserve=1.0,
        reverb_amount=0.0,
        target_lufs=-20.0,
    )
    opts.update(kw)
    return planner.build_plan(stems, planner.load_preset("pop_2020s"), **opts)


def _unbounded(stems, **kw):
    """The same plan built with the bound widened out of the way.

    This is the behaviour before the fix -- an estimate of any size -- and it
    is what the bounded plan is compared against.
    """
    with mock.patch.object(planner, "COMP_THRESHOLD_DB_RANGE", (-1e9, 1e9)):
        return _build(stems, **kw)


def _capture_group():
    return [{"id": "guitar-capture", "members": ["music", "mic_b"], "pan": 0.375}]


def _comp(plan: dict, group: str) -> dict | None:
    return (plan["buses"].get(group) or {}).get("comp")


def _bus_level_db(stems, plan: dict, group: str) -> float | None:
    """The bus level estimate, recomputed from the plan and the stems."""
    by_id = {analyze.track_id(s.name): s for s in stems}
    power = 0.0
    for t in plan["tracks"]:
        if planner.ROLE_GROUP.get(t["role"], "music") != group:
            continue
        stem = by_id.get(t["id"])
        if stem is None:
            continue
        lufs = float(stem.metrics.get("lufs", -70.0))
        if lufs <= -70.0:
            continue
        power += 10.0 ** ((lufs + float(t.get("gain_db", 0.0))) / 10.0)
    if power <= 0.0:
        return None
    return 10.0 * float(np.log10(power))


# ------------------------------------------------------- the upper bound case

def _loud_protected_pair():
    """Two protected guitar mics trimmed up by a legal +18 dB.

    Their summed level estimates above 0 dB, so the bus threshold estimate was
    +3.5 dB -- outside the schema's range -- while the renderer used the
    offset and never that number.
    """
    stems = [
        _stem("music", _guitar(), "guitar_electric"),
        _stem("mic_b", _guitar(0.7), "guitar_electric"),
        _stem("lead", _vocal(), "lead_vocal"),
    ]
    return stems, dict(
        capture_groups=_capture_group(),
        trims={"music": 18.0, "mic_b": 18.0},
        pans={"music": -1.0, "mic_b": 1.0},
    )


def test_upper_out_of_range_estimate_is_bounded_to_the_schema_max():
    low, high = _schema_threshold_range()
    stems, kw = _loud_protected_pair()

    unbounded = _unbounded(stems, **kw)
    bounded = _build(stems, **kw)

    assert _comp(unbounded, "music")["threshold_db"] > high, "the fixture must reproduce the defect"
    assert _comp(bounded, "music")["threshold_db"] == high


def test_lower_out_of_range_estimate_is_bounded_to_the_schema_min():
    low, high = _schema_threshold_range()
    # A quiet protected pair pulled down 10 dB: both stems stay well above the
    # -60 dBFS active floor (so neither is rejected as silent), but the summed
    # bus level estimates below the schema's minimum.
    stems = [
        _stem("music", _guitar(0.01), "guitar_electric"),
        _stem("mic_b", _guitar(0.007), "guitar_electric"),
        _stem("lead", _vocal(), "lead_vocal"),
    ]
    kw = dict(
        capture_groups=_capture_group(),
        trims={"music": -10.0, "mic_b": -10.0},
        pans={"music": -1.0, "mic_b": 1.0},
    )

    unbounded = _unbounded(stems, **kw)
    bounded = _build(stems, **kw)

    assert _comp(unbounded, "music")["threshold_db"] < low, "the fixture must reproduce the defect"
    assert _comp(bounded, "music")["threshold_db"] == low


# ------------------------------------------------------- nothing else changed

def test_the_bound_moves_the_estimate_and_nothing_else():
    low, high = _schema_threshold_range()
    stems, kw = _loud_protected_pair()

    unbounded = _unbounded(stems, **kw)
    bounded = _build(stems, **kw)

    for group in BUS_GROUPS:
        before = _comp(unbounded, group)
        after = _comp(bounded, group)
        assert (before is None) == (after is None), group
        if before is None:
            continue
        for key in before:
            if key == "threshold_db":
                continue
            assert before[key] == after[key], f"{group}.{key} moved"
        # the estimate itself is untouched; only its bound changed
        level = _bus_level_db(stems, unbounded, group)
        if level is None:
            assert before["threshold_db"] == after["threshold_db"], group
        else:
            estimate = round(level + float(before["threshold_offset_db"]), 1)
            assert before["threshold_db"] == estimate, group
            assert after["threshold_db"] == min(high, max(low, estimate)), group


def test_the_offset_is_exact():
    """The offset is what the renderer uses; it must not move by a rounding."""
    stems, kw = _loud_protected_pair()
    preset = planner.load_preset("pop_2020s")
    bounded = _build(stems, **kw)
    unbounded = _unbounded(stems, **kw)

    for group in BUS_GROUPS:
        after = _comp(bounded, group)
        before = _comp(unbounded, group)
        if after is None:
            continue
        if "threshold_offset_db" not in before:
            # no level estimate for this bus, so there is no offset at all
            assert "threshold_offset_db" not in after, group
            continue
        assert after["threshold_offset_db"] == before["threshold_offset_db"], group
        amount = float(preset["compression"].get(group, 0.0)) * 0.75
        # R117 / R124: the drum and vocal buses sit at a fixed offset above the average
        fixed = {"drum": planner.DRUM_BUS_THRESHOLD_OFFSET_DB, "vocal": planner.VOCAL_BUS_THRESHOLD_OFFSET_DB}
        expected = fixed.get(group, round(-1.0 - 6.0 * amount, 1))
        assert after["threshold_offset_db"] == expected, group


def test_an_in_range_estimate_is_left_exact():
    """A bus estimate that was already legal must come out byte-identical."""
    stems = [
        _stem("kick", 0.5 * np.sin(2 * np.pi * 60 * np.arange(8 * SR) / SR), "kick"),
        _stem("bass", 0.3 * np.sin(2 * np.pi * 55 * np.arange(8 * SR) / SR), "bass"),
        _stem("music", _guitar(), "guitar_electric"),
        _stem("lead", _vocal(), "lead_vocal"),
    ]
    bounded = _build(stems)
    unbounded = _unbounded(stems)

    assert bounded["buses"] == unbounded["buses"]
    seen = 0
    low, high = _schema_threshold_range()
    for group in BUS_GROUPS:
        comp = _comp(bounded, group)
        if comp is None:
            continue
        seen += 1
        level = _bus_level_db(stems, bounded, group)
        if level is not None:
            estimate = round(level + float(comp["threshold_offset_db"]), 1)
            assert comp["threshold_db"] == estimate, group
        assert low <= comp["threshold_db"] <= high, group
    assert seen >= 4, "drum, bass, music and vocal buses must all be covered"


def test_every_bus_group_with_a_compressor_stays_inside_the_schema_range():
    low, high = _schema_threshold_range()
    stems = [
        _stem("kick", 0.5 * np.sin(2 * np.pi * 60 * np.arange(8 * SR) / SR), "kick"),
        _stem("bass", 0.3 * np.sin(2 * np.pi * 55 * np.arange(8 * SR) / SR), "bass"),
        _stem("music", _guitar(), "guitar_electric"),
        _stem("lead", _vocal(), "lead_vocal"),
        _stem("fx", _guitar(0.5), "fx"),
    ]
    plan = _build(stems)
    for group in BUS_GROUPS:
        comp = _comp(plan, group)
        if comp is None:
            # no style asks for a compressor on this bus, so there is no
            # threshold to bound
            continue
        assert low <= comp["threshold_db"] <= high, group


def test_the_plan_validates():
    low, high = _schema_threshold_range()
    stems, kw = _loud_protected_pair()
    plan = _build(stems, **kw)
    planner.validate(plan)
    for group in BUS_GROUPS:
        comp = _comp(plan, group)
        if comp is not None:
            assert low <= comp["threshold_db"] <= high, group
    try:
        import jsonschema
    except ImportError:
        return
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        jsonschema.validate(plan, json.load(fh))


def test_no_capture_groups_is_compatible():
    low, high = _schema_threshold_range()
    stems = [
        _stem("music", _guitar(), "guitar_electric"),
        _stem("lead", _vocal(), "lead_vocal"),
    ]
    for groups in (None, []):
        plan = _build(stems, capture_groups=groups)
        planner.validate(plan)
        for group in BUS_GROUPS:
            comp = _comp(plan, group)
            if comp is not None:
                assert low <= comp["threshold_db"] <= high, group


# ------------------------------------------------------------- the audio does not move

def test_the_rendered_audio_is_identical_for_the_old_and_the_bounded_plan():
    """The renderer uses the offset, so bounding the estimate changes no audio.

    The old plan is *not* schema-valid -- that is the defect -- so the schema
    validator is deliberately not run on it; it is only rendered.
    """
    low, high = _schema_threshold_range()
    stems, kw = _loud_protected_pair()

    bounded = _build(stems, **kw)
    old = copy.deepcopy(bounded)
    old["buses"]["music"]["comp"]["threshold_db"] = _unbounded(stems, **kw)["buses"]["music"][
        "comp"
    ]["threshold_db"]
    assert old["buses"]["music"]["comp"]["threshold_db"] > high

    old_audio, _ = render.render(stems, old, {})
    new_audio, _ = render.render(stems, bounded, {})

    assert np.all(np.isfinite(old_audio))
    assert np.array_equal(old_audio, new_audio)
