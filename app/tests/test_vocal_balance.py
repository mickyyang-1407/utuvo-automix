"""The vocal ends up where a professional mix puts it (R109).

Reported 2026-11-02: across several songs the lead vocal came out much too
quiet, and the backing vocals too.  Measured on 11 MUSDB18 songs, the vocal
lost 3.5-4.5 dB between the planned balance and the finished buses: it is
compressed twice (track and bus) with half make-up, and a sustained vocal
loses more to that than drums do.  A raw session with many accompaniment
tracks buries a single vocal however its own gain is set.

So the plan carries a target for the vocal against everything else and the
renderer moves the whole vocal bus there after bus processing.  Everything
here uses synthetic signals built in the test.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import dsp, plan as planner, render

SR = 48000


def _tone(seconds, freq, level_db, start=0.0, sr=SR):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq * t) * dsp.db_to_lin(level_db)
    x[: int(start * sr)] = 0.0
    return np.stack([x, x], axis=1)


def test_the_ratio_is_measured_only_where_the_vocal_sings():
    rest = _tone(20, 220, -20)
    vocal = _tone(20, 1000, -20, start=10.0)  # silent for the first half
    value, windows = dsp.vocal_to_rest_db(vocal, rest, SR)
    assert windows > 0
    # same level where it sings (a K-weighted 1 kHz tone reads about the same
    # as 220 Hz); an intro without singing must not drag it down
    assert abs(value) < 1.0


def test_the_vocal_bus_is_moved_to_the_target_and_reported():
    buses = {"vocal": _tone(12, 1000, -30), "music": _tone(12, 300, -18), "drum": _tone(12, 120, -20)}
    before, _ = dsp.vocal_to_rest_db(buses["vocal"], buses["music"] + buses["drum"], SR)
    warnings = []
    spec = {"balance": {"law": "vocal_var_v1", "target_db": -1.6, "max_boost_db": 18.0, "max_cut_db": 6.0}}
    result = render._vocal_balance(buses, spec, SR, warnings)
    after, _ = dsp.vocal_to_rest_db(buses["vocal"], buses["music"] + buses["drum"], SR)
    assert result["measured_db"] == pytest.approx(before, abs=0.01)
    assert after == pytest.approx(-1.6, abs=0.05)
    assert result["applied_db"] == pytest.approx(-1.6 - before, abs=0.01)
    assert not warnings


def test_a_limited_correction_says_so():
    # well above the "is it singing" floor, far below the target
    buses = {"vocal": _tone(12, 1000, -40), "music": _tone(12, 300, -10)}
    warnings = []
    spec = {"balance": {"law": "vocal_var_v1", "target_db": 0.0, "max_boost_db": 9.0, "max_cut_db": 6.0}}
    result = render._vocal_balance(buses, spec, SR, warnings)
    assert result["applied_db"] == 9.0
    assert warnings and "limited to +9.0 dB" in warnings[0]


def test_a_plan_without_the_block_leaves_the_buses_alone():
    vocal = _tone(6, 1000, -30)
    buses = {"vocal": vocal.copy(), "music": _tone(6, 300, -18)}
    assert render._vocal_balance(buses, {}, SR, []) is None
    assert np.array_equal(buses["vocal"], vocal)


def test_an_unknown_law_is_refused():
    buses = {"vocal": _tone(6, 1000, -30), "music": _tone(6, 300, -18)}
    with pytest.raises(render.RenderError):
        render._vocal_balance(buses, {"balance": {"law": "nope", "target_db": 0}}, SR, [])


def test_the_target_is_the_professional_median_plus_the_styles_forwardness():
    # measured with the renderer's own meter on professional mixes
    assert planner.VOCAL_VAR_PRO_MEDIAN_DB == -1.9  # R114: 46 MedleyDB mixes (R109: 11, -2.7)
    preset = planner.load_preset("pop_2020s")
    want = round(-1.9 + float(preset["vocal_forward_db"]) + float(preset["techniques"]["vocal_lift_db"]), 2)
    assert want == pytest.approx(0.1)  # R114: the pop vocal was heard as a little small


@pytest.fixture(scope="module")
def demo_stems():
    import os
    from automix import analyze
    demo = os.path.join(os.path.dirname(__file__), "..", "demo_stems")
    stems = analyze.load_stems(demo)
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_a_plan_with_a_vocal_carries_the_style_target(demo_stems):
    plan = planner.build_plan(demo_stems, planner.load_preset("pop_2020s"))
    bal = plan["buses"]["vocal"]["balance"]
    assert bal["law"] == "vocal_var_v1"
    # an unproven session: the recording levels of the vocal do not count
    assert plan["style"]["source_balance"]["origin"] == "default"
    assert "source_db" not in bal
    assert bal["target_db"] == bal["style_db"] == pytest.approx(0.1)
    planner.validate(plan)


def test_vouched_for_stems_keep_their_own_vocal_balance(demo_stems):
    plan = planner.build_plan(demo_stems, planner.load_preset("pop_2020s"),
                              source_balance_profile="balanced_stems")
    bal = plan["buses"]["vocal"]["balance"]
    assert "source_db" in bal
    # balanced_stems keeps all of the source's balance
    assert bal["target_db"] == pytest.approx(max(-12.0, min(6.0, bal["source_db"])), abs=0.01)


def test_a_plan_without_a_vocal_has_no_vocal_target(demo_stems):
    stems = [s for s in demo_stems if s.role not in ("lead_vocal", "backing_vocal")]
    plan = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert "balance" not in plan["buses"]["vocal"]


def test_the_reverb_bus_counts_for_neither_side():
    # a loud fx return must not make the vocal look quiet: dry against dry
    base = {"vocal": _tone(12, 1000, -30), "music": _tone(12, 300, -18)}
    with_fx = {k: v.copy() for k, v in base.items()}
    with_fx["fx"] = _tone(12, 500, -10)
    spec = {"balance": {"law": "vocal_var_v1", "target_db": -1.6, "max_boost_db": 18.0, "max_cut_db": 6.0}}
    a = render._vocal_balance(base, spec, SR, [])
    b = render._vocal_balance(with_fx, spec, SR, [])
    assert a["measured_db"] == b["measured_db"]
    assert np.array_equal(base["vocal"], with_fx["vocal"])


def test_an_accepted_reference_fit_is_measured_with_the_fit_applied(demo_stems):
    vocal = next(s for s in demo_stems if s.role == "lead_vocal")
    raw = planner._source_vocal_balance(demo_stems)
    fitted = planner._source_vocal_balance(demo_stems, {analyze_track_id(vocal.name): 3.0})
    assert fitted == pytest.approx(raw + 3.0, abs=0.05)


def analyze_track_id(name):
    from automix import analyze
    return analyze.track_id(name)


def test_a_balance_target_on_another_bus_is_refused(demo_stems):
    plan = planner.build_plan(demo_stems, planner.load_preset("pop_2020s"))
    plan["buses"]["music"]["balance"] = dict(plan["buses"]["vocal"]["balance"])
    with pytest.raises(planner.PlanError):
        planner.validate(plan)
