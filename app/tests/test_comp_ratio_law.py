"""A compressor block's ratio means what it says -- for new plans only.

Plans written before ``ratio_law`` existed were rendered with the ratio
mapped twice (a written 2:1 compressed at 1.2:1).  Those plans must replay
exactly as before; new plans compress at the written ratio.
"""
from __future__ import annotations

import copy

import jsonschema
import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000


def _slope(spec: dict) -> float:
    """Input dB change per output dB change, well above threshold and knee."""
    t = np.arange(SR) / SR
    out = []
    for level_db in (-12.0, -4.0):
        amp = 10 ** (level_db / 20)
        x = np.stack([amp * np.sin(2 * np.pi * 1000 * t)] * 2, axis=1)
        y = render._apply_comp(x, SR, spec)
        mid = slice(SR // 4, 3 * SR // 4)  # steady state only
        out.append(20 * np.log10(np.sqrt(np.mean(y[mid] ** 2))))
    return 8.0 / (out[1] - out[0])


def _spec(ratio: float, law: str | None) -> dict:
    spec = {"threshold_db": -40.0, "ratio": ratio, "attack_ms": 5.0,
            "release_ms": 50.0, "makeup": {"mode": "none", "cap_db": 12.0}}
    if law is not None:
        spec["ratio_law"] = law
    return spec


@pytest.mark.parametrize("ratio", [1.5, 2.0, 4.0, 8.0])
def test_a_literal_ratio_is_the_measured_ratio(ratio):
    assert _slope(_spec(ratio, "literal_v1")) == pytest.approx(ratio, rel=0.02)


@pytest.mark.parametrize("ratio", [1.5, 2.0, 4.0])
def test_a_plan_without_the_field_keeps_the_old_reading(ratio):
    effective = 1 + (ratio - 1) ** 2 / 5
    assert _slope(_spec(ratio, None)) == pytest.approx(effective, rel=0.02)
    assert _slope(_spec(ratio, "legacy_amount_v1")) == pytest.approx(effective, rel=0.02)


def test_a_literal_unity_ratio_does_nothing():
    x = np.random.default_rng(1).normal(0, 0.3, (SR, 2))
    assert render._apply_comp(x, SR, _spec(1.0, "literal_v1")) is x


@pytest.fixture()
def stems(tmp_path):
    t = np.arange(2 * SR) / SR
    for name, f, amp in (("kick.wav", 60, 0.5), ("bass.wav", 55, 0.3),
                         ("vocal_lead.wav", 440, 0.25), ("pad.wav", 220, 0.15)):
        x = amp * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        pcm = (np.clip(np.stack([x, x], axis=1), -1, 1) * 32767).astype("<i2")
        import wave
        with wave.open(str(tmp_path / name), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(pcm.tobytes())
    stems = analyze.load_stems(str(tmp_path))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


def _comps(plan):
    return [t["comp"] for t in plan["tracks"] if t.get("comp")] + \
        [b["comp"] for b in plan["buses"].values() if b.get("comp")]


def test_new_plans_write_the_literal_law_and_old_plans_still_replay(stems):
    plan = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert _comps(plan) and all(c["ratio_law"] == "literal_v1" for c in _comps(plan))
    planner.validate(plan)

    old = copy.deepcopy(plan)
    for c in _comps(old):
        del c["ratio_law"]
    planner.validate(old)
    explicit = copy.deepcopy(old)
    for c in _comps(explicit):
        c["ratio_law"] = "legacy_amount_v1"

    new_mix, _ = render.render(stems, plan)
    old_mix, _ = render.render(stems, old)
    explicit_mix, _ = render.render(stems, explicit)
    assert np.array_equal(old_mix, explicit_mix)
    assert not np.array_equal(new_mix, old_mix)


def test_an_unknown_ratio_law_is_refused(stems, schema_free):
    plan = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _comps(plan)[0]["ratio_law"] = "louder_v9"
    with pytest.raises(jsonschema.ValidationError):
        planner.validate(plan)
    with pytest.raises(planner.PlanError, match="ratio_law"):
        planner.validate(plan, schema_free)
