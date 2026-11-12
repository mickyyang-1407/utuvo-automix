"""R126: the drum mics' hits put back, level-independently.

A transient shaper in the manner SPL published for the Transient Designer:
two followers of one level, their difference drives the gain, no threshold.
It must give a soft hit and a loud hit the same dB, leave sustained sound and
bleed alone, land its ramp before the hit, and stay a plan field
(tracks[].transient): plans without it are untouched (test_replay_r11*.py).
"""
from __future__ import annotations

import copy
import glob
import os

import jsonschema
import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 48000
SPEC = {"law": "transient_diff_env_v1", "attack_db": 4.0, "sustain_db": -2.0, **planner.TRANSIENT}
HIT = SR // 10          # the first hit of _hits() starts here


def _snare(n=SR // 2, seed=0):
    t = np.arange(n) / SR
    return np.random.default_rng(seed).standard_normal(n) * np.exp(-t / 0.06)


def _kick(n=SR // 2):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.12)


def _hits(one, count=4, level=0.5):
    return level * np.concatenate([np.zeros(HIT)] + [one] * count)


def _st(v):
    return np.stack([v, v], 1)


def _db(a):
    return 10 * np.log10(np.mean(np.square(a)) + 1e-30)


@pytest.mark.parametrize("one", [_snare(), _kick()], ids=["snare", "kick"])
def test_a_soft_hit_and_a_loud_hit_get_the_same_shaping(one):
    loud = dsp.transient_gain_db(_hits(one), SR, SPEC)
    soft = dsp.transient_gain_db(_hits(one, level=0.5 * 10 ** (-30 / 20)), SR, SPEC)
    assert np.max(np.abs(loud - soft)) < 1e-6
    assert loud.max() == pytest.approx(4.0, abs=0.05)
    assert loud.min() == pytest.approx(-2.0, abs=0.05)


@pytest.mark.parametrize("freq", [40.0, 55.0, 100.0, 440.0, 3000.0])
def test_a_held_note_is_given_back_untouched(freq):
    t = np.arange(SR * 2) / SR
    x = _st(0.4 * np.sin(2 * np.pi * freq * t))
    # past the note's own start, which is a rise like any other
    g = dsp.transient_gain_db(x, SR, SPEC)[SR // 2:]
    assert np.max(np.abs(g)) < 0.05


def test_a_swelling_pad_is_not_a_hit():
    t = np.arange(SR * 3) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    assert np.max(np.abs(dsp.transient_gain_db(x, SR, SPEC)[SR // 2:])) < 0.05


def test_the_hit_stands_further_above_its_body_and_the_tail_is_shorter():
    x = _st(_hits(_snare()))
    y, rec = dsp.transient_shape(x, SR, SPEC)
    head = slice(HIT, HIT + SR // 100)                       # the first 10 ms
    body = slice(HIT + int(0.03 * SR), HIT + int(0.10 * SR))
    tail = slice(HIT + int(0.15 * SR), HIT + int(0.30 * SR))
    assert _db(y[head]) - _db(x[head]) == pytest.approx(4.0, abs=0.3)
    assert abs(_db(y[body]) - _db(x[body])) < 1.5
    assert _db(y[tail]) - _db(x[tail]) < -1.0
    assert rec == {"max_gain_db": pytest.approx(4.0, abs=0.05), "min_gain_db": pytest.approx(-2.0, abs=0.05)}


def test_attack_alone_never_cuts_and_sustain_alone_never_lifts():
    x = _hits(_snare())
    up = dsp.transient_gain_db(x, SR, {**SPEC, "sustain_db": 0.0})
    down = dsp.transient_gain_db(x, SR, {**SPEC, "attack_db": 0.0})
    assert up.min() >= -1e-9 and up.max() > 3.9
    assert down.max() <= 1e-9 and down.min() < -1.9


def test_the_ramp_is_up_before_the_hit_and_gone_once_the_hit_has_passed():
    g = dsp.transient_gain_db(_hits(_kick()), SR, {**SPEC, "sustain_db": 0.0})
    assert g[HIT - SR // 400] == 0.0                         # 2.5 ms early: nothing yet
    assert g[HIT] > 3.5                                      # at the hit: there already
    assert np.max(np.abs(g[HIT + int(0.020 * SR):HIT + int(0.4 * SR)])) < 0.05   # 20 ms on: over


def test_bleed_far_under_the_tracks_own_hits_is_left_alone():
    loud, quiet = _hits(_snare(), count=2), _hits(_snare(seed=1), count=2, level=0.5 * 10 ** (-45 / 20))
    x = np.concatenate([loud, quiet])
    g = dsp.transient_gain_db(x, SR, SPEC)
    assert np.max(np.abs(g[loud.size + HIT // 2:])) == 0.0
    assert g[:loud.size].max() > 3.9
    near = np.concatenate([loud, _hits(_snare(seed=1), count=2, level=0.5 * 10 ** (-20 / 20))])
    assert dsp.transient_gain_db(near, SR, SPEC)[loud.size + HIT // 2:].max() > 3.9


def test_silence_and_zero_amounts_give_back_the_very_same_array():
    z = np.zeros((SR, 2), dtype=np.float32)
    assert dsp.transient_shape(z, SR, SPEC)[0] is z
    x = _st(_hits(_snare())).astype(np.float32)
    y, rec = dsp.transient_shape(x, SR, {**SPEC, "attack_db": 0.0, "sustain_db": 0.0})
    assert y is x and rec == {"max_gain_db": 0.0, "min_gain_db": 0.0}


def test_mono_and_stereo_get_one_linked_gain():
    v = _hits(_snare())
    x = np.stack([v, 0.25 * v], 1)
    y, _ = dsp.transient_shape(x, SR, SPEC)
    m, _ = dsp.transient_shape(v, SR, SPEC)
    assert np.allclose(y[:, 0], m) and np.allclose(y[:, 1], 0.25 * m)


@pytest.mark.parametrize("side", [0, 1])
def test_a_hit_on_either_channel_moves_both_by_the_same_gain(side):
    # the hits on one side only, a quiet held tone on the other: the image must not lurch
    t = np.arange(_hits(_snare()).size) / SR
    x = np.zeros((t.size, 2))
    x[:, side] = _hits(_snare())
    x[:, 1 - side] = 0.01 * np.sin(2 * np.pi * 440 * t)
    g = dsp.transient_gain_db(x, SR, SPEC)
    assert g[HIT] > 3.5
    y, _ = dsp.transient_shape(x, SR, SPEC)
    lin = 10 ** (g / 20)
    assert np.allclose(y[:, 0], x[:, 0] * lin) and np.allclose(y[:, 1], x[:, 1] * lin)


def test_a_wrong_law_is_refused():
    with pytest.raises(ValueError):
        dsp.transient_gain_db(_hits(_snare()), SR, {**SPEC, "law": "nope"})


# ------------------------------------------------------------- plan + render

def _session(seconds=6, second_kick=False):
    n = SR * seconds
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)

    def loop(one, every):
        out = np.zeros(n)
        for start in range(HIT, n - one.size, int(every * SR)):
            out[start:start + one.size] += one
        return out

    parts = {"Kick.wav": 0.5 * loop(_kick(SR // 4), 0.5),
             "Snare.wav": 0.4 * loop(_snare(SR // 4), 1.0),
             "Hi Hat.wav": 0.1 * loop(rng.standard_normal(SR // 20) * np.exp(-np.arange(SR // 20) / (0.01 * SR)), 0.25),
             "Bass.wav": 0.3 * np.sin(2 * np.pi * 82 * t),
             "Lead Vocal.wav": 0.2 * np.sin(2 * np.pi * 330 * t) * ((t % 2.0) < 1.5)}
    if second_kick:
        parts["Kick Out.wav"] = 0.3 * loop(_kick(SR // 4), 0.5)
    stems = [analyze.Stem(name=name, path=name, sr=SR, audio=_st(a).astype(np.float32)) for name, a in parts.items()]
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def _ts(p):
    return {t["id"]: t["transient"] for t in p["tracks"] if "transient" in t}


def test_new_plans_shape_the_close_drums_by_the_style(stems):
    kick, snare = analyze.track_id("Kick.wav"), analyze.track_id("Snare.wav")
    p = planner.build_plan(stems, planner.load_preset("rock"))
    planner.validate(p)
    got = _ts(p)
    assert sorted(got) == sorted([kick, snare])              # not the hats, the bass or the voice
    assert got[kick] == {"law": "transient_diff_env_v1", "attack_db": 3.0, "sustain_db": 0.0, **planner.TRANSIENT}
    assert got[snare] == {"law": "transient_diff_env_v1", "attack_db": 4.0, "sustain_db": -1.0, **planner.TRANSIENT}
    assert any("transient shaping" in e for t in p["tracks"] if t["id"] == snare for e in t["evidence"])
    metal = _ts(planner.build_plan(stems, planner.load_preset("heavy_rock_metal")))
    assert metal[snare]["attack_db"] == 5.0 and metal[snare]["sustain_db"] == -1.25
    half = _ts(planner.build_plan(stems, planner.load_preset("rock"), intensity=0.5))
    assert half[snare]["attack_db"] == 2.0 and half[snare]["sustain_db"] == -0.5 and half[kick]["sustain_db"] == 0.0


@pytest.mark.parametrize("style", ["jazz_natural", "ambient_space"])
def test_styles_that_ask_for_none_get_none(stems, style):
    assert _ts(planner.build_plan(stems, planner.load_preset(style))) == {}


def test_every_shipped_style_states_its_amount():
    names = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(planner.__file__))), "presets", "*.json"))]
    assert len(names) >= 14
    for name in names:
        amount = planner.load_preset(name)["techniques"]["drum_transient"]
        assert 0.0 <= amount <= 1.5, name


def test_the_amount_is_capped(stems, monkeypatch):
    preset = copy.deepcopy(planner.load_preset("rock"))
    preset["techniques"]["drum_transient"] = 9.0
    got = _ts(planner.build_plan(stems, preset))
    assert all(v["attack_db"] <= 6.0 and v["sustain_db"] >= -6.0 for v in got.values())
    planner.validate(planner.build_plan(stems, preset))


def test_the_render_shapes_after_the_saturation_and_says_so(stems, tmp_path):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    mix, rep = render.render(stems, p)
    snare = analyze.track_id("Snare.wav")
    assert sorted(r["track"] for r in rep["transient"]) == sorted(_ts(p))
    row = next(r for r in rep["transient"] if r["track"] == snare)
    assert row["attack_db"] == 4.0 and row["max_gain_db"] > 3.5 and row["min_gain_db"] < -0.5
    steps = [c["step"] for c in next(t for t in rep["tracks"] if t["id"] == snare)["chain"]]
    assert steps.index("saturation") < steps.index("transient shaping") < steps.index("width") \
        if "saturation" in steps else steps.index("compressor") < steps.index("transient shaping") < steps.index("width")
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    assert "drum hits sharpened" in path.read_text(encoding="utf-8")

    off = copy.deepcopy(p)
    for t in off["tracks"]:
        t.pop("transient", None)
    mix0, rep0 = render.render(stems, off)
    assert "transient" not in rep0
    assert not np.array_equal(np.asarray(mix), np.asarray(mix0))


@pytest.mark.parametrize("bad", [{"law": "nope"}, {"attack_db": 7.0}, {"sustain_db": -7.0}, {"attack_ms": 1.0},
                                 {"sustain_ms": 50.0}, {"range_db": 5.0}, {"attack_db": float("nan")},
                                 {"attack_db": True}])
def test_a_wrong_transient_block_is_refused(stems, bad, schema_free):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    tid = next(iter(_ts(p)))
    next(t for t in p["tracks"] if t["id"] == tid)["transient"].update(bad)
    with pytest.raises(render.RenderError):
        render.render(stems, p)


@pytest.mark.parametrize("bad", [{"law": "nope"}, {"attack_db": 7.0}, {"range_db": 5.0}, {"extra": 1}])
def test_the_schema_refuses_it_too(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    next(t for t in p["tracks"] if "transient" in t)["transient"].update(bad)
    with pytest.raises(jsonschema.ValidationError):
        planner.validate(p)


# ------------------------------------------------------------ the peak budget

def _budget(stems, limiter_readings, plan=None):
    """Run pipeline._with_transient_budget over a mix whose limiter reads the given p95 values, run by run."""
    plan = plan or planner.build_plan(stems, planner.load_preset("rock"))
    seen = []

    def run(candidate):
        seen.append(candidate)
        # the rest of the mix may hand back its own edited copy of the plan (a headroom trim, a ride)
        return dict(candidate, ran=len(seen)), "audio%d" % len(seen), {"limiter": {"gain_reduction_p95_db": limiter_readings[len(seen) - 1]}}

    out, audio, report = pipeline._with_transient_budget(plan, run)
    return plan, out, audio, report, seen


def test_a_limiter_inside_the_guard_keeps_the_shaping_and_runs_once(stems):
    plan, out, audio, report, seen = _budget(stems, [-2.4])
    assert seen == [plan] and audio == "audio1" and _ts(out) == _ts(plan)


def test_past_the_guard_the_shaping_is_halved_and_that_is_kept_if_it_is_enough(stems):
    plan, out, audio, report, seen = _budget(stems, [-2.9, -2.3])
    assert len(seen) == 2 and audio == "audio2" and out["ran"] == 2
    assert report["limiter"]["gain_reduction_p95_db"] == -2.3
    before, after = _ts(plan), _ts(out)
    assert sorted(after) == sorted(before) and before          # the input plan is not edited in place
    for tid in before:
        assert after[tid]["attack_db"] == round(before[tid]["attack_db"] * 0.5, 2)
        assert after[tid]["sustain_db"] == round(before[tid]["sustain_db"] * 0.5, 2)
    assert all(any("scaled to attack" in e for e in t["evidence"]) for t in out["tracks"] if "transient" in t)
    planner.validate(seen[1])


def test_still_past_the_guard_the_shaping_is_left_out(stems):
    plan, out, audio, report, seen = _budget(stems, [-2.9, -2.7, -3.3])
    assert len(seen) == 3 and audio == "audio3"
    assert _ts(out) == {} and _ts(plan)
    # each retry starts from the plan as first built, not from the previous retry
    assert [max(v["attack_db"] for v in _ts(c).values()) if _ts(c) else 0.0 for c in seen] == [4.0, 2.0, 0.0]
    assert sum(any("transient shaping left out" in e for e in t["evidence"]) for t in out["tracks"]) == len(_ts(plan))
    assert not any("scaled to attack" in e for t in out["tracks"] for e in t["evidence"])
    planner.validate(seen[2])


def test_the_budget_sits_half_a_db_inside_the_squash_guard():
    assert pipeline.TRANSIENT_BUDGET_GR_DB == -2.5 and pipeline.TRANSIENT_BUDGET_HALF_REACH_DB == 0.5


def test_far_past_the_guard_the_half_step_is_not_tried(stems):
    plan, out, audio, report, seen = _budget(stems, [-3.1, -3.4])
    assert len(seen) == 2 and audio == "audio2"
    assert _ts(seen[1]) == {} and _ts(out) == {}


def test_a_plan_without_shaping_is_run_once_whatever_the_limiter_says(stems):
    plan = planner.build_plan(stems, planner.load_preset("jazz_natural"))
    _, out, audio, _, seen = _budget(stems, [-6.0], plan)
    assert seen == [plan] and audio == "audio1"


def test_the_whole_mix_ends_inside_the_guard_or_without_shaping(stems):
    plan, audio, report = pipeline.mix(stems, planner.load_preset("heavy_rock_metal"))
    gr = report["limiter"]["gain_reduction_p95_db"]
    assert gr >= pipeline.TRANSIENT_BUDGET_GR_DB or _ts(plan) == {}
    assert ("transient" in report) == bool(_ts(plan))


def test_a_declared_capture_group_is_left_as_imported():
    stems = _session(second_kick=True)
    kick, out, snare = (analyze.track_id(n) for n in ("Kick.wav", "Kick Out.wav", "Snare.wav"))
    free = planner.build_plan(stems, planner.load_preset("rock"))
    assert sorted(_ts(free)) == sorted([kick, out, snare])
    p = planner.build_plan(stems, planner.load_preset("rock"), capture_groups=[{"id": "kick mics", "members": [kick, out]}])
    planner.validate(p)
    assert sorted(_ts(p)) == [snare]


def test_a_mix_matched_to_a_reference_keeps_the_budget_on_both_passes(stems, tmp_path, monkeypatch):
    from automix import reference as refmod
    ref_path = tmp_path / "ref.wav"
    total = sum(np.asarray(st.audio, dtype=np.float64) for st in stems)
    render.write_wav24(str(ref_path), 0.5 * total / np.max(np.abs(total)), SR)
    ref = refmod.analyse(str(ref_path))
    real, shaped = pipeline._with_transient_budget, []

    def watched(plan, run):
        shaped.append(bool(_ts(plan)))
        return real(plan, run)

    monkeypatch.setattr(pipeline, "_with_transient_budget", watched)
    p, _, report = pipeline.mix(stems, planner.load_preset("rock"), reference=ref)
    assert shaped == [True, True] and report["reference_passes"] == 2
    assert ("transient" in report) == bool(_ts(p))
