"""R130: a lead vocal's own compressor follows how much range the voice has.

The style's compression dial used to squash every lead vocal the same way.  A
voice that arrived already even (compressed on the way in, or a steady singer)
lost several dB of envelope range at its own compressor before the vocal bus,
the thickness chain and the limiter even started.  New plans simulate the
track's level path and turn the dial down until that stage takes no more than
the range above 17 dB (what finished professional lead vocals measure); a
voice with more range keeps the style's setting.  Receipt in comp_style_dial
(law vocal_range_fit_v1); the renderer does not read it, so older plans render
as they were.

The numbers below are written out on purpose (17 dB, 0.25 dB): a test that
reads the planner's constant passes with the constant broken.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000
TARGET = 17.0
TOL = 0.25


def _voice(seconds: float, spread_db: float, seed: int, level: float = 0.25) -> np.ndarray:
    """Syllables of 380 ms with 20 ms gaps; each syllable at its own level, spread
    uniformly over ``spread_db``; the first half sung 4 dB under the second (a
    ride's job), with a lot of 220 Hz body (the planner's corrective bell)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.25 * np.sin(2 * np.pi * 660 * t) + 0.12 * np.sin(2 * np.pi * 1320 * t))
    tone *= 1 + 0.15 * rng.standard_normal(n)
    # stand rumble under the whole take: the high-pass is part of the level path
    tone += 1.2 * np.sin(2 * np.pi * 38 * t)
    env = np.zeros(n)
    syl, gap = int(0.38 * SR), int(0.02 * SR)
    for s in range(0, n - syl, syl + gap):
        db = -rng.uniform(0.0, spread_db)
        if s < n // 2:
            db -= 4.0
        ramp = np.minimum(1.0, np.minimum(np.arange(syl), np.arange(syl)[::-1]) / (0.01 * SR))
        env[s:s + syl] = 10 ** (db / 20) * ramp
    return level * env * tone


def _band(spread_db: float, seed: int = 3, vocal_gain: float = 1.0, backing: bool = False, seconds: float = 24.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    kick = np.zeros(n)
    for s in range(0, n - 9600, SR // 2):
        tt = t[:9600]
        kick[s:s + 9600] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt / .08)
    parts = [("Kick.wav", .5 * kick), ("Bass.wav", .1 * np.sin(2 * np.pi * 55 * t)),
             ("Keys.wav", .05 * np.sin(2 * np.pi * 330 * t)),
             ("Lead Vocal.wav", vocal_gain * _voice(seconds, spread_db, seed))]
    if backing:
        parts.append(("Backing Vocal.wav", 0.6 * _voice(seconds, 10.0, seed + 7)))
    stems = [analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32))
             for name, a in parts]
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def _lead(p):
    (t,) = [t for t in p["tracks"] if t["role"] == "lead_vocal"]
    return t


@pytest.fixture(scope="module")
def even():
    # delivered hot (+9.5 dB): with the style's fixed dial this voice lost ~4 dB of range
    stems = _band(10.0, vocal_gain=3.0, backing=True)
    return stems, planner.build_plan(stems, planner.load_preset("pop_2020s"))


@pytest.fixture(scope="module")
def dynamic():
    stems = _band(34.0, seed=5, vocal_gain=3.0)
    return stems, planner.build_plan(stems, planner.load_preset("pop_2020s"))


def _ruler(x, sung=None):
    """An independent reading of the ruler: 100 ms (4800 samples at 48 kHz) mean
    power over both channels, p95 - p10 over the windows within 30 dB of the loudest."""
    m = np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=1)
    n = m.shape[0] // 4800
    e = 10 * np.log10(np.mean(m[: n * 4800].reshape(n, 4800), axis=1) + 1e-20)
    if sung is None:
        sung = e > e.max() - 30.0
    v = e[sung[:n]]
    return np.percentile(v, 95) - np.percentile(v, 10), sung


def test_the_ruler():
    rng = np.random.default_rng(1)
    # bursts of random length and level, some 35 dB down (outside the 30 dB band)
    chunks = []
    for _ in range(60):
        db = rng.choice([0.0, -4.0, -9.0, -15.0, -22.0, -35.0])
        chunks.append(10 ** (db / 20) * rng.standard_normal((int(rng.uniform(0.03, 0.4) * SR), 2)))
    x = np.concatenate(chunks)
    want, sung = _ruler(x)
    got, got_sung = planner.vocal_range_db(x, SR)
    assert got == pytest.approx(want, abs=1e-9)
    assert np.array_equal(got_sung, sung)
    # the windows found before are the ones read after
    y = x * np.linspace(1.0, 0.05, x.shape[0])[:, None]
    assert planner.vocal_range_db(y, SR, sung)[0] == pytest.approx(_ruler(y, sung)[0], abs=1e-9)
    assert planner.vocal_range_db(y, SR, sung)[0] != pytest.approx(_ruler(y)[0], abs=0.05)
    assert planner.vocal_range_db(np.zeros((4800 * 5, 2)), SR)[0] is None


def test_an_even_voice_is_compressed_less(even):
    stems, p = even
    planner.validate(p)
    t = _lead(p)
    dial = t["comp_style_dial"]
    assert dial["law"] == "vocal_range_fit_v1"
    assert dial["target_db"] == TARGET
    assert dial["range_in_db"] < TARGET
    # the precondition that makes this test mean something: the style's own
    # setting really did squash this voice
    assert dial["style_takes_db"] > 1.0
    assert dial["fit_amount"] < dial["amount"]
    assert dial["takes_db"] <= TOL + 0.05
    style = planner._comp_params("vocal", dial["amount"])
    c = t["comp"]
    assert c["threshold_db"] > style["threshold_db"] and c["ratio"] < style["ratio"]
    assert c == planner._comp_params("vocal", dial["fit_amount"])
    assert any("vocal compression" in e and "17 dB" in e for e in t["evidence"])


def test_a_dynamic_voice_keeps_the_style(dynamic):
    _, p = dynamic
    planner.validate(p)
    t = _lead(p)
    dial = t["comp_style_dial"]
    assert dial["range_in_db"] > TARGET + 3.0
    assert dial["fit_amount"] == dial["amount"]
    assert dial["takes_db"] == dial["style_takes_db"]
    assert t["comp"] == planner._comp_params("vocal", dial["amount"])
    assert not any("vocal compression" in e for e in t["evidence"])


def test_the_renderer_takes_what_the_receipt_says(even):
    stems, p = even
    t = _lead(p)
    # not vacuous: the simulated path has a ride and a corrective EQ to get right
    assert t.get("ride") and t.get("eq")
    seen = {}
    orig = render._apply_comp

    def spy(x, sr, spec, law="centred_box_v1"):
        y = orig(x, sr, spec, law)
        if spec == t["comp"] and "in" not in seen:
            seen["in"], seen["out"] = np.array(x), np.array(y)
        return y

    render._apply_comp = spy
    try:
        render.render(stems, p)
    finally:
        render._apply_comp = orig
    r_in, sung = planner.vocal_range_db(seen["in"], SR)
    r_out, _ = planner.vocal_range_db(seen["out"], SR, sung)
    dial = t["comp_style_dial"]
    assert r_in == pytest.approx(dial["range_in_db"], abs=0.5)
    assert r_in - r_out == pytest.approx(dial["takes_db"], abs=0.5)
    assert r_in - r_out < dial["style_takes_db"] - 0.5


def test_how_hot_the_file_was_delivered_no_longer_decides():
    """The style's threshold is absolute and the gain stage keeps half of the
    source balance, so the same voice delivered louder used to be squashed more.
    Fitted, an even voice keeps its range at any delivered level."""
    dials = [_lead(planner.build_plan(_band(10.0, vocal_gain=g), planner.load_preset("pop_2020s")))["comp_style_dial"]
             for g in (1.0, 3.0, 6.0)]
    assert [d["range_in_db"] for d in dials] == pytest.approx([dials[0]["range_in_db"]] * 3, abs=0.2)
    # the old behaviour, as the precondition: louder file, more squashed
    assert dials[0]["style_takes_db"] < 1.0 < 2.5 < dials[1]["style_takes_db"] < dials[2]["style_takes_db"]
    for d in dials:
        assert d["takes_db"] <= TOL + 0.05


def test_only_the_lead_vocal_is_fitted(even):
    stems, p = even
    preset = planner.load_preset("pop_2020s")
    intensity = float(p["style"]["intensity"])
    (bv,) = [t for t in p["tracks"] if t["role"] == "backing_vocal"]
    assert "comp_style_dial" not in bv
    assert bv["comp"] == planner._comp_params("vocal", float(preset["compression"]["vocal"]) * intensity)
    for t in p["tracks"]:
        if t["role"] not in analyze.VOCAL_ROLES:
            assert (t.get("comp_style_dial") or {}).get("law") != "vocal_range_fit_v1", t["id"]


def test_the_receipt_is_not_read_by_the_renderer(even):
    stems, p = even
    q = copy.deepcopy(p)
    for t in q["tracks"]:
        t.pop("comp_style_dial", None)
    a, _ = render.render(stems, p)
    b, _ = render.render(stems, q)
    assert np.array_equal(a, b)


def test_nothing_to_measure_keeps_the_style():
    st = analyze.Stem(name="Lead Vocal.wav", path="x", sr=SR, audio=np.zeros((int(0.5 * SR), 2), np.float32))
    assert planner._fit_vocal_comp(st, 0.0, [], 80.0, [], 0.55, "vocal") == (None, None)


def test_a_voice_too_hot_for_any_compression_gets_none():
    st = _band(10.0, vocal_gain=3.0)[3]
    assert st.role == "lead_vocal"
    comp, dial = planner._fit_vocal_comp(st, 30.0, [], 80.0, [], 0.55, "vocal")
    assert comp is None
    assert dial["fit_amount"] == 0.0 and dial["takes_db"] == 0.0 and dial["style_takes_db"] > 1.0


def test_a_protected_lead_vocal_is_not_fitted():
    stems = _band(10.0, vocal_gain=3.0)
    second = analyze.Stem(name="Lead Vocal B.wav", path="Lead Vocal B.wav", sr=SR,
                          audio=(0.8 * stems[3].audio).astype(np.float32))
    stems.append(second)
    analyze.detect_roles(stems)
    analyze.measure(second)
    leads = [s for s in stems if s.role == "lead_vocal"]
    assert len(leads) == 2
    ids = [analyze.track_id(s.name) for s in leads]
    free = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert all(t["comp_style_dial"]["law"] == "vocal_range_fit_v1" for t in free["tracks"] if t["id"] in ids)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"),
                           capture_groups=[{"id": "vox", "members": ids, "pan": 0.0}])
    for t in p["tracks"]:
        if t["id"] in ids:
            assert (t.get("comp_style_dial") or {}).get("law") != "vocal_range_fit_v1", t["id"]


def test_the_simulation_is_the_renderers_level_path():
    """Gain, high-pass, then the compressor with the law new plans render with
    (trailing_box_v1: the detector reads only the past).  The legacy centred
    detector would take a different amount, so this pins the law too."""
    from automix import dsp
    st = _band(10.0, vocal_gain=3.0)[3]
    x = np.asarray(st.audio, dtype=np.float32) * np.float32(dsp.db_to_lin(-5.0))
    x = dsp.apply_eq(x, SR, [("highpass", 80.0, 0.0, 0.707)])
    r0, sung = _ruler(x)
    takes = {law: r0 - _ruler(render._apply_comp(x, SR, planner._comp_params("vocal", 0.55), law), sung)[0]
             for law in ("trailing_box_v1", "centred_box_v1")}
    assert abs(takes["trailing_box_v1"] - takes["centred_box_v1"]) > 0.3  # the check can tell them apart
    _, dial = planner._fit_vocal_comp(st, -5.0, [], 80.0, [], 0.55, "vocal")
    assert dial["range_in_db"] == pytest.approx(r0, abs=0.01)
    assert dial["style_takes_db"] == pytest.approx(takes["trailing_box_v1"], abs=0.02)
    # and the fit is the tightest setting inside the allowance (17 dB target, 0.25 dB grain):
    # a hair more compression would already take too much
    assert dial["range_in_db"] < TARGET and dial["fit_amount"] < 0.55
    more = planner._comp_params("vocal", dial["fit_amount"] + 0.01)
    over = r0 - _ruler(render._apply_comp(x, SR, more, "trailing_box_v1"), sung)[0]
    assert over > TOL


def test_silence_is_not_a_range():
    three_s = np.zeros((3 * SR, 2), np.float32)
    assert planner.vocal_range_db(three_s, SR) == (None, None)
    # 1 s of voice and 4 s of nothing: 10 windows to read, too few to call it a range
    rng = np.random.default_rng(2)
    sparse = np.concatenate([0.1 * rng.standard_normal((SR, 2)), np.zeros((4 * SR, 2))])
    assert planner.vocal_range_db(sparse, SR)[0] is None
    st = analyze.Stem(name="Lead Vocal.wav", path="x", sr=SR, audio=three_s)
    assert planner._fit_vocal_comp(st, 0.0, [], 80.0, [], 0.55, "vocal") == (None, None)


@pytest.mark.parametrize("amount", [0.511, 0.568, 0.639, 0.73])
def test_the_fit_never_rounds_into_too_much(amount):
    # every setting takes too much of this voice (+30 dB into the compressor); the
    # search ends just under 0.02, which must stay "no compressor", never 0.02
    st = _band(10.0, vocal_gain=3.0)[3]
    comp, dial = planner._fit_vocal_comp(st, 30.0, [], 80.0, [], amount, "vocal")
    assert comp is None and dial["fit_amount"] == 0.0


def test_a_manual_trim_is_part_of_the_level_path():
    stems = _band(10.0, vocal_gain=3.0)
    lead_id = analyze.track_id(stems[3].name)
    p0 = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"), trims={lead_id: 6.0})
    t0, t = _lead(p0), _lead(p)
    assert t["gain_db"] == pytest.approx(t0["gain_db"] + 6.0, abs=0.01)
    # 6 dB hotter into the same style setting takes more of the voice ...
    assert t["comp_style_dial"]["style_takes_db"] > t0["comp_style_dial"]["style_takes_db"] + 1.0
    # ... and the fit was made on that hotter path, so the renderer takes what the receipt says
    seen = {}
    orig = render._apply_comp

    def spy(x, sr, spec, law="centred_box_v1"):
        y = orig(x, sr, spec, law)
        if spec == t.get("comp") and "in" not in seen:
            seen["in"], seen["out"] = np.array(x), np.array(y)
        return y

    render._apply_comp = spy
    try:
        render.render(stems, p)
    finally:
        render._apply_comp = orig
    r_in, sung = planner.vocal_range_db(seen["in"], SR)
    r_out, _ = planner.vocal_range_db(seen["out"], SR, sung)
    assert r_in - r_out == pytest.approx(t["comp_style_dial"]["takes_db"], abs=0.5)
    assert r_in - r_out <= TOL + 0.5


def test_the_ruler_on_a_long_take():
    """More windows than one pass of the block loop (2048): the blocks must join
    exactly, so a long session reads the same as a short one would."""
    sr = 8000
    w = int(0.1 * sr)
    rng = np.random.default_rng(4)
    levels = rng.choice([0.0, -5.0, -11.0, -18.0, -26.0, -45.0], size=5000)
    # the first window of the second block is the loudest of the take: a seam that
    # loses or repeats it moves the whole 30 dB window
    levels[2048] = 9.0
    x = np.repeat(10 ** (levels / 20), w)[:, None] * rng.standard_normal((5000 * w, 2))
    m = np.mean(x ** 2, axis=1)
    e = 10 * np.log10(np.mean(m.reshape(5000, w), axis=1) + 1e-20)
    sung = e > e.max() - 30.0
    want = np.percentile(e[sung], 95) - np.percentile(e[sung], 10)
    got, got_sung = planner.vocal_range_db(x, sr)
    assert got == pytest.approx(want, abs=1e-9)
    assert np.array_equal(got_sung, sung)


def test_a_plan_whose_fit_is_none_has_no_compressor():
    """Pushed 18 dB by a manual trim, every setting of this voice's compressor takes
    too much: the plan must drop the style's compressor, not keep it."""
    stems = _band(10.0, vocal_gain=3.0)
    lead_id = analyze.track_id(stems[3].name)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"), trims={lead_id: 18.0})
    planner.validate(p)
    t = _lead(p)
    assert t["comp_style_dial"]["fit_amount"] == 0.0 and t["comp_style_dial"]["style_takes_db"] > 1.0
    assert "comp" not in t
    mix, _ = render.render(stems, p)
    assert np.isfinite(mix).all()
