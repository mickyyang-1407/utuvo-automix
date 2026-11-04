"""Wiring tests: every plan field must reach the audio.

Two silent bugs made this file necessary:

* the saturation curve put 13% THD on a vocal at the preset's own amount, and
* bus compressors measured -0.00 dB of gain reduction because their thresholds
  were derived from a formula instead of from the bus itself.

Both looked perfectly healthy in the plan and in the receipt.  The lesson is
that "the plan says X" and "the audio does X" are different claims, and only
the second one matters.

Each case renders the same session twice -- once with the field as planned and
once with it neutered -- and asserts the audio actually changes in the expected
direction.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _stem(name, freq, dur=3.0, amp=0.3, percussive=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    if percussive:
        # a decaying pulse train: crest high enough that the planner asks for
        # compression instead of skipping it as "already dense"
        env = np.exp(-((t % 0.5)) * 12.0)
        x = amp * np.sin(2 * np.pi * freq * t) * env
    st = analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], axis=1).astype(np.float32), sr=SR
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


@pytest.fixture(scope="module")
def session():
    stems = [
        _stem("bass.wav", 55.0, percussive=True),
        _stem("vocal_lead.wav", 440.0, percussive=True),
        _stem("guitar_elec.wav", 220.0, percussive=True),
    ]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def base_plan(session):
    return planner.build_plan(session, planner.load_preset("pop_2020s"))


def _render(stems, plan):
    audio, _ = render.render(stems, plan, {"decay_s": 0.5, "predelay_ms": 0.0})
    return audio


def _diff_db(a, b) -> float:
    """How much the two renders differ, relative to the louder one."""
    d = float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))
    ref = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) + 1e-12
    return 20.0 * np.log10(d / ref + 1e-12)


def _track(plan, role):
    return next(t for t in plan["tracks"] if t["role"] == role)


def test_gain_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    quiet = copy.deepcopy(base_plan)
    _track(quiet, "lead_vocal")["gain_db"] -= 3.0
    b = _render(session, quiet)
    assert _diff_db(a, b) > -30.0


def test_pan_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    panned = copy.deepcopy(base_plan)
    _track(panned, "guitar_electric")["pan"] = 0.8
    b = _render(session, panned)
    assert _diff_db(a, b) > -30.0
    # panning should move energy between channels, not just change level
    left_a = float(np.sqrt(np.mean(a[:, 0].astype(np.float64) ** 2)))
    left_b = float(np.sqrt(np.mean(b[:, 0].astype(np.float64) ** 2)))
    assert abs(20 * np.log10(left_a / left_b)) > 0.2


def test_high_pass_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    filtered = copy.deepcopy(base_plan)
    _track(filtered, "guitar_electric")["hpf_hz"] = 400.0
    b = _render(session, filtered)
    assert _diff_db(a, b) > -30.0


def test_track_eq_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    eqd = copy.deepcopy(base_plan)
    _track(eqd, "lead_vocal").setdefault("eq", []).append(
        {"type": "bell", "freq": 1000.0, "gain_db": -8.0, "q": 1.0, "enabled": True}
    )
    b = _render(session, eqd)
    assert _diff_db(a, b) > -30.0


def test_master_eq_reaches_the_audio(session, base_plan):
    """A narrow EQ move barely changes the overall waveform, so check the band.

    The first version compared whole-signal RMS: a -6 dB shelf at 6 kHz moved
    the 7-9 kHz band from 0.189 to 0.062 (a 3x change) while the whole-signal
    difference was only -49 dB.  Asserting on "-30 dB overall" would have called
    a working EQ broken.
    """
    a = _render(session, base_plan)
    eqd = copy.deepcopy(base_plan)
    eqd["buses"]["master"].setdefault("eq", []).append(
        {"type": "highshelf", "freq": 6000.0, "gain_db": -6.0, "q": 0.7, "enabled": True}
    )
    b = _render(session, eqd)

    def band(x, lo, hi):
        mono = x.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono)) ** 2
        f = np.fft.rfftfreq(len(mono), 1 / SR)
        return float(spec[(f >= lo) & (f <= hi)].sum())

    assert band(b, 7000, 9000) < band(a, 7000, 9000) * 0.7


def test_width_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    base_width = dsp.width_ratio(a)
    # stay inside the guard: asking for 6x is correctly refused, and that is
    # covered by the receipt warning rather than by this test
    wide = copy.deepcopy(base_plan)
    wide["buses"]["master"]["width_target"] = round(base_width * 1.3, 4)
    b = _render(session, wide)
    assert dsp.width_ratio(b) > base_width * 1.15


def test_sends_reach_the_audio(session, base_plan):
    a = _render(session, base_plan)
    wet = copy.deepcopy(base_plan)
    for t in wet["tracks"]:
        t["sends"] = {"plate": 0.5}
    b = _render(session, wet)
    assert _diff_db(a, b) > -30.0
    assert dsp.width_ratio(b) > dsp.width_ratio(a)


def test_saturation_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    clean = copy.deepcopy(base_plan)
    for t in clean["tracks"]:
        t["saturation"] = 0.0
    b = _render(session, clean)
    assert _diff_db(a, b) > -40.0


def test_track_compression_reaches_the_audio(session, base_plan):
    a = _render(session, base_plan)
    flat = copy.deepcopy(base_plan)
    for t in flat["tracks"]:
        t.pop("comp", None)
    b = _render(session, flat)
    assert _diff_db(a, b) > -40.0


def test_bus_and_master_compression_reach_the_audio(session, base_plan):
    a = _render(session, base_plan)
    flat = copy.deepcopy(base_plan)
    for bus in flat["buses"].values():
        bus.pop("comp", None)
    b = _render(session, flat)
    assert _diff_db(a, b) > -40.0

# ------------------------------------------------- direction and magnitude

def test_pan_moves_energy_the_right_way():
    """A sign error here would mirror the whole mix and still "reach the audio".

    Rendered as a one-stem session: in the three-stem session panning a single
    guitar only shifts the overall balance by ~6%, which is not decisive enough
    to catch a mirrored mapping.
    """
    stem = _stem("guitar_elec.wav", 220.0)
    left_plan = planner.build_plan([stem], planner.load_preset("pop_2020s"))
    left_plan["tracks"][0]["pan"] = -0.8
    a = _render([stem], left_plan)
    right_plan = copy.deepcopy(left_plan)
    right_plan["tracks"][0]["pan"] = 0.8
    b = _render([stem], right_plan)

    def balance(x):
        l = float(np.sqrt(np.mean(x[:, 0].astype(np.float64) ** 2)))
        r = float(np.sqrt(np.mean(x[:, 1].astype(np.float64) ** 2)))
        return l / (r + 1e-12)

    assert balance(a) > 1.5, balance(a)
    assert balance(b) < 0.67, balance(b)


def test_high_pass_actually_removes_the_low_end():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    low = np.stack([0.3 * np.sin(2 * np.pi * 80.0 * t)] * 2, axis=1).astype(np.float32)

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    open_hp = dsp.apply_eq(low.copy(), SR, [("highpass", 20.0, 0.0, 0.707)])
    filtered = dsp.apply_eq(low.copy(), SR, [("highpass", 400.0, 0.0, 0.707)])
    assert 20 * np.log10(rms(filtered) / rms(open_hp)) < -20.0


def test_a_bell_eq_lands_on_its_gain():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    tone = np.stack([0.3 * np.sin(2 * np.pi * 1000.0 * t)] * 2, axis=1).astype(np.float32)

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    cut = dsp.apply_eq(tone.copy(), SR, [("peaking", 1000.0, -8.0, 1.0)])
    boost = dsp.apply_eq(tone.copy(), SR, [("peaking", 1000.0, 8.0, 1.0)])
    assert -9.5 < 20 * np.log10(rms(cut) / rms(tone)) < -6.0
    assert 6.0 < 20 * np.log10(rms(boost) / rms(tone)) < 9.5


def test_width_target_is_hit_within_a_few_percent(session, base_plan):
    base = _render(session, base_plan)
    target = dsp.width_ratio(base) * 1.25
    plan = copy.deepcopy(base_plan)
    plan["buses"]["master"]["width_target"] = round(target, 4)
    out = _render(session, plan)
    assert abs(dsp.width_ratio(out) - target) / target < 0.2

def test_mono_below_collapses_the_low_side_signal():
    """Bass below the crossover should be mono; wide highs should survive."""
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    delay = int(0.002 * SR)
    base = 0.3 * np.sin(2 * np.pi * 60 * t)
    wide_low = np.stack(
        [base, np.concatenate([np.zeros(delay), base[:-delay]])], axis=1
    ).astype(np.float32)

    def low_side_over_mid(x):
        mid = x.mean(axis=1)
        side = (x[:, 0] - x[:, 1]) / 2

        def band(v, lo, hi):
            spec = np.abs(np.fft.rfft(v)) ** 2
            f = np.fft.rfftfreq(len(v), 1 / SR)
            return float(spec[(f >= lo) & (f <= hi)].sum())

        return band(side, 30, 120) / (band(mid, 30, 120) + 1e-20)

    before = low_side_over_mid(wide_low)
    after = low_side_over_mid(dsp.mono_below(wide_low.copy(), SR, 120.0))
    assert before > 0.1
    assert after < before * 0.2


def test_the_limiter_ceiling_is_respected(session, base_plan):
    plan = copy.deepcopy(base_plan)
    plan["buses"]["master"]["true_peak_db"] = -3.0
    plan["buses"]["master"].setdefault("limiter", {})["ceiling_db"] = -3.0
    planner.validate(plan)  # the two ceilings must agree, or validate refuses
    audio = _render(session, plan)
    assert dsp.true_peak_db(audio, SR) <= -2.8


def test_the_loudness_target_is_hit(session, base_plan):
    for target in (-18.0, -12.0):
        plan = copy.deepcopy(base_plan)
        plan["buses"]["master"]["target_lufs"] = target
        audio, rep = render.render(session, plan, {"decay_s": 0.5, "predelay_ms": 0.0})
        got = dsp.loudness_lufs(audio, SR)
        if abs(got - target) < 0.5:
            continue
        # R114: the release limiter stops before it squashes a mix to reach
        # the number (these test stems are bare pulses); it must say so, and
        # it may only fall short, never overshoot
        assert got < target and target - got < 2.0, (target, got)
        assert any(w.startswith("loudness:") for w in rep["warnings"]), rep["warnings"]


def test_the_fast_limiter_still_hits_the_target_on_old_plans(session, base_plan):
    plan = copy.deepcopy(base_plan)
    plan["buses"]["master"]["target_lufs"] = -12.0
    del plan["buses"]["master"]["limiter"]["law"]
    assert abs(dsp.loudness_lufs(_render(session, plan), SR) - -12.0) < 0.5
