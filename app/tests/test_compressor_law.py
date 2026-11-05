"""R119: a compressor's detector reads only the past (compressor_law trailing_box_v1).

The compressor's attack/release windows were centred on the sample, so the
detector saw half a window ahead: with a 30 ms attack a drum hit's first 5 ms
were turned down more than its body, and the bed was turned down before the
hit.  A slow attack is supposed to do the opposite -- let the hit's start
through and hold the body -- which is where a compressor's punch comes from.
New plans carry compressor_law: trailing_box_v1; plans without it render
with the centred detector they were made with (test_replay_r11*.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from automix import dsp, plan as planner, render

SR = 48000


def _hits(seconds=4):
    t = np.arange(SR * seconds)
    bed = 0.01 * np.random.default_rng(1).standard_normal(t.size)
    hit = np.zeros(t.size)
    n = np.arange(int(0.25 * SR))
    for s in range(SR // 2, t.size - SR // 2, SR // 2):
        hit[s:s + n.size] += np.sin(2 * np.pi * 80 * n / SR) * np.exp(-n / SR / 0.05)
    x = bed + 0.5 * hit
    return np.stack([x, x], 1)


def _gain_db(x, y, a_ms, b_ms, at=SR):
    a, b = at + int(a_ms * SR / 1000), at + int(b_ms * SR / 1000)
    g = 20 * np.log10(np.abs(y[a:b, 0]) + 1e-12) - 20 * np.log10(np.abs(x[a:b, 0]) + 1e-12)
    return float(np.median(g))


def _comp(x, law, attack_ms=30.0, release_ms=120.0):
    return dsp.compress(x, SR, 1.0, threshold_db=-30.0, ratio_max=4.0, attack_ms=attack_ms,
                        release_ms=release_ms, makeup_mode="none", law=law)


def test_the_new_detector_lets_a_slow_attack_through_and_holds_the_body():
    x = _hits()
    y = _comp(x, "trailing_box_v1")
    assert _gain_db(x, y, 0, 5) > -0.5          # the hit's start passes
    assert _gain_db(x, y, 30, 100) < -6.0       # the body is held
    assert _gain_db(x, y, -40, -20) > -0.1      # nothing happens before the hit


def test_the_old_detector_is_kept_for_old_plans_and_is_what_was_wrong():
    x = _hits()
    y = _comp(x, "centred_box_v1")
    # documented, not endorsed: the old detector turns the start down more
    # than the body, and dips the bed before the hit
    assert _gain_db(x, y, 0, 5) < _gain_db(x, y, 30, 100) - 2.0
    assert _gain_db(x, y, -40, -20) < -2.0
    # and the default is still the old detector, so old callers are unchanged
    assert np.array_equal(dsp.compress(x, SR, 1.0, threshold_db=-30.0, ratio_max=4.0, attack_ms=30.0,
                                       release_ms=120.0, makeup_mode="none"), y)


def test_a_steady_tone_is_compressed_the_same_by_both_detectors():
    s = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR)
    s = np.stack([s, s], 1)
    old = _comp(s, "centred_box_v1", 10.0, 100.0)[SR // 2:SR]
    new = _comp(s, "trailing_box_v1", 10.0, 100.0)[SR // 2:SR]
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    assert 20 * np.log10(rms(new) / rms(old)) == pytest.approx(0.0, abs=0.02)


def test_the_trailing_average_reads_only_the_past():
    x = np.zeros(100)
    x[50] = 1.0
    y = dsp._box_trailing(x, 10)
    assert not y[:50].any()                      # nothing before the impulse
    assert y[50:60] == pytest.approx(np.full(10, 0.1))
    assert not y[60:].any()
    assert dsp._box_trailing(np.arange(5.0), 1).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_an_unknown_law_is_refused():
    with pytest.raises(ValueError):
        dsp.compress(_hits(1), SR, 1.0, law="lookahead_v9")


def test_the_parallel_copy_follows_the_law():
    x = _hits()
    old = dsp.parallel_compress(x, SR, 0.5, attack_ms=30.0, law="centred_box_v1")
    new = dsp.parallel_compress(x, SR, 0.5, attack_ms=30.0, law="trailing_box_v1")
    assert not np.array_equal(old, new)
    assert np.array_equal(dsp.parallel_compress(x, SR, 0.5, attack_ms=30.0), old)


def _session(seconds=8):
    from automix import analyze
    t = np.arange(SR * seconds) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    sings = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    stems = []
    for name, a in (("Kick.wav", .5 * kick), ("Bass.wav", .1 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t)),
                    ("Lead Vocal.wav", .1 * sings * np.sin(2 * np.pi * 440 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_new_plans_carry_the_law_and_every_compressor_gets_it(monkeypatch):
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    assert p["compressor_law"] == "trailing_box_v1"
    seen = []
    real = dsp.compress

    def spy(*a, **kw):
        seen.append(kw.get("law", "centred_box_v1"))
        return real(*a, **kw)
    monkeypatch.setattr(dsp, "compress", spy)
    _, rep = render.render(stems, p)
    assert seen and set(seen) == {"trailing_box_v1"}
    assert rep["compressor_law"] == "trailing_box_v1"
    # a plan without the field: the old detector everywhere, and no report key
    seen.clear()
    old = dict(p)
    old.pop("compressor_law")
    _, rep = render.render(stems, old)
    assert seen and set(seen) == {"centred_box_v1"}
    assert "compressor_law" not in rep


def test_a_wrong_law_in_a_plan_is_refused():
    stems = _session(4)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p["compressor_law"] = "lookahead_v9"
    with pytest.raises(render.RenderError):
        render.render(stems, p)


def test_nothing_before_a_hit_is_touched():
    """Below the threshold the new compressor is a wire, right up to the hit:
    neither the detector nor the gain smoothing reaches ahead."""
    s = SR // 2
    rng = np.random.default_rng(7)
    x = 0.001 * rng.standard_normal((SR, 2))
    x[s:] += 0.5
    y = dsp.compress(x, SR, 1.0, threshold_db=-30.0, ratio_max=4.0, attack_ms=2.0, release_ms=80.0,
                     makeup_mode="none", law="trailing_box_v1")
    assert np.array_equal(y[:s], x[:s])
    old = dsp.compress(x, SR, 1.0, threshold_db=-30.0, ratio_max=4.0, attack_ms=2.0, release_ms=80.0,
                       makeup_mode="none", law="centred_box_v1")
    assert not np.array_equal(old[:s], x[:s])
