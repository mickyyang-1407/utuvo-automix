"""R114: a bigger, less squashed mix with a wetter vocal.

Reported 2026-11-04: "整個 mix 都會有一點小夾子器或是 over compress" (the fast
limiter flutters at audio rate), "歌的畫面要很大" (a wider top end), "vocal
都會有一點偏幹，你的 reverb 要用好一點" (a reverb that sounds like a space).
"""
import numpy as np
import pytest

from automix import dsp

SR = 48000


def _hf_share_db(y):
    f = np.fft.rfftfreq(len(y), 1 / SR)
    s = np.abs(np.fft.rfft(y[:, 0])) ** 2
    return 10 * np.log10(s[f > 1000].sum() / s.sum())


def _loud_bass_with_kicks():
    n = SR * 6
    t = np.arange(n) / SR
    x = .6 * np.sin(2 * np.pi * 80 * t)
    k = np.arange(int(.12 * SR)) / SR
    hit = np.sin(2 * np.pi * 55 * k) * np.exp(-k / .05)
    for s in range(0, n - hit.size, SR // 2):
        x[s:s + hit.size] += .9 * hit
    return np.stack([x, x], 1) * 2.5


def test_the_release_limiter_adds_far_less_distortion():
    x = _loud_bass_with_kicks()
    fast = dsp.true_peak_limit(x, SR, -1.0, law="tp_fast_v1")
    slow = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert _hf_share_db(slow) < _hf_share_db(fast) - 6.0
    assert dsp.true_peak_db(slow, SR) <= -1.0


def test_the_release_limiter_holds_the_ceiling_on_isolated_peaks():
    x = np.zeros((SR * 2, 2))
    x[SR:SR + 50] = 3.0
    y = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert dsp.true_peak_db(y, SR) <= -1.0


def test_plate_v2_sounds_like_a_space():
    ir = dsp.make_ir_plate_v2(SR, decay_s=2.0, predelay_ms=0.0, high_cut_hz=12000.0)
    assert np.sum(ir ** 2) == pytest.approx(1.0, rel=1e-6)
    f = np.fft.rfftfreq(ir.shape[0], 1 / SR)
    s = np.abs(np.fft.rfft(ir[:, 0])) ** 2
    assert s[f < 120].sum() / s.sum() < .01                       # no low end in the return
    late = ir[int(1.0 * SR):]
    early = ir[: int(.3 * SR)]
    def band(x, lo, hi):
        F = np.fft.rfftfreq(x.shape[0], 1 / SR); S = np.abs(np.fft.rfft(x[:, 0])) ** 2
        return S[(F >= lo) & (F < hi)].sum()
    # highs die faster than the mids: their share falls from early to late
    assert band(late, 5000, 10000) / band(late, 600, 3000) < .5 * band(early, 5000, 10000) / band(early, 600, 3000)
    # early reflections: distinct peaks before 50 ms stand out of the build-up
    first = np.abs(ir[: int(.05 * SR), :]).max()
    assert first > 4 * np.abs(ir[int(.002 * SR): int(.006 * SR)]).max()
    # the two sides are not the same reverb
    assert abs(np.corrcoef(ir[:, 0], ir[:, 1])[0, 1]) < .2


def test_side_air_widens_the_top_and_leaves_mono_alone():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR * 2, 2)) * .1
    x[:, 1] = .7 * x[:, 0] + .3 * x[:, 1]
    y = dsp.side_shelf(x, SR, 2.0)
    assert np.allclose(y[:, 0] + y[:, 1], x[:, 0] + x[:, 1], atol=1e-9)   # mono fold-down unchanged
    def side_hf(z):
        s = .5 * (z[:, 0] - z[:, 1]); F = np.fft.rfftfreq(len(s), 1 / SR); S = np.abs(np.fft.rfft(s)) ** 2
        return 10 * np.log10(S[F > 9000].sum())
    assert side_hf(y) - side_hf(x) == pytest.approx(2.0, abs=.4)


def test_the_release_limiter_holds_the_ceiling_across_its_block_edges():
    # blocks are 8 s: put peaks right at the edges and in between
    x = .05 * np.random.default_rng(3).standard_normal((SR * 20, 2))
    for at in (8 * SR - 30, 8 * SR, 16 * SR + 7, 12 * SR):
        x[at:at + 60] = 2.5
    y = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert dsp.true_peak_db(y, SR) <= -1.0
    # the release carries on across the edge: a peak that ends 3 ms before
    # 8 s is not seen by the next block's own samples, yet 10 ms after the
    # edge the gain is still well down (30 dB/s from ~9 dB)
    x2 = .05 * np.random.default_rng(4).standard_normal((SR * 12, 2))
    x2[8 * SR - 200: 8 * SR - 140] = 2.5
    y2 = dsp.true_peak_limit(x2, SR, -1.0, law="tp_release_v1")
    at = 8 * SR + int(.01 * SR)
    seg = slice(at, at + 48)
    gain = np.sqrt(np.mean(y2[seg] ** 2) / np.mean(x2[seg] ** 2))
    assert gain < .5, gain


def test_drums_and_bass_both_land_on_their_targets():
    from automix import analyze, plan as planner, render
    t = np.arange(SR * 12) / SR
    rng = np.random.default_rng(9)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    add("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t) + .02 * rng.standard_normal(t.size))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    for name, pb in rep["part_balance"].items():
        assert pb["result_db"] == pytest.approx(pb["target_db"], abs=.5), (name, pb)
