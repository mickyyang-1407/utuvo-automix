"""R117: saturation that follows each passage's level, oversampled.

peak_tanh_v1 (plans without ``saturation_law``) normalised the tanh to a
track's single largest peak at the sample rate: below that peak the curve is
a gain of up to k/tanh(k), so every saturated track and bus also compressed
the loud parts against the quiet ones, and harmonics above Nyquist folded
back.  env_tanh_os4_v1 normalises to the signal's own 30 ms peak envelope at
4x the sample rate and keeps the RMS.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000
NEW = "env_tanh_os4_v1"


def _stereo(x):
    return np.stack([x, x], axis=1)


def _line_db(y, f, sr=SR):
    seg = y[sr:3 * sr, 0] * np.hanning(2 * sr)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(seg.size, 1 / sr)
    i = int(np.argmin(np.abs(freqs - f)))
    return 20 * np.log10(spec[i - 3:i + 4].max() + 1e-20)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-20)


@pytest.mark.parametrize("drive", [0.1, 0.25, 0.35])
def test_a_steady_full_scale_tone_gets_the_same_harmonics(drive):
    t = np.arange(SR * 4) / SR
    x = _stereo(0.9 * np.sin(2 * np.pi * 440 * t))
    old = dsp.saturate(x, drive)
    new = dsp.saturate(x, drive, law=NEW, sr=SR)
    for h in (3, 5):
        rel_old = _line_db(old, 440 * h) - _line_db(old, 440)
        rel_new = _line_db(new, 440 * h) - _line_db(new, 440)
        assert rel_new == pytest.approx(rel_old, abs=0.5)


def test_the_loud_quiet_contour_is_kept():
    t = np.arange(SR * 4) / SR
    x = np.sin(2 * np.pi * 200 * t)
    x[:2 * SR] *= 0.1  # a passage 20 dB quieter
    x = _stereo(0.9 * x)
    def contour(y):
        return _rms_db(y[3 * SR:4 * SR - 100]) - _rms_db(y[SR // 2:2 * SR - 1000])
    old = dsp.saturate(x, 0.25)
    new = dsp.saturate(x, 0.25, law=NEW, sr=SR)
    assert contour(new) == pytest.approx(20.0, abs=0.1)
    # the old law brought the quiet passage up: a compressor
    assert contour(old) < 18.0


def test_harmonics_above_nyquist_do_not_fold_back():
    t = np.arange(SR * 4) / SR
    x = _stereo(0.9 * np.sin(2 * np.pi * 15000 * t))
    # the 3rd harmonic (45 kHz) folds to 3 kHz at 48 kHz without oversampling
    old = dsp.saturate(x, 0.35)
    new = dsp.saturate(x, 0.35, law=NEW, sr=SR)
    assert _line_db(old, 3000) - _line_db(old, 15000) > -30.0
    assert _line_db(new, 3000) - _line_db(new, 15000) < -70.0


def test_the_level_is_kept_and_silence_stays_silent():
    rng = np.random.default_rng(1)
    x = _stereo(0.2 * rng.standard_normal(SR * 3))
    x[SR:2 * SR] = 0.0
    y = dsp.saturate(x, 0.3, law=NEW, sr=SR)
    assert _rms_db(y) == pytest.approx(_rms_db(x), abs=0.01)
    # a band-limited result cannot be exactly zero next to a hard cut: the
    # brick-wall at the original Nyquist rings.  250 ms from the cut it is
    # below -80 dBFS (measured -88 on this -14 dBFS noise)
    quiet = y[SR + SR // 4:2 * SR - SR // 4]
    assert quiet.shape[0] == SR // 2
    assert 20 * np.log10(np.max(np.abs(quiet)) + 1e-300) < -80.0
    assert y.dtype == x.dtype and y.shape == x.shape
    # a block with nothing in it is skipped: exact silence stays exact
    z = np.zeros((SR * 20, 2))
    z[:SR] = x[:SR]
    out = dsp.saturate(z, 0.3, law=NEW, sr=SR)
    assert not np.any(out[9 * SR:])
    assert not np.any(dsp.saturate(np.zeros((SR, 2)), 0.3, law=NEW, sr=SR))


def test_a_long_signal_has_no_seams_between_blocks():
    """Processed in 8 s blocks with 0.25 s of context: across a join the
    result matches the same signal processed in one piece.  The tone's
    frequency and phase put the cuts away from its zero crossings."""
    t = np.arange(SR * 20) / SR
    x = _stereo(0.7 * np.sin(2 * np.pi * 111.3 * t + 2.0))
    y = dsp.saturate(x, 0.25, law=NEW, sr=SR)
    whole = dsp._saturate_env_os(x, 0.25, SR, block_s=1000.0)
    for cut in (8 * SR, 16 * SR):
        assert abs(x[cut, 0]) > 0.3  # the cut is not at a zero crossing
        d = y[cut - 2400:cut + 2400, 0] - whole[cut - 2400:cut + 2400, 0]
        assert _rms_db(d) - _rms_db(whole[cut - 2400:cut + 2400, 0]) < -80.0


def test_mono_input_works():
    x = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    y = dsp.saturate(x, 0.2, law=NEW, sr=SR)
    assert y.shape == x.shape


def test_the_law_needs_the_rate_and_an_unknown_law_is_refused():
    x = np.zeros((10, 2))
    x[3] = 0.5
    with pytest.raises(ValueError):
        dsp.saturate(x, 0.2, law=NEW)
    with pytest.raises(ValueError):
        dsp.saturate(x, 0.2, law="tape_v9", sr=SR)


def test_a_drive_of_zero_is_bypassed():
    x = np.random.default_rng(2).standard_normal((SR, 2))
    assert dsp.saturate(x, 0.0, law=NEW, sr=SR) is x


def _stems():
    t = np.arange(SR * 8) / SR
    stems = []
    for name, a in (("Kick.wav", .5 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 2 * t) > .9)),
                    ("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t)),
                    ("Lead Vocal.wav", .1 * np.sin(2 * np.pi * 440 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_new_plans_carry_the_law_and_the_renderer_uses_it():
    stems = _stems()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    assert p["saturation_law"] == NEW
    calls = []
    real = dsp.saturate

    def spy(x, drive, law="peak_tanh_v1", sr=None):
        calls.append(law)
        return real(x, drive, law=law, sr=sr)
    dsp.saturate = spy
    try:
        _, rep = render.render(stems, p)
        new_calls = list(calls)
        calls.clear()
        old = copy.deepcopy(p)
        old.pop("saturation_law")
        _, rep_old = render.render(stems, old)
    finally:
        dsp.saturate = real
    # every saturation in the plan (tracks, music bus, vocal character) used it
    assert new_calls and set(new_calls) == {NEW}
    assert calls and set(calls) == {"peak_tanh_v1"}
    assert rep["saturation_law"] == NEW and "saturation_law" not in rep_old


def test_an_unknown_plan_law_is_refused():
    stems = _stems()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p["saturation_law"] = "tape_v9"
    with pytest.raises(render.RenderError):
        render.render(stems, p)
