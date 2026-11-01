"""Regression tests for the DSP primitives.

The bug this file exists for: ``apply_eq`` called ``np.fft.rfft(x, nf)``
without ``axis=0``.  numpy's rfft defaults to the *last* axis, so a
``(samples, channels)`` array was transformed across its 2 channels,
zero-padded to ``nf``, and the resulting ``(samples, nf//2+1)`` spectrum
was multiplied by the EQ curve and inverse transformed back to
``(samples, nf)`` -- a 48000 x 48000 float64 array, i.e. ~18 GB.  The
symptom was an apparent hang followed by the OOM killer, not an error.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automix import dsp  # noqa: E402

SR = 48000


def _stereo(n: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2)) * 0.1


def _mono(n: int = SR) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal(n) * 0.1


# ---------------------------------------------------------------- layout

def test_apply_eq_keeps_stereo_shape():
    x = _stereo()
    y = dsp.apply_eq(x, SR, [("peaking", 1000.0, 3.0, 1.0)])
    assert y.shape == x.shape
    assert y.nbytes == x.nbytes


def test_apply_eq_keeps_mono_shape():
    x = _mono()
    y = dsp.apply_eq(x, SR, [("peaking", 1000.0, 3.0, 1.0)])
    assert y.shape == x.shape


def test_apply_eq_short_buffer_is_not_quadratic():
    """A 0.25 s stereo buffer must stay small; the old bug allocated n*n."""
    n = SR // 4
    x = _stereo(n)
    y = dsp.apply_eq(x, SR, [("peaking", 800.0, 6.0, 0.9)])
    assert y.shape == (n, 2)
    assert y.nbytes == x.nbytes


def test_tonal_tilt_shape_and_identity():
    x = _stereo()
    flat = dict.fromkeys(("low", "low_mid", "mid", "presence", "air"), 0.0)
    y = dsp.tonal_tilt(x, SR, flat)
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_mono_below_shape():
    x = _stereo()
    y = dsp.mono_below(x, SR, 120.0)
    assert y.shape == x.shape


def test_mono_below_collapses_sub_crossover_to_mono():
    """Only content *below* the crossover is summed to mono.

    The two channels carry the same 50 Hz tone at different levels, so the
    input really is stereo in the sub band.  After mono_below the channels
    must agree (that is the whole point of the operation) and the energy must
    survive -- collapsing to mono averages the two levels, it does not gate.
    """
    n = SR
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 50.0 * t)
    x = np.stack([tone, 0.5 * tone], axis=1)
    # crossover well above the tone, so the 50 Hz content sits deep in the low
    # band and the highpass leakage that stays legitimately stereo is tiny
    y = dsp.mono_below(x, SR, 500.0)
    assert y.shape == x.shape
    # stereo in, mono in the sub band out.  Agreement is asserted relative to
    # peak: whatever the highpass leaves above the crossover stays stereo by
    # design, so the two channels match only to within that residue.
    assert not np.allclose(x[:, 0], x[:, 1], atol=1e-2)
    peak = np.max(np.abs(y))
    assert np.max(np.abs(y[:, 0] - y[:, 1])) < 0.02 * peak
    # the sub-band survives rather than being nulled out
    assert np.sqrt(np.mean(y ** 2)) > 0.05


def test_mono_below_cancels_anti_phase_sub():
    """Anti-phase sub content must cancel, not be preserved.

    This is the intended consequence of summing the low band to mono: a
    perfectly out-of-phase sub is inaudible on a mono club system anyway, so
    the collapse to mono is what makes the low end predictable.  Asserted
    explicitly so nobody "fixes" the cancellation later.
    """
    n = SR
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 50.0 * t)
    x = np.stack([tone, -tone], axis=1)
    y = dsp.mono_below(x, SR, 500.0)
    # the low band is gone; what is left is only the highpass transition band,
    # so drop is measured relative to the input rather than as absolute silence
    ratio = np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))
    assert ratio < 0.35, f"anti-phase sub not cancelled, kept {ratio:.2f}"
    # and the residue is still out of phase, i.e. it is the stereo high band
    assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] < 0.0


def test_mono_below_zero_is_noop():
    x = _stereo()
    np.testing.assert_array_equal(dsp.mono_below(x, SR, 0.0), x)


# ---------------------------------------------------------------- dynamics

def test_dynamics_keep_shape():
    x = _stereo()
    for out in (
        dsp.compress(x, SR, 0.5),
        dsp.saturate(x, 0.4),
        dsp.soft_limit(x, -1.0),
        dsp.reverb_send(x, SR, 0.3, 1.5, 20.0, 8000.0),
    ):
        assert out.shape == x.shape


def test_compress_lowers_crest_factor():
    x = _stereo()
    y = dsp.compress(x, SR, 1.0, threshold_db=-30.0)
    assert dsp.crest_db(y) < dsp.crest_db(x)


def test_soft_limit_respects_ceiling():
    x = _stereo() * 100.0
    y = dsp.soft_limit(x, -1.0)
    assert np.max(np.abs(y)) <= dsp.db_to_lin(-1.0) + 1e-9


def test_reverb_zero_amount_is_silent_wet():
    x = _stereo()
    wet = dsp.reverb_send(x, SR, 0.0, 1.0, 10.0, 8000.0)
    assert wet.shape == x.shape
    assert not np.any(wet)


def test_make_ir_decays():
    ir = dsp.make_ir(SR, decay_s=1.0, predelay_ms=0.0, high_cut_hz=8000.0)
    assert ir.shape == (SR, 2)
    head = np.sqrt(np.mean(ir[:1000] ** 2))
    tail = np.sqrt(np.mean(ir[-1000:] ** 2))
    assert head > tail * 2.0


# ---------------------------------------------------------------- metering

def test_loudness_on_reference_tone():
    """BS.1770 sanity against the 997 Hz calibration tone.

    A full-scale (amplitude 1.0) 997 Hz sine measures mean square 0.5, i.e.
    -3.01 dBFS RMS.  K-weighting adds about +0.7 dB at 997 Hz, so:

      * one channel only  -> -3.01 LUFS
      * the same in both  ->  0.00 LUFS  (the +3 dB of a second equal channel)

    Both figures are what a reference meter reports, and both are asserted
    here because getting the channel axis wrong hides as a plausible-looking
    "-3 dB either way" answer.
    """
    n = SR * 2
    t = np.arange(n) / SR
    sine = np.sin(2 * np.pi * 997.0 * t)
    mono = dsp.loudness_lufs(np.stack([sine], axis=1), SR)
    dual = dsp.loudness_lufs(np.stack([sine, sine], axis=1), SR)
    assert abs(mono - (-3.01)) < 0.1, f"mono {mono:.2f}"
    assert abs(dual - 0.0) < 0.1, f"dual {dual:.2f}"
    # -20 dB of amplitude is exactly -20 LUFS, whatever the absolute anchor
    quiet = dsp.loudness_lufs(np.stack([0.1 * sine, 0.1 * sine], axis=1), SR)
    assert abs(quiet - (dual - 20.0)) < 0.05, f"quiet {quiet:.2f}"


def test_loudness_scales_with_level():
    """Doubling amplitude must add 6 dB of loudness, no more, no less."""
    n = SR * 2
    t = np.arange(n) / SR
    sine = np.sin(2 * np.pi * 1000 * t)
    quiet = dsp.loudness_lufs(np.stack([0.05 * sine] * 2, axis=1), SR)
    loud = dsp.loudness_lufs(np.stack([0.10 * sine] * 2, axis=1), SR)
    assert abs((loud - quiet) - 6.02) < 0.15


def test_loudness_of_silence_is_floor():
    assert dsp.loudness_lufs(np.zeros((SR, 2)), SR) == -70.0


def test_true_peak_not_below_sample_peak():
    x = _stereo()
    sample_peak = dsp.lin_to_db(np.max(np.abs(x)))
    assert dsp.true_peak_db(x, SR) >= sample_peak - 1e-6


def _plateau_source(n: int = SR, amp: float = 0.75, fade: int = 4800) -> np.ndarray:
    """Synthetic flat-top fixture: 0.75 plateau, half-cosine fades at both ends.

    A hand-made signal, not a claim about what audio can physically reach.
    Band-limited interpolation may reconstruct slightly above the plateau, so
    the 0.05 dB allowance used with this fixture is fixture allowance only --
    it is not a standards figure and not a guarantee about the true peak of
    real material.
    """
    x = np.full(n, amp, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
    x[:fade] = amp * ramp
    x[n - fade:] = amp * ramp[::-1]
    return x


@pytest.mark.parametrize("channels", [1, 2])
def test_true_peak_ignores_context_ringing(channels):
    """A block's context is FFT scaffolding, not signal.

    With 0.08 s blocks the meter cuts the signal into short segments and
    pads each one with context so the FFT has something to chew on.  The
    cut leaves ringing in the *context*, and the old meter took its maximum
    over the whole upsampled array -- context included -- so it reported the
    overshoot of the cut (~-1.39 dB) instead of the plateau (-2.50 dB).
    Each block must only be charged for the samples it owns.
    """
    mono = _plateau_source()
    x = mono if channels == 1 else np.stack([mono, 0.5 * mono], axis=1)
    bound_db = dsp.lin_to_db(0.75)
    meter = dsp.true_peak_db(x, SR, block_s=0.08)
    # 0.05 dB is fixture allowance for the fixture, not a standards claim.
    assert meter <= bound_db + 0.05, f"meter {meter:.6f} dB above bound {bound_db:.6f} dB"
    assert meter >= bound_db - 1e-6, f"meter {meter:.6f} dB below sample peak {bound_db:.6f} dB"


def test_true_peak_detects_intersample_peak():
    """A real inter-sample peak must still be found.

    12 kHz at 48 kHz is four samples per cycle, so a pi/4 phase puts the
    sine's maximum *between* samples: the analytic peak is 3 dB above the
    sample peak, and the meter's 4x grid lands on that maximum.
    """
    n = SR
    t = np.arange(n) / SR
    amp = 0.5
    x = amp * np.sin(2 * np.pi * 12000 * t + np.pi / 4)
    analytic_db = dsp.lin_to_db(amp)
    sample_db = dsp.lin_to_db(float(np.max(np.abs(x))))
    meter = dsp.true_peak_db(x, SR)
    assert meter > sample_db + 2.5, f"meter {meter:.6f} dB vs sample peak {sample_db:.6f} dB"
    assert abs(meter - analytic_db) < 0.05, f"meter {meter:.6f} dB vs analytic {analytic_db:.6f} dB"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("position", ["first", "last", "middle", "burst"])
def test_true_peak_impulse_never_below_sample_peak(channels, position):
    """Short impulses at the edges and in the middle must never read below
    the sample peak -- cropping the context may not lose the samples."""
    n = SR
    mono = np.zeros(n)
    if position == "first":
        mono[0] = 1.0
    elif position == "last":
        mono[n - 1] = 1.0
    elif position == "middle":
        mono[12345] = 1.0
    else:
        mono[12345:12350] = 1.0
    x = mono if channels == 1 else np.stack([mono, np.zeros(n)], axis=1)
    sample_db = dsp.lin_to_db(1.0)
    meter = dsp.true_peak_db(x, SR)
    assert meter >= sample_db - 1e-6, f"meter {meter:.6f} dB below sample peak {sample_db:.6f} dB"


def test_true_peak_empty_and_silence_floors():
    """Empty input keeps its sentinel, non-empty silence keeps the floor."""
    assert dsp.true_peak_db(np.zeros((0, 2)), SR) == -120.0
    silence = dsp.true_peak_db(np.zeros((SR, 2)), SR)
    assert silence == dsp.lin_to_db(0.0)
    assert silence <= -200.0


# ------------------------------------------------ true-peak Nyquist-bin split

def _nyquist_alt(n: int, amp: float = 0.25) -> np.ndarray:
    """Alternating sequence: all of it lives in the Nyquist bin."""
    return amp * (-1.0) ** np.arange(n)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("oversample", [2, 4])
def test_true_peak_nyquist_bin_is_not_doubled(channels, oversample):
    """The Nyquist bin of an even-length FFT is a singleton, not a pair.

    Copied into a longer inverse it turns into an ordinary +/- frequency
    pair, so its contribution is counted twice and the meter reads 6 dB
    high.  Under the periodic Fourier interpolation this meter uses, the
    reconstruction of an alternating sequence is that sequence, so the peak
    must stay at the sample amplitude -- a property of that convention,
    not a claim that the underlying analogue waveform is unique.
    """
    n = 1000
    mono = _nyquist_alt(n)
    x = mono if channels == 1 else np.stack([mono, 0.5 * mono], axis=1)
    meter = dsp.true_peak_db(x, SR, oversample=oversample)
    expect = dsp.lin_to_db(0.25)
    assert abs(meter - expect) < 1e-10, f"meter {meter!r} dB vs expected {expect!r} dB"


def test_true_peak_oversample_one_keeps_the_nyquist_bin():
    """Without oversampling the Nyquist bin is still the singleton it was."""
    n = 1000
    x = _nyquist_alt(n)
    meter = dsp.true_peak_db(x, SR, oversample=1)
    expect = dsp.lin_to_db(0.25)
    assert abs(meter - expect) < 1e-10, f"meter {meter!r} dB vs expected {expect!r} dB"


@pytest.mark.parametrize("oversample", [2, 4])
def test_true_peak_odd_length_highest_bin_is_not_nyquist(oversample):
    """An odd FFT length has no Nyquist bin -- its highest bin is a normal pair.

    n=15 puts the energy in bin 7.  Halving that bin would be wrong, and
    the reconstruction has to stay exact.

    The DC offset and the phase put the interpolated peak on the last point
    of the oversampled grid and nowhere in the original samples, so the
    sample floor cannot stand in for it: a reconstruction that dropped the
    highest bin would fall back to the (smaller) sample peak and fail.
    """
    n = 15
    j = np.arange(n)
    phase = 2 * np.pi * 7 / (n * oversample)
    x = 0.1 + 0.25 * np.cos(2 * np.pi * 7 * j / n + phase)
    sample_peak = float(np.max(np.abs(x)))
    assert sample_peak < 0.35 - 1e-4, (
        f"sample peak {sample_peak} is not below the interpolated peak"
    )
    meter = dsp.true_peak_db(x, SR, oversample=oversample)
    expect = dsp.lin_to_db(0.35)
    assert abs(meter - expect) < 1e-10, f"meter {meter!r} dB vs expected {expect!r} dB"


def test_true_peak_singleton_dc_is_unchanged():
    """A one-sample signal has no Nyquist bin to split."""
    meter = dsp.true_peak_db(np.array([0.5]), SR, oversample=4)
    expect = dsp.lin_to_db(0.5)
    assert abs(meter - expect) < 1e-10, f"meter {meter!r} dB vs expected {expect!r} dB"


def test_true_peak_reconstruction_passes_through_the_original_samples(monkeypatch):
    """The upsampled reconstruction must pass through the original samples.

    DC + a partial tone + Nyquist on an even length: the band-limited
    reconstruction of the block has to hit the original samples exactly on
    the original grid, which only happens when the Nyquist bin is weighted
    once instead of twice.  The meter's own inverse return is inspected, so
    no second meter is needed.
    """
    n = 64
    j = np.arange(n)
    x = 0.3 + 0.2 * np.cos(2 * np.pi * 3 * j / n) + 0.25 * (-1.0) ** j

    taken = []
    real_irfft = np.fft.irfft

    def spy(a, m, *args, **kwargs):
        out = real_irfft(a, m, *args, **kwargs)
        taken.append(out)
        return out

    monkeypatch.setattr(np.fft, "irfft", spy)
    meter = dsp.true_peak_db(x, SR, oversample=4)
    assert taken, "true_peak_db never called irfft"
    up = taken[-1] * 4
    assert up.shape == (n * 4,)
    assert np.allclose(up[::4], x, atol=1e-9), (
        f"original grid off by {np.max(np.abs(up[::4] - x)):.3e}"
    )
    assert abs(meter - dsp.lin_to_db(0.75)) < 1e-10, f"meter {meter!r} dB"


def test_meters_on_a_pure_tone():
    n = SR
    t = np.arange(n) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    assert abs(dsp.spectral_centroid(x, SR) - 440.0) < 20.0
    assert dsp.stereo_correlation(x) > 0.99
    assert dsp.width_ratio(x) < 0.01
    bands = dsp.band_energy(x, SR)
    assert bands.shape == (5,)
    assert abs(float(bands.sum()) - 1.0) < 1e-6
    assert bands[1] + bands[2] > 0.9


# ---------------------------------------------------------------- utility

def test_resample_fft_lengths():
    x = _mono(SR)
    y = dsp.resample_fft(x, SR, 44100)
    assert abs(y.shape[0] - 44100) <= 1
    assert dsp.resample_fft(y, 44100, SR).shape == x.shape


def test_next_fast_len_is_five_smooth():
    for n in (1, 17, 480, 44100, 48000, 96001):
        m = dsp._next_fast_len(n)
        assert m >= n
        rest = m
        for p in (2, 3, 5):
            while rest % p == 0:
                rest //= p
        assert rest == 1, f"{m} is not 5-smooth"

def test_saturation_is_warmth_not_fuzz():
    """drive 0.1 must not put 13% THD on a sustained tone.

    The curve is peak-normalised, so the mapping decides everything.  Regression
    guard for the ``1 + 6*drive`` version, which measured 10% THD at drive 0.05
    and 20% at 0.2 -- audible fuzz on every vocal in every preset.
    """
    SR = 48000
    n = 4 * SR
    t = np.arange(n) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1)

    def thd(v):
        mono = v.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        f = np.fft.rfftfreq(len(mono), 1 / SR)

        def peak(hz, w=6.0):
            m = (f >= hz - w) & (f <= hz + w)
            return float(spec[m].max()) if np.any(m) else 0.0

        return 100 * np.sqrt(sum(peak(440 * k) ** 2 for k in (2, 3, 4, 5))) / peak(440)

    assert thd(dsp.saturate(x, 0.03)) < 1.0
    assert thd(dsp.saturate(x, 0.10)) < 4.0
    # and it must still get aggressive when asked
    assert thd(dsp.saturate(x, 0.35)) > 8.0
    # monotone in drive
    values = [thd(dsp.saturate(x, d)) for d in (0.05, 0.15, 0.3)]
    assert values[0] < values[1] < values[2]


# The plain all-channels resampler on the R63 grid (both FFT grids span the
# same duration; scale M/nf).  R63 deliberately replaced the R9 grid, which
# drifted the clock; these tests still check that the memory-saving paths
# return exactly what the plain algorithm returns.
_next_fast_len = dsp._next_fast_len
def _plain_resample_oracle(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    n = x.shape[0]
    m = int(round(n * target / sr))
    common = math.gcd(sr, target)
    numerator, denominator = target // common, sr // common
    nf = denominator * _next_fast_len((max(n, m) + denominator - 1) // denominator)
    M = (nf // denominator) * numerator
    # All channels at once: one transform each way, and the transform runs
    # along the contiguous axis instead of on a strided column view.
    if x.ndim == 2:
        X = np.fft.rfft(x, nf, axis=0)
        Y = np.zeros((M // 2 + 1, x.shape[1]), dtype=X.dtype)
        k = min(len(X), len(Y))
        Y[:k] = X[:k]
        del X
        return np.fft.irfft(Y, M, axis=0)[:m] * (M / nf)
    X = np.fft.rfft(x, nf)
    Y = np.zeros(M // 2 + 1, dtype=X.dtype)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    del X
    return np.fft.irfft(Y, M)[:m] * (M / nf)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sr,target", [(44100,48000),(48000,44100)])
@pytest.mark.parametrize("layout", ["C","strided","F"])
def test_resample_workspace_matches_the_plain_algorithm_for_long_stereo(dtype,sr,target,layout):
    rng=np.random.default_rng(210)
    backing=rng.standard_normal((2_000_006,2)).astype(dtype)*.04
    x=backing[::2] if layout=="strided" else backing[:1_000_003].copy(order=layout)
    original=x.copy()
    expected=_plain_resample_oracle(x,sr,target)
    x.flags.writeable=False
    actual=dsp.resample_fft(x,sr,target)
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    np.testing.assert_array_equal(x,original)


@pytest.mark.parametrize("shape,dtype", [((0,2),np.float64),((17,),np.float32),((17,0),np.float64),
    ((17,1),np.float64),((17,3),np.float64),((17,2),np.float16),((17,2),np.int16),
    ((17,2),np.longdouble),((17,2),np.complex128),((4,2,2),np.float64)])
def test_resample_workspace_preserves_other_legacy_contracts(shape,dtype):
    x=np.ones(shape,dtype=dtype)
    try:
        expected=_plain_resample_oracle(x,44100,48000)
    except Exception as old_error:
        with pytest.raises(type(old_error)) as current:
            dsp.resample_fft(x,44100,48000)
        assert str(current.value)==str(old_error)
    else:
        actual=dsp.resample_fft(x,44100,48000)
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype==expected.dtype and actual.strides==expected.strides
