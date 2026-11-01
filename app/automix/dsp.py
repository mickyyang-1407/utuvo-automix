"""Vectorised DSP primitives. No scipy: everything is numpy + FFT.

Design notes
------------
* Filters are applied **zero-phase in the frequency domain** (magnitude only).
  For a mix-prep MVP this is a feature: no phase smearing, no filter
  instability, and it costs one rfft/irfft pair per stage.
* The compressor uses a *smoothed magnitude envelope* (box filters via cumsum)
  instead of a per-sample one-pole recursion. Fully vectorised, and the
  attack/release asymmetry is approximated by blending two smoothing widths.
* Reverb is FFT convolution with a generated exponentially-decaying IR.
"""

from __future__ import annotations

import math

import numpy as np

from . import _memory

EPS = 1e-12


# ---------------------------------------------------------------- helpers


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, EPS)))


def _box_smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, O(n) via cumulative sum. Edge-padded."""
    width = max(1, int(width))
    if width == 1:
        return x.copy()
    pad = width // 2
    xp = np.pad(x, (pad, width - 1 - pad), mode="edge")
    c = np.cumsum(xp, dtype=np.float64)
    c = np.concatenate(([0.0], c))
    out = (c[width:] - c[:-width]) / width
    return out[: x.shape[0]]


def _next_fast_len(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n, for fast FFTs."""
    if n <= 16:
        return n
    best = 1 << int(np.ceil(np.log2(n)))
    for a in range(0, 30):
        p2 = 2**a
        if p2 > best:
            break
        for b in range(0, 20):
            p3 = p2 * 3**b
            if p3 > best:
                break
            for c in range(0, 14):
                p5 = p3 * 5**c
                if p5 >= n:
                    best = min(best, p5)
                    break
    return best


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution, same length as x (tail truncated)."""
    n = x.shape[0] + h.shape[0] - 1
    nf = _next_fast_len(n)
    X = np.fft.rfft(x, nf)
    H = np.fft.rfft(h, nf)
    y = np.fft.irfft(X * H, nf)
    return y[: x.shape[0]]


# ---------------------------------------------------------------- EQ


def _biquad_response(kind: str, freq: float, gain_db: float, q: float, sr: int):
    """Magnitude response of an RBJ biquad, evaluated at rfft bin freqs."""
    f = np.fft.rfftfreq(_EQ_N, 1.0 / sr)
    w0 = 2.0 * np.pi * min(max(freq, 1.0), sr * 0.49) / sr
    alpha = np.sin(w0) / (2.0 * max(q, 0.05))
    A = 10.0 ** (gain_db / 40.0)

    if kind == "peaking":
        b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
    elif kind == "lowshelf":
        s = 2.0 * np.sqrt(max(A, EPS)) * alpha
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + s)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - s)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + s
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - s
    elif kind == "highshelf":
        s = 2.0 * np.sqrt(max(A, EPS)) * alpha
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + s)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - s)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + s
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - s
    elif kind in ("lowpass", "highpass"):
        if kind == "lowpass":
            b0, b1, b2 = (1 - np.cos(w0)) / 2, 1 - np.cos(w0), (1 - np.cos(w0)) / 2
        else:
            b0, b1, b2 = (1 + np.cos(w0)) / 2, -(1 + np.cos(w0)), (1 + np.cos(w0)) / 2
        a0, a1, a2 = 1 + alpha, -2 * np.cos(w0), 1 - alpha
    else:
        raise ValueError(f"unknown filter kind: {kind}")

    z = np.exp(-2j * np.pi * f / sr)
    num = b0 + b1 * z + b2 * z * z
    den = a0 + a1 * z + a2 * z * z
    return np.abs(num / den)


_EQ_N = 0  # set per call


def apply_eq(x: np.ndarray, sr: int, bands: list[tuple]) -> np.ndarray:
    """Apply a chain of RBJ biquads, zero-phase, in one pass.

    bands: list of (kind, freq_hz, gain_db, q)
    """
    global _EQ_N
    n = x.shape[0]
    nf = _next_fast_len(n)
    _EQ_N = nf

    H = np.ones(nf // 2 + 1, dtype=np.float64)
    for kind, freq, gain_db, q in bands:
        if abs(gain_db) < 1e-3 and kind in ("peaking", "lowshelf", "highshelf"):
            continue
        H *= _biquad_response(kind, freq, gain_db, q, sr)

    # NB: axis=0 is load-bearing. rfft defaults to the *last* axis, which for a
    # (samples, channels) array means transforming across channels and
    # broadcasting the result into an (N, N) matrix.
    X = np.fft.rfft(x, nf, axis=0)
    H2 = H[:, None] if x.ndim == 2 else H
    y = np.fft.irfft(X * H2, nf, axis=0)
    # come back in the caller's dtype: mixing buffers are float32, and letting
    # every filter promote them to float64 doubles the memory for nothing
    return y[:n].astype(x.dtype, copy=False)


def tonal_tilt(x: np.ndarray, sr: int, tilt: dict[str, float]) -> np.ndarray:
    """5-band tilt from a preset's `tonal_tilt_db`."""
    bands = [
        ("lowshelf", 120.0, tilt.get("low", 0.0), 0.7),
        ("peaking", 400.0, tilt.get("low_mid", 0.0), 0.9),
        ("peaking", 1200.0, tilt.get("mid", 0.0), 0.9),
        ("peaking", 3200.0, tilt.get("presence", 0.0), 0.8),
        ("highshelf", 9000.0, tilt.get("air", 0.0), 0.7),
    ]
    return apply_eq(x, sr, [b for b in bands if abs(b[2]) > 1e-3])


def mono_below(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Collapse content below `hz` to mono (keeps the low end solid)."""
    if hz <= 0:
        return x
    mono = x.mean(axis=1, keepdims=True)
    low = apply_eq(mono, sr, [("lowpass", hz, 0.0, 0.707)])
    high = apply_eq(x, sr, [("highpass", hz, 0.0, 0.707)])
    return low + high


# ---------------------------------------------------------------- dynamics


def _env(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Smoothed magnitude envelope, attack/release approximated by box blend."""
    mag = np.sqrt(np.mean(x * x, axis=1)) if x.ndim == 2 else np.abs(x)
    a = max(1, int(sr * attack_ms / 1000.0))
    r = max(1, int(sr * release_ms / 1000.0))
    ea = _box_smooth(mag, a)
    er = _box_smooth(mag, r)
    # fast movement uses the attack window, slow body uses the release window
    return np.maximum(ea, er * 0.85)


def compress(
    x: np.ndarray,
    sr: int,
    amount: float,
    threshold_db: float = -24.0,
    ratio_max: float = 4.0,
    attack_ms: float = 12.0,
    release_ms: float = 120.0,
    makeup_mode: str = "half_average_reduction",
    makeup_cap_db: float = 12.0,
) -> np.ndarray:
    """Soft-knee downward compressor. `amount` in 0..1 scales ratio & depth."""
    if amount <= 1e-3:
        return x
    env = _env(x, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    over = np.maximum(env_db - threshold_db, 0.0)
    ratio = 1.0 + (ratio_max - 1.0) * float(np.clip(amount, 0.0, 1.0))
    gain_db = -over * (1.0 - 1.0 / ratio)
    # soften the knee so it doesn't sound like a gate
    knee = 6.0
    soft = np.clip((env_db - (threshold_db - knee / 2)) / knee, 0.0, 1.0)
    gain_db = gain_db * (0.35 + 0.65 * soft)
    g = 10.0 ** (gain_db / 20.0)
    g = _box_smooth(g, max(1, int(sr * 0.004)))
    # auto make-up: recover roughly half the average reduction.  The rule
    # comes from the plan (mode and cap), so the plan still determines the
    # audio; "makeup_db: 0" would have been untrue, and a rule is honest.
    if makeup_mode == "none":
        makeup = 0.0
    elif makeup_mode == "full_reduction":
        makeup = float(np.clip(-gain_db.mean(), 0.0, makeup_cap_db))
    else:  # half_average_reduction
        makeup = float(np.clip(-gain_db.mean() * 0.5, 0.0, makeup_cap_db))
    gd = (g * db_to_lin(makeup)).astype(x.dtype, copy=False)
    return x * gd[:, None] if x.ndim == 2 else x * gd


def _band_split(x: np.ndarray, sr: int, freq: float, q: float = 0.7):
    """Zero-phase split into (below, above) around ``freq``."""
    n = x.shape[0]
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)[:, None]
    # 2nd-order-ish highpass response, magnitude only (zero phase)
    fr = np.maximum(f, 1e-9) / freq
    hp = (fr ** 2) / np.sqrt((fr ** 2 - 1.0) ** 2 + (fr / max(q, 0.05)) ** 2)
    high = np.fft.irfft(spec * hp, n=n, axis=0)
    return x - high, high


def deess(
    x: np.ndarray,
    sr: int,
    amount: float = 0.6,
    freq: float = 6000.0,
    q: float = 0.7,
    range_db: float = 12.0,
) -> np.ndarray:
    """Reduce the loudest sibilant bursts in the high band only.

    Split at ``freq``, follow the high band's envelope, and pull gain *only*
    where that envelope is near its own loudest (the S sounds).  The band is
    recombined with the untouched low part, so the body of the vocal is not
    high-passed or otherwise coloured -- which is what separates a de-esser
    from turning the treble down.

    ``amount`` is how many dB of reduction per dB above the sibilance
    threshold; ``range_db`` caps the total so a bad take cannot be gutted.
    """
    if x.shape[0] == 0 or amount <= 0.0:
        return x
    low, high = _band_split(x, sr, freq, q)
    env = np.sqrt(np.mean(high * high, axis=1)) if high.ndim == 2 else np.abs(high)
    if float(np.max(env)) <= 1e-9:
        return x

    env = _box_smooth(env, max(1, int(0.002 * sr)))   # 2 ms
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    # the sibilance threshold is relative to this stem's own loudest highs,
    # so a quiet take and a hot take get the same *musical* result
    thr = float(np.percentile(env_db, 97.0)) - 6.0
    over = np.maximum(env_db - thr, 0.0)
    reduction = np.minimum(over * float(amount), float(range_db))
    gain = (10.0 ** (-reduction / 20.0)).astype(np.float64)
    if gain.ndim == 1 and x.ndim == 2:
        gain = gain[:, None]
    return (low + high * gain).astype(x.dtype)


def saturate(x: np.ndarray, drive: float) -> np.ndarray:
    """tanh saturation with level compensation. drive 0..1.

    The curve is normalised to the signal's own peak, so the *shape* of the
    drive curve is what decides how much harmonic distortion a low setting
    produces.  The previous mapping (``1 + 6*drive``) put a full-scale sine at
    k=1.6 for drive 0.1 -- measured 13% THD on a vocal, which is fuzz, not
    warmth.  ``5*drive`` gives ~2% at 0.1 and still 15% at 0.35 where the lofi
    and rock presets actually want grit.
    """
    if drive <= 1e-3:
        return x
    k = 5.0 * float(np.clip(drive, 0.0, 1.0))
    peak = np.max(np.abs(x)) + EPS
    y = np.tanh(k * x / peak)
    return y * (peak / (np.max(np.abs(y)) + EPS)) * 0.98



def soft_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Smoothed gain reduction towards a ceiling; transparent below it.

    This bounds *sample* peaks.  It cannot bound true peak: the final clip
    sprays harmonics towards Nyquist, and the reconstructed waveform then
    overshoots between samples -- on dense material by up to ~2.7 dB, and
    *more* the harder you clamp.  Use :func:`true_peak_limit` when the number
    that matters is dBTP.
    """
    ceil = db_to_lin(ceiling_db)
    mag = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    # widen the lookaround so we don't chase single samples
    from numpy import maximum
    guard = maximum(mag, 1e-9)
    need = np.minimum(1.0, ceil / guard)
    smooth = _box_smooth(need, 64)  # ~1.3 ms at 48k
    gain = np.minimum(need, smooth * 1.15)
    gain = gain.astype(x.dtype, copy=False)
    y = x * gain[:, None] if x.ndim == 2 else x * gain
    # final safety clip
    return np.clip(y, -db_to_lin(ceiling_db + 0.2), db_to_lin(ceiling_db + 0.2))


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling maximum over a window of ``w`` samples (centred, edge-clamped).

    Implemented as ``w`` shifted maxima, which is O(n*w) but with w in the
    tens and n in the hundreds of thousands -- fast enough, and far easier to
    trust than a deque-based rewrite.
    """
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, w - pad - 1), mode="edge")
    out = xp[: x.size].copy()
    for k in range(1, w):
        np.maximum(out, xp[k : k + x.size], out=out)
    return out


# ---------------------------------------------------------------- space


def make_ir(
    sr: int,
    decay_s: float = 1.5,
    predelay_ms: float = 20.0,
    high_cut_hz: float = 8000.0,
    seed: int = 7,
) -> np.ndarray:
    """A synthetic, decorrelated stereo impulse response.

    Exponentially decaying noise, high-cut with a zero-phase filter, plus a
    short pre-delay.  Not a real room, but it is stereo, it decays, and it is
    deterministic -- which is what the MVP needs from a reverb until there is
    an IR library behind it.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(decay_s * sr))
    t = np.arange(n) / sr
    # -60 dB at decay_s, so the tail is genuinely gone by the end
    env = np.exp(-6.9078 * t / max(decay_s, 1e-3))
    ir = rng.standard_normal((n, 2)) * env[:, None]
    ir = apply_eq(ir, sr, [("lowpass", min(high_cut_hz, sr * 0.45), 0.0, 0.707)])

    pre = int(max(0.0, predelay_ms) * sr / 1000.0)
    if pre > 0:
        ir = np.vstack([np.zeros((pre, 2)), ir])

    # Normalise by *energy*, not by peak.  Peak-normalising a 1.6 s noise tail
    # leaves an IR with roughly 30 dB more energy than a unit impulse, so a
    # send of 0.1 arrived 8 dB louder than the dry signal and the reverb
    # swallowed the mix.  With unit energy the send amount means what it says.
    energy = float(np.sqrt(np.sum(ir ** 2)))
    if energy > 0:
        ir /= energy
    return ir


def reverb_send(
    dry: np.ndarray,
    sr: int,
    amount: float,
    decay_s: float = 1.5,
    predelay_ms: float = 20.0,
    high_cut_hz: float = 8000.0,
) -> np.ndarray:
    """Convolution reverb *send*: returns the wet signal only.

    Returning wet-only keeps the caller in charge of the dry/wet balance, which
    is what a send is for.  ``amount`` 0 returns silence rather than the dry
    signal so that summing sends can never double the dry path.
    """
    if amount <= 1e-6:
        return np.zeros_like(dry)
    if dry.ndim == 1:
        dry = dry[:, None]
    ir = make_ir(sr, decay_s, predelay_ms, high_cut_hz)
    wet = np.stack([fft_convolve(dry[:, c], ir[:, c]) for c in range(dry.shape[1])], axis=1)
    return (wet * float(amount)).astype(dry.dtype, copy=False)


# ------------------------------------------------------------- metering math


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply both BS.1770 stages in a single pass over all channels.

    Two stages applied one after the other meant four full-length FFTs per
    channel.  The stages are LTI, so their magnitudes multiply: one pass, one
    transform for the whole (samples, channels) array.  On a 3-minute session
    this is the difference between 1.0 s and 0.5 s per stem, and measurement was
    the slowest part of the whole pipeline.
    """
    if x.ndim == 1:
        x = x[:, None]
    (b1, a1), (b2, a2) = _k_weighting_coeffs()
    n = x.shape[0]
    nf = _next_fast_len(n)
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    z = np.exp(-2j * np.pi * f / sr)

    H = np.ones_like(f)
    for b, a in ((b2, a2), (b1, a1)):
        num = b[0] + b[1] * z + b[2] * z * z
        den = a[0] + a[1] * z + a[2] * z * z
        H *= np.abs(num / den)
    del f, z, num, den

    # A transposed version of this was measured: it needs a full copy of the
    # data and came out 11% *slower* than transforming axis 0 directly.
    X = np.fft.rfft(x, nf, axis=0)
    # axis=0 again: rfft defaults to the last axis, which for (samples, channels)
    # would transform across channels and broadcast into a huge matrix
    weighted = X * H[:, None]
    del X, H
    return np.fft.irfft(weighted, nf, axis=0)[:n].astype(x.dtype, copy=False)


def _k_weighting_coeffs():
    """BS.1770-4 K-weighting: (shelf, RLB high-pass), both at 48 kHz.

    The coefficients are the ones published in the recommendation for 48 kHz;
    ``loudness_lufs`` resamples to 48 kHz before using them, which is why they
    do not take ``sr`` as an argument.
    """
    # stage 1: high shelf (+4 dB above ~1.5 kHz)
    b1 = (1.53512485958697, -2.69169618940638, 1.19839281085285)
    a1 = (1.0, -1.69065929318241, 0.73248077421585)
    # stage 2: RLB high pass (~38 Hz)
    b2 = (1.0, -2.0, 1.0)
    a2 = (1.0, -1.99004745483398, 0.99007225036621)
    return (b1, a1), (b2, a2)


def _lfilter(x: np.ndarray, b, a, sr: int) -> np.ndarray:
    """Frequency-domain application of a biquad (zero-phase magnitude)."""
    global _EQ_N
    n = x.shape[0]
    nf = _next_fast_len(n)
    _EQ_N = nf
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    z = np.exp(-2j * np.pi * f / sr)
    num = b[0] + b[1] * z + b[2] * z * z
    den = a[0] + a[1] * z + a[2] * z * z
    H = np.abs(num / den)
    return np.fft.irfft(np.fft.rfft(x, nf) * H, nf)[:n].astype(x.dtype, copy=False)


def _tp_gain_curve(seg: np.ndarray, sr: int, ceil: float, oversample: int) -> np.ndarray:
    """Per-sample gain that keeps a (short) segment under ``ceil`` at 1x.

    Split out of true_peak_limit so the caller can run it block by block.
    """
    hi = resample_fft(seg, sr, sr * oversample)
    mag = np.max(np.abs(hi), axis=1)
    del hi

    dec = 16
    usable = (mag.size // dec) * dec
    mag_d = mag[:usable].reshape(-1, dec).max(axis=1)

    win = max(3, int(sr * oversample * 0.002) // dec)  # ~2 ms at 4x rate
    env = _rolling_max(mag_d, win)

    need = np.minimum(1.0, ceil / np.maximum(env, 1e-12))
    need = _box_smooth(need, max(1, win // 2))

    gain_hi = np.repeat(need, dec)
    if gain_hi.size < mag.size:
        gain_hi = np.concatenate([gain_hi, np.full(mag.size - gain_hi.size, gain_hi[-1])])

    groups = (gain_hi.size // oversample) * oversample
    gain = gain_hi[:groups].reshape(-1, oversample).min(axis=1)
    if gain.size < seg.shape[0]:
        gain = np.concatenate([gain, np.full(seg.shape[0] - gain.size, gain[-1])])
    return gain[: seg.shape[0]]


def true_peak_limit(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    oversample: int = 4,
    block_s: float = 8.0,
) -> np.ndarray:
    """Limit so the *reconstructed* waveform stays under the ceiling.

    The naive version -- gain per sample of ``ceil/|x|``, then min-pool -- is
    wrong in a way that is easy to miss: a gain curve that dips for single
    samples is itself broadband, so the output rings between samples and the
    true peak ends up *higher* than before.

    So the envelope is built the way a real limiter does it: peak magnitude at
    4x the sample rate, smoothed with a forward-looking rolling maximum
    (~2 ms), turned into a slowly varying gain.

    Processing is **block-wise** with a quarter-second pad.  A whole 3-minute
    mix at 4x is ~550 MB per intermediate array and pushed peak RSS past 11 GB;
    eight-second blocks make the cost independent of song length.  Blocks are
    combined by taking the smaller gain, which is the safe direction and keeps
    the curve continuous.
    """
    # a small safety margin: two true-peak estimators never agree exactly
    ceil = db_to_lin(ceiling_db - 0.4)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return x[:, 0] if squeeze else x

    bs = max(1, int(block_s * sr))
    pad = min(n, int(0.25 * sr))
    gain = np.ones(n, dtype=np.float64)

    start = 0
    while start < n:
        lo = max(0, start - pad)
        hi = min(n, start + bs + pad)
        g = _tp_gain_curve(x[lo:hi], sr, ceil, oversample)
        np.minimum(gain[lo:hi], g, out=gain[lo:hi])
        start += bs

    y = x * gain.astype(x.dtype, copy=False)[:, None]
    return y[:, 0] if squeeze else y


def resample_fft(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or x.shape[0] < 1_000_000:
        return _resample_fft(x, sr, target)
    _memory.reclaim_unused()
    try:
        return _resample_fft(x, sr, target)
    finally:
        # Reclaim at the caller boundary after the inner return/unwind,
        # rather than keeping its normal FFT frame active during cleanup.
        _memory.reclaim_unused()


def _resample_fft(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    n = x.shape[0]
    m = int(round(n * target / sr))
    common = math.gcd(sr, target)
    numerator, denominator = target // common, sr // common
    # Both FFT grids must describe the same duration, even after padding.
    units = (max(n, m) + denominator - 1) // denominator
    nf = denominator * _next_fast_len(units)
    M = (nf // denominator) * numerator
    # All channels at once: one transform each way, and the transform runs
    # along the contiguous axis instead of on a strided column view.
    if x.ndim == 2:
        X = np.fft.rfft(x, nf, axis=0)
        Y = np.zeros((M // 2 + 1, x.shape[1]), dtype=X.dtype)
        k = min(len(X), len(Y))
        Y[:k] = X[:k]
        del X
        if n >= 1_000_000 and x.shape[1] == 2 and x.dtype == np.dtype("float64"):
            # Keep the original full inverse grid and final crop/scale. Each
            # strided column uses less simultaneous complex128 FFT workspace.
            full = np.empty((M, 2), dtype=Y.real.dtype)
            for channel in range(2):
                full[:, channel] = np.fft.irfft(Y[:, channel], M)
            return full[:m] * (M / nf)
        return np.fft.irfft(Y, M, axis=0)[:m] * (M / nf)
    X = np.fft.rfft(x, nf)
    Y = np.zeros(M // 2 + 1, dtype=X.dtype)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    del X
    return np.fft.irfft(Y, M)[:m] * (M / nf)


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness, BS.1770-4 with -70/-10 gating."""
    if x.ndim == 1:
        x = x[:, None]
    if sr != 48000:
        x = resample_fft(x, sr, 48000)
        sr = 48000
    y = _k_weight(x, sr)

    block = int(0.4 * sr)
    hop = int(0.1 * sr)
    if y.shape[0] < block:
        return -70.0
    n_blocks = 1 + (y.shape[0] - block) // hop

    # Block mean squares from a prefix sum, NOT by fancy-indexing the blocks.
    # y[idx] materialises (n_blocks, 19200, channels) -- on a 3-minute stereo
    # file that is ~550 MB *per call*, and this gets called per stem plus once
    # per loudness iteration.  The prefix sum costs one extra array of the
    # signal's own size and is exact.
    sq = y ** 2
    cs = np.zeros((sq.shape[0] + 1, sq.shape[1]), dtype=np.float64)
    np.cumsum(sq, axis=0, out=cs[1:])
    starts = np.arange(n_blocks) * hop
    sums = cs[starts + block] - cs[starts]
    # BS.1770 uses the *mean* square within each gating block, then sums that
    # over channels (sum G_i * z_i).  Two separate traps here:
    #   * the sample axis must be a mean, not a sum.  A 0.4 s block at 48 kHz
    #     is 19200 samples, so summing instead of averaging adds
    #     10*log10(19200) = +42.8 dB to every reading.
    #   * the channel axis must be a sum, not a mean.  Averaging would make a
    #     centre-panned source measure 3 dB quieter than the same source
    #     hard-panned, and would leave every mix 3 dB hot against -14 LUFS.
    # y[idx] is (blocks, samples, channels): the channel axis is LAST because
    # the advanced index consumes axis 0 and leaves the rest in place.  Getting
    # these two axes the wrong way round is silent -- it just reports numbers
    # that are tens of dB wrong.
    power = (sums / block).sum(axis=1)
    with np.errstate(divide="ignore"):
        block_lufs = -0.691 + 10.0 * np.log10(np.maximum(power, EPS))
    sel = block_lufs > -70.0
    if not np.any(sel):
        return -70.0
    rel = -0.691 + 10.0 * np.log10(np.mean(power[sel])) - 10.0
    sel2 = block_lufs > rel
    if not np.any(sel2):
        sel2 = sel
    return float(-0.691 + 10.0 * np.log10(np.mean(power[sel2])))


def momentary_lufs(x: np.ndarray, sr: int) -> float:
    """BS.1770 *momentary* loudness: K-weighted mean square, no gating.

    Integrated loudness needs 400 ms gating blocks, so a 150 ms one-shot has no
    valid blocks at all and ``loudness_lufs`` correctly returns -70.  That is
    right for a whole mix and wrong for a single short stem: the stem is not
    silent, it is just shorter than a block.  This is the measure to use there,
    and it is the same K-weighting with the gating removed.
    """
    if x.ndim == 1:
        x = x[:, None]
    if sr != 48000:
        x = resample_fft(x, sr, 48000)
        sr = 48000
    y = _k_weight(x, sr)
    power = float(np.mean(y ** 2, axis=0).sum())
    return float(-0.691 + 10.0 * np.log10(max(power, EPS)))


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 4, block_s: float = 8.0) -> float:
    """Inter-sample peak estimate via FFT oversampling.

    Measured in blocks for the same reason the limiter is: oversampling a
    three-minute stereo mix in one go needs a 34-million-point iFFT, which is
    ~550 MB per channel per call, and this meter runs once per stem plus once
    per loudness iteration.  Blocks with a little overlap keep the answer while
    making the cost independent of length.

    The context around a block is FFT scaffolding, not signal: the segment
    ends on an artificial cut, which rings, and the FFT pads the rest with
    zeros.  Taking the maximum over the whole upsampled array therefore
    charged the meter for that ringing -- it read the overshoot of its own
    cut instead of the audio's peak.  Each block is now only charged for the
    samples it owns; the reading is floored at the sample peak, accumulated
    from the owned samples of each block, so cropping can never report less
    than the input's own peak and no whole-file temporary is materialised.
    """
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return -120.0

    bs = max(1, int(block_s * sr))
    pad = min(n, int(0.1 * sr))
    peak = 0.0

    start = 0
    while start < n:
        lo = max(0, start - pad)
        hi = min(n, start + bs + pad)
        seg = x[lo:hi]
        nf = _next_fast_len(seg.shape[0])
        M = nf * oversample
        # Samples this block owns, as rows of ``seg`` and as indices into the
        # upsampled array.  ``up`` holds the reconstruction of ``seg`` sampled
        # at ``oversample`` points per sample, so the owned range starts
        # ``(start - lo)`` samples into the segment and ends at the block's
        # last owned sample.
        owned0 = start - lo
        owned1 = min(start + bs, n) - lo
        i0 = owned0 * oversample
        i1 = owned1 * oversample
        for c in range(seg.shape[1]):
            # Floor: this block's own samples, block by block and channel by
            # channel, so the meter never reads below the sample peak and no
            # temporary grows with the length of the whole signal.
            peak = max(peak, float(np.max(np.abs(seg[owned0:owned1, c]))))
            X = np.fft.rfft(seg[:, c], nf)
            Y = np.zeros(M // 2 + 1, dtype=X.dtype)
            k = min(len(X), len(Y))
            Y[:k] = X[:k]
            # With an even length the Nyquist bin is a singleton, not a pair.
            # Copied into the longer inverse it becomes an ordinary +/-
            # frequency pair and counts twice, so halve it once here.
            if nf % 2 == 0 and oversample > 1:
                Y[nf // 2] *= 0.5
            up = np.fft.irfft(Y, M) * oversample
            peak = max(peak, float(np.max(np.abs(up[i0:i1]))))
            del up, Y, X
        start += bs
    return lin_to_db(peak)


def short_term_db(
    x: np.ndarray, sr: int, win_s: float = 0.4, hop_s: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Short-term RMS level in dB, window by window, and the window centres.

    Used for level riding: a static gain cannot follow a quiet verse into a
    loud chorus, and that is the difference between "mixed" and "balanced".
    Prefix sums again -- the naive version materialises every window.
    """
    if x.ndim == 2:
        mono = x.mean(axis=1)
    else:
        mono = x
    n = mono.shape[0]
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if n < win:
        return np.array([n / (2.0 * sr)]), np.array([lin_to_db(float(np.sqrt(np.mean(mono ** 2))))])

    sq = (mono.astype(np.float64) ** 2)
    cs = np.zeros(sq.shape[0] + 1)
    np.cumsum(sq, out=cs[1:])
    starts = np.arange(0, n - win + 1, hop)
    means = (cs[starts + win] - cs[starts]) / win
    db = 10.0 * np.log10(np.maximum(means, EPS))
    times = (starts + win / 2.0) / sr
    return times, db


def crest_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x ** 2)) + EPS
    return lin_to_db(np.max(np.abs(x)) / rms)


def tail_ratio(x: np.ndarray, sr: int, max_onsets: int = 60) -> float:
    """How much energy survives after transients: a reverb-specific measure.

    For each detected onset, compare the energy in the 80-250 ms after it with
    the energy in its first 80 ms.  A dry percussive mix drops away (ratio near
    0.3); a wet one keeps ringing (0.5+).  Unlike stereo width or correlation,
    this does not move when the *source* is wide, which is why it is the feature
    used for matching a reference's wetness rather than guessing it.

    Returns 0.0 when the material has no transients to measure -- a sustained
    pad says nothing about reverb tails, and pretending otherwise would be
    worse than admitting it.
    """
    if x.ndim == 2:
        mono = x.mean(axis=1).astype(np.float64)
    else:
        mono = x.astype(np.float64)
    n = mono.shape[0]
    win = max(1, int(0.01 * sr))
    if n < 4 * win:
        return 0.0

    env = np.sqrt(np.convolve(mono ** 2, np.ones(win) / win, mode="same"))

    # Rise over 5 ms rather than a sample-to-sample difference: the absolute
    # threshold then means something (a real transient jumps by a large
    # fraction of the envelope maximum, a steady pad does not).  Comparing a
    # per-sample difference against an envelope-scale threshold, as the first
    # version did, silently rejected every real onset.
    lag = max(1, int(0.005 * sr))
    rise = np.zeros_like(env)
    rise[lag:] = env[lag:] - env[:-lag]

    thresh = max(float(np.percentile(rise, 99.5)), 0.15 * float(env.max()))
    if thresh <= 0:
        return 0.0
    guard = int(0.3 * sr)
    idx = np.arange(rise.size)
    peaks = np.flatnonzero((rise > thresh) & (idx > guard) & (idx < rise.size - guard))
    if peaks.size < 3:
        return 0.0

    b = int(0.08 * sr)
    c = int(0.25 * sr)
    step = max(1, peaks.size // max_onsets)
    ratios = []
    for p in peaks[::step]:
        e1 = float(np.sum(env[p : p + b] ** 2))
        e2 = float(np.sum(env[p + b : p + c] ** 2))
        if e1 > 1e-9:
            ratios.append(e2 / e1)
    return float(np.median(ratios)) if ratios else 0.0


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = min(x.shape[0], 1 << 18)
    seg = x[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    s = spec.sum() + EPS
    return float((spec * freqs).sum() / s)


def band_energy(
    x: np.ndarray,
    sr: int,
    edges=(20, 120, 400, 1500, 5000, 20000),
    exclude_hz: float = 0.0,
) -> np.ndarray:
    """Fractional energy per band (linear power, sums to ~1).

    ``exclude_hz`` drops a narrow line (a mains hum) from the measurement, so
    a steady 50 Hz tone cannot invent a low end that the music does not have.
    """
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = min(x.shape[0], 1 << 18)
    seg = x[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    if exclude_hz > 0:
        spec = spec.copy()
        spec[np.abs(freqs - exclude_hz) <= 3.0] = 0.0
    out = []
    total = spec.sum() + EPS
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(spec[m].sum() / total))
    return np.array(out)


def stereo_correlation(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[1] < 2:
        return 1.0
    a, b = x[:, 0], x[:, 1]
    d = np.sqrt(np.mean(a * a) * np.mean(b * b)) + EPS
    return float(np.mean(a * b) / d)


def width_ratio(x: np.ndarray) -> float:
    """Side/mid energy ratio (0 = mono, 1 = wide)."""
    if x.ndim != 2 or x.shape[1] < 2:
        return 0.0
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1])
    return float(np.sqrt(np.mean(s * s)) / (np.sqrt(np.mean(m * m)) + EPS))
