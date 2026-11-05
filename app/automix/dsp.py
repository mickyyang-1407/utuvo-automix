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


def _box_trailing(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing moving average: each output is the mean of the last ``width``
    inputs (only the past), O(n) via cumulative sum, edge-padded at the start."""
    width = max(1, int(width))
    if width == 1:
        return x.copy()
    xp = np.pad(x, (width - 1, 0), mode="edge")
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


def side_shelf(x: np.ndarray, sr: int, gain_db: float, freq: float = 5000.0) -> np.ndarray:
    """Lift (or cut) only the side signal above ``freq``: a wider top end
    that leaves the mono fold-down -- the mid -- exactly as it was."""
    if x.ndim != 2 or x.shape[1] < 2 or abs(gain_db) < 1e-3:
        return x
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    side = apply_eq(side[:, None], sr, [("highshelf", freq, gain_db, 0.7)])[:, 0]
    return np.stack([mid + side, mid - side], axis=1).astype(x.dtype, copy=False)


def mono_below(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Collapse content below `hz` to mono (keeps the low end solid)."""
    if hz <= 0:
        return x
    mono = x.mean(axis=1, keepdims=True)
    low = apply_eq(mono, sr, [("lowpass", hz, 0.0, 0.707)])
    high = apply_eq(x, sr, [("highpass", hz, 0.0, 0.707)])
    return low + high


# ---------------------------------------------------------------- dynamics


#: R119: how the compressor's detector reads the signal.
#:   centred_box_v1  -- the attack and release windows are centred on the
#:                      sample (plans without ``compressor_law``).  The
#:                      detector sees half a window into the future: with a
#:                      30 ms attack a drum hit's first 5 ms are turned down
#:                      more than its body (10.6 against 7.0 dB on a synthetic
#:                      kick, measured in R119) and the bed is turned down
#:                      20-40 ms *before* the hit -- the opposite of what a
#:                      slow attack is for.
#:   trailing_box_v1 -- the same two windows, ending at the sample: the
#:                      detector reads only the past, as a real compressor
#:                      does, so a hit's start passes for about the attack
#:                      time and the body is held.  Steady-state level and
#:                      gain are the same as centred_box_v1.
COMPRESSOR_LAWS = ("centred_box_v1", "trailing_box_v1")


def _env(x: np.ndarray, sr: int, attack_ms: float, release_ms: float,
         law: str = "centred_box_v1") -> np.ndarray:
    """Smoothed magnitude envelope, attack/release approximated by box blend."""
    mag = np.sqrt(np.mean(x * x, axis=1)) if x.ndim == 2 else np.abs(x)
    a = max(1, int(sr * attack_ms / 1000.0))
    r = max(1, int(sr * release_ms / 1000.0))
    box = _box_trailing if law == "trailing_box_v1" else _box_smooth
    ea = box(mag, a)
    er = box(mag, r)
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
    law: str = "centred_box_v1",
) -> np.ndarray:
    """Soft-knee downward compressor. `amount` in 0..1 scales ratio & depth.

    ``law`` is the detector (see COMPRESSOR_LAWS)."""
    if law not in COMPRESSOR_LAWS:
        raise ValueError(f"unknown compressor law {law!r}")
    if amount <= 1e-3:
        return x
    env = _env(x, sr, attack_ms, release_ms, law=law)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))
    over = np.maximum(env_db - threshold_db, 0.0)
    ratio = 1.0 + (ratio_max - 1.0) * float(np.clip(amount, 0.0, 1.0))
    gain_db = -over * (1.0 - 1.0 / ratio)
    # soften the knee so it doesn't sound like a gate
    knee = 6.0
    soft = np.clip((env_db - (threshold_db - knee / 2)) / knee, 0.0, 1.0)
    gain_db = gain_db * (0.35 + 0.65 * soft)
    g = 10.0 ** (gain_db / 20.0)
    g = (_box_trailing if law == "trailing_box_v1" else _box_smooth)(g, max(1, int(sr * 0.004)))
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


#: R117: how ``saturate`` shapes the signal.
#:   peak_tanh_v1    -- the curve is normalised to the track's single largest
#:                      peak and runs at the sample rate (plans without
#:                      ``saturation_law``).  Below that peak the curve is a
#:                      gain of up to k/tanh(k) (+3.4 dB at drive 0.25): quiet
#:                      passages come up against the loud ones, so every
#:                      saturated track and bus is also a soft peak
#:                      compressor, and the harmonics above Nyquist fold back.
#:   env_tanh_os4_v1 -- the same tanh, normalised to the signal's own
#:                      envelope (a 30 ms peak hold) and run at 4x the sample
#:                      rate: the level contour between passages is kept
#:                      (within each cycle the waveform is shaped, as by any
#:                      saturator), every passage gets the drive's harmonics,
#:                      and harmonics above the original Nyquist are filtered
#:                      out instead of folding back (what still folds from
#:                      beyond 4x is tiny: -87 dB on a 15 kHz tone at 0.35).
#:                      Level: the RMS is kept.  A steady full-scale sine gets
#:                      the same harmonics as with peak_tanh_v1.
SATURATION_LAWS = ("peak_tanh_v1", "env_tanh_os4_v1")
SAT_ENV_HOLD_S = 0.030
SAT_OVERSAMPLE = 4


def saturate(x: np.ndarray, drive: float, law: str = "peak_tanh_v1", sr: int | None = None) -> np.ndarray:
    """Saturation with level compensation, drive 0..1, by ``law`` (SATURATION_LAWS)."""
    if law == "env_tanh_os4_v1":
        if sr is None:
            raise ValueError("env_tanh_os4_v1 needs the sample rate")
        return _saturate_env_os(x, drive, int(sr))
    if law != "peak_tanh_v1":
        raise ValueError(f"unknown saturation law {law!r}")
    return _saturate_peak(x, drive)


def _sat_envelope(mag: np.ndarray, sr: int) -> np.ndarray:
    """A smooth peak envelope of ``mag`` (1x rate): the largest value within
    SAT_ENV_HOLD_S around each sample (so it holds over a 33 Hz cycle and
    does not follow the waveform), smoothed over the same span."""
    n = mag.size
    dec = max(1, sr // 1000)  # 1 ms cells
    cells = (n + dec - 1) // dec
    padded = np.zeros(cells * dec, dtype=np.float64)
    padded[:n] = mag
    cell = padded.reshape(-1, dec).max(axis=1)
    span = max(3, int(round(SAT_ENV_HOLD_S * 1000)))
    env = _box_smooth(_rolling_max(cell, span), span)
    centres = (np.arange(cells) + 0.5) * dec
    return np.interp(np.arange(n), centres, env)


def _saturate_env_os(x: np.ndarray, drive: float, sr: int, block_s: float = 8.0) -> np.ndarray:
    """Law env_tanh_os4_v1 (see SATURATION_LAWS)."""
    if drive <= 1e-3:
        return x
    squeeze = x.ndim == 1
    xs = x[:, None] if squeeze else x
    n = xs.shape[0]
    if n == 0 or not np.any(xs):
        return x
    k = 5.0 * float(np.clip(drive, 0.0, 1.0))
    norm = float(np.tanh(k))
    env = _sat_envelope(np.max(np.abs(xs), axis=1).astype(np.float64), sr)
    os_ = SAT_OVERSAMPLE
    out = np.empty(xs.shape, dtype=np.float64)
    bs = max(1, int(block_s * sr))
    # FFT resampling rings at a segment's cut ends; a quarter second of
    # context on each side keeps that out of the part that is kept
    pad = min(n, int(0.25 * sr))
    start = 0
    while start < n:
        lo, hi = max(0, start - pad), min(n, start + bs + pad)
        end = min(n, start + bs)
        seg = xs[lo:hi].astype(np.float64)
        if not np.any(seg):
            out[start:end] = 0.0
            start += bs
            continue
        up = resample_fft(seg, sr, sr * os_)
        e = np.interp(np.arange(up.shape[0]) / os_, np.arange(seg.shape[0]), env[lo:hi])
        # never below the signal itself, so |u| <= 1 and no sample is pushed
        # past its own envelope
        np.maximum(e, np.max(np.abs(up), axis=1), out=e)
        np.maximum(e, 1e-9, out=e)
        y = np.tanh(k * up / e[:, None]) * (e[:, None] / norm)
        del up, e
        down = resample_fft(y, sr * os_, sr)
        del y
        out[start:end] = down[start - lo:end - lo]
        del down
        start += bs
    rin = float(np.sqrt(np.mean(xs.astype(np.float64) ** 2)))
    rout = float(np.sqrt(np.mean(out ** 2)))
    if rout > EPS:
        out *= rin / rout
    out = out.astype(x.dtype, copy=False)
    return out[:, 0] if squeeze else out


def _saturate_peak(x: np.ndarray, drive: float) -> np.ndarray:
    """tanh saturation with level compensation. drive 0..1 (peak_tanh_v1).

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


def make_ir_plate_v2(
    sr: int,
    decay_s: float = 1.5,
    predelay_ms: float = 20.0,
    high_cut_hz: float = 8000.0,
    seed: int = 7,
) -> np.ndarray:
    """A reverb that behaves more like a real space (law plate_v2).

    make_ir is one noise burst decaying at one rate: every frequency dies
    together, it starts at full density and it carries the source's low end
    into the tail -- grainy, flat, and it muddies the mix before it gives a
    voice any air.  Here:
      * the tail decays by band: lows (below 500 Hz) at 1.2x the decay time,
        mids at 1x, highs (above 4 kHz) at 0.55x, as air and surfaces absorb;
      * eight early reflections between 7 and 45 ms, alternating sides,
        give the space a size before the tail arrives;
      * the tail builds up over 15 ms instead of starting at full density;
      * everything below 200 Hz is taken out of the return, so the reverb can
        be heard without thickening the low end.
    Unit energy, like make_ir, so a send amount means the same thing.
    """
    rng = np.random.default_rng(seed)
    long_s = decay_s * 1.2
    n = max(1, int(long_s * sr))
    t = np.arange(n) / sr
    noise = rng.standard_normal((n, 2))
    spec = np.fft.rfft(noise, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)[:, None]
    tail = np.zeros((n, 2))
    for lo, hi, scale in ((0.0, 500.0, 1.2), (500.0, 4000.0, 1.0), (4000.0, sr / 2.0 + 1, 0.55)):
        band = np.fft.irfft(spec * ((f >= lo) & (f < hi)), n, axis=0)
        tail += band * np.exp(-6.9078 * t / max(decay_s * scale, 1e-3))[:, None]
    tail *= np.minimum(1.0, t / 0.015)[:, None]
    er = np.zeros((int(0.05 * sr) + 1, 2))
    for i, ms in enumerate(np.sort(rng.uniform(7.0, 45.0, 8))):
        er[int(ms * sr / 1000.0), i % 2] += 0.9 * np.exp(-ms / 40.0) * rng.choice((-1.0, 1.0))
    er_energy = float(np.sqrt(np.sum(er ** 2)))
    tail_energy = float(np.sqrt(np.sum(tail ** 2)))
    if er_energy > 0 and tail_energy > 0:
        er *= 0.35 * tail_energy / er_energy  # reflections ~9 dB under the tail
    m = max(n, er.shape[0])
    ir = np.zeros((m, 2))
    ir[:n] += tail
    ir[: er.shape[0]] += er
    ir = apply_eq(ir, sr, [("lowpass", min(high_cut_hz, sr * 0.45), 0.0, 0.707),
                           ("highpass", 200.0, 0.0, 0.707)])
    pre = int(max(0.0, predelay_ms) * sr / 1000.0)
    if pre > 0:
        ir = np.vstack([np.zeros((pre, 2)), ir])
    energy = float(np.sqrt(np.sum(ir ** 2)))
    if energy > 0:
        ir /= energy
    return ir


REVERB_IR_LAWS = ("noise_v1", "plate_v2")


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
    ir_law: str = "noise_v1",
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
    maker = make_ir_plate_v2 if ir_law == "plate_v2" else make_ir
    ir = maker(sr, decay_s, predelay_ms, high_cut_hz)
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


#: the release law's recovery and attack slopes (tp_release_v1).  Measured on
#: an 80 Hz tone with kick hits, 8 dB over the ceiling: the fast law adds harmonics at -47.9 dB (input -58.0); 60 dB/s
#: -53.5; 30 dB/s with 1 dB/ms attack -57.1, for 0.7 dB less loudness than
#: 60 dB/s.  Slower still costs loudness and starts to pump.
LIMITER_RELEASE_DB_PER_S = 30.0
LIMITER_ATTACK_DB_PER_MS = 1.0


def _tp_gain_curve_release(seg: np.ndarray, sr: int, ceil: float, oversample: int) -> np.ndarray:
    """Like _tp_gain_curve, but shaped like a mastering limiter.

    The fast curve lets go of a peak within ~2 ms, so on a loud mix the gain
    flutters at audio rate -- distortion, the "clipped" sound.  Here the
    attenuation needed at each moment (peaks at 4x) is held and released at
    LIMITER_RELEASE_DB_PER_S, and ramps in ahead of a peak at
    LIMITER_ATTACK_DB_PER_MS (look-ahead).  Both are running maxima of a
    sloped line in dB, so it stays vectorised; the attenuation is never less
    than what the peak needs, so the ceiling holds.
    """
    hi = resample_fft(seg, sr, sr * oversample)
    mag = np.max(np.abs(hi), axis=1)
    del hi
    dec = 16
    usable = (mag.size // dec) * dec
    if usable == 0:
        return np.ones(seg.shape[0])
    mag_d = mag[:usable].reshape(-1, dec).max(axis=1)
    rate = sr * oversample / dec  # control samples per second
    need_db = np.maximum(0.0, 20.0 * np.log10(np.maximum(mag_d, 1e-12) / ceil))
    idx = np.arange(need_db.size, dtype=np.float64)
    rel = LIMITER_RELEASE_DB_PER_S / rate
    held = np.maximum.accumulate(need_db + rel * idx) - rel * idx
    att = LIMITER_ATTACK_DB_PER_MS * 1000.0 / rate
    rev = held[::-1]
    ridx = np.arange(rev.size, dtype=np.float64)
    ramped = (np.maximum.accumulate(rev + att * ridx) - att * ridx)[::-1]
    smooth = np.maximum(_box_smooth(ramped, max(1, int(rate * 0.0005))), need_db)
    gain_c = 10.0 ** (-smooth / 20.0)
    gain_hi = np.repeat(gain_c, dec)
    if gain_hi.size < mag.size:
        gain_hi = np.concatenate([gain_hi, np.full(mag.size - gain_hi.size, gain_hi[-1])])
    groups = (gain_hi.size // oversample) * oversample
    gain = gain_hi[:groups].reshape(-1, oversample).min(axis=1)
    if gain.size < seg.shape[0]:
        gain = np.concatenate([gain, np.full(seg.shape[0] - gain.size, gain[-1])])
    return gain[: seg.shape[0]]


LIMITER_LAWS = ("tp_fast_v1", "tp_release_v1")


def true_peak_limit(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    oversample: int = 4,
    block_s: float = 8.0,
    law: str = "tp_fast_v1",
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
    # the release law holds attenuation for longer than the fast one: give
    # each block enough context that a release never stops at its edge
    pad = min(n, int((1.0 if law == "tp_release_v1" else 0.25) * sr))
    gain = np.ones(n, dtype=np.float64)

    start = 0
    while start < n:
        lo = max(0, start - pad)
        hi = min(n, start + bs + pad)
        curve = _tp_gain_curve_release if law == "tp_release_v1" else _tp_gain_curve
        g = curve(x[lo:hi], sr, ceil, oversample)
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


def windowed_lufs(x: np.ndarray, sr: int, win_s: float = 3.0, hop_s: float = 1.0) -> np.ndarray:
    """K-weighted loudness of each window (BS.1770 weighting, no gating), in LUFS.

    3 s windows are the "short-term" loudness of EBU R128.  Silence reads
    -inf as -120, so callers can threshold without special cases.
    """
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == 0:
        return np.array([-120.0])
    # the K-weighting coefficients are the 48 kHz ones (as in loudness_lufs)
    if sr != 48000:
        x = resample_fft(x, sr, 48000)
        sr = 48000
    n = x.shape[0]
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    k = _k_weight(x.astype(np.float64, copy=False), sr)
    power = np.sum(k * k, axis=1)
    del k
    cs = np.zeros(n + 1)
    np.cumsum(power, out=cs[1:])
    if n < win:
        means = np.array([cs[n] / n])
    else:
        starts = np.arange(0, n - win + 1, hop)
        means = (cs[starts + win] - cs[starts]) / win
    return -0.691 + 10.0 * np.log10(np.maximum(means, 1e-12))


def vocal_to_rest_db(
    vocal: np.ndarray, rest: np.ndarray, sr: int, active_db: float = 15.0
) -> tuple[float | None, int]:
    """How loud ``vocal`` sits against ``rest`` where the vocal is singing.

    Both are cut into 3 s windows; windows where the vocal is within
    ``active_db`` of its own loudest window (and the rest is not silent) are
    kept, and the median of vocal minus rest over them is returned with the
    number of windows used.  A long intro or solo without singing therefore
    does not make the vocal look quiet.  (None, 0) when the vocal never sings.
    """
    n = min(len(vocal), len(rest))
    lv = windowed_lufs(vocal[:n], sr)
    lr = windowed_lufs(rest[:n], sr)
    active = (lv > float(np.max(lv)) - active_db) & (lv > -50.0) & (lr > -60.0)
    if not active.any():
        return None, 0
    return float(np.median(lv[active] - lr[active])), int(active.sum())


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


# ------------------------------------------------- sidechain, parallel, delay


def key_level_db(key: np.ndarray, sr: int, attack_ms: float, trailing: bool = False) -> np.ndarray:
    """The key's own short-window level, with no hold: what it really does.

    ``trailing`` (R119, voice_duck) averages the last ``attack_ms`` only; the
    default centred window (the kick duck, plans before R119) sees half of
    it ahead."""
    mag = np.sqrt(np.mean(key * key, axis=1)) if key.ndim == 2 else np.abs(key)
    level = (_box_trailing if trailing else _box_smooth)(mag * mag, max(1, int(sr * attack_ms / 1000.0)))
    return 10.0 * np.log10(np.maximum(level, EPS * EPS))


def key_envelope_db(key: np.ndarray, sr: int, attack_ms: float, release_ms: float,
                    level_db: np.ndarray | None = None) -> np.ndarray:
    """A detector that rises at once and falls ``20 dB`` per ``release_ms``.

    The compressor's box envelope is symmetric, so a duck built on it would
    start dipping the bass half a release *before* the kick hits.  This one
    is causal on the way down: a peak hold with a straight-line release in
    dB, which is the running maximum of ``level + slope*n`` minus the slope.
    """
    if level_db is None:
        level_db = key_level_db(key, sr, attack_ms)
    slope = 20.0 / max(1.0, sr * release_ms / 1000.0)
    ramp = slope * np.arange(level_db.shape[0], dtype=np.float64)
    return np.maximum.accumulate(level_db + ramp) - ramp


#: how far a key's hits must stand above its resting level to be followed
DUCK_MIN_HIT_DB = 6.0

#: R119 voice_duck_v1: the duck is full while the voice is at or above its
#: own singing level (the median where it is present) and opens over the
#: VOICE_DUCK_RANGE_DB below that, so quiet breaths and gaps leave the
#: return open
VOICE_DUCK_RANGE_DB = 10.0
VOICE_DUCK_PRESENT_DB = 40.0


def voice_duck(x: np.ndarray, key: np.ndarray, sr: int, depth_db: float,
               attack_ms: float = 20.0, release_ms: float = 250.0) -> tuple[np.ndarray, dict]:
    """Turn a reverb return down while the voice that feeds it sings (R119).

    The detector is the key's level over the last 10 ms (a trailing window:
    it reads only what was sung) with a causal hold that falls 20 dB per
    ``release_ms`` (key_envelope_db).  Full depth at or above the voice's
    own singing level -- the median of its level where it is present (within
    40 dB of its loudest) -- and none ``VOICE_DUCK_RANGE_DB`` below it, so the
    tail blooms in the gaps.  The gain moves in over ``attack_ms`` (a trailing
    average: it only reacts to what was sung).  Returns (signal, receipt).
    """
    n = x.shape[0]
    if depth_db <= 0.0 or n == 0 or not key.any():
        return x, {"skipped": "nothing to duck"}
    level_db = key_level_db(key, sr, 10.0, trailing=True)
    env_db = key_envelope_db(key, sr, 10.0, release_ms, level_db=level_db)
    peak_db = float(np.max(level_db))
    present = level_db[level_db > peak_db - VOICE_DUCK_PRESENT_DB]
    sing_db = float(np.median(present))
    amount = np.clip((env_db - (sing_db - VOICE_DUCK_RANGE_DB)) / VOICE_DUCK_RANGE_DB, 0.0, 1.0)
    gain_db = _box_trailing(-float(depth_db) * amount, max(1, int(sr * attack_ms / 1000.0)))
    g = (10.0 ** (gain_db / 20.0)).astype(x.dtype, copy=False)
    out = x * g[:, None] if x.ndim == 2 else x * g
    singing = level_db > sing_db - 3.0
    gaps = level_db < sing_db - 30.0
    return out, {"singing_db": round(sing_db, 2),
                 "gain_while_singing_db": round(float(np.median(gain_db[singing])), 2) if singing.any() else None,
                 "gain_in_gaps_db": round(float(np.median(gain_db[gaps])), 2) if gaps.any() else None}


def duck(
    x: np.ndarray,
    key: np.ndarray,
    sr: int,
    depth_db: float,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
    follow_hits: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``x`` down by up to ``depth_db`` while ``key`` hits.

    The dip scales with how hard the key hits: its loudest hits (the 99th
    percentile of its level where it is present) get the full depth, and its
    resting level gets none, so a kick track with rumble or bleed under it
    does not hold the bass down between hits.  The resting level is the
    median of the key's own level without the detector's hold (silence
    counted as 60 dB down): the hold's release tails would raise it on a
    dense kick pattern and leave real hits under-ducked.
    A key with less than ``DUCK_MIN_HIT_DB`` between the two has no hits to
    follow and leaves ``x`` alone.  With ``follow_hits=False`` the key's
    presence counts, not its hits (a sustained voice ducking its own delay):
    no resting level is taken off.  Returns (signal, gain curve).
    """
    level_db = key_level_db(key, sr, attack_ms)
    env_db = key_envelope_db(key, sr, attack_ms, release_ms, level_db=level_db)
    peak_db = float(np.max(level_db))
    present = level_db[level_db > peak_db - 60.0]
    if depth_db <= 0.0 or present.size == 0:
        return x, np.ones(x.shape[0])
    ref_db = float(np.percentile(present, 99.0))
    rel = 10.0 ** ((env_db - ref_db) / 20.0)
    if follow_hits:
        floor_db = float(np.percentile(np.maximum(level_db, peak_db - 60.0), 50.0))
        if ref_db - floor_db < DUCK_MIN_HIT_DB:
            return x, np.ones(x.shape[0])
        floor = 10.0 ** ((floor_db - ref_db) / 20.0)
        amount = np.clip((rel - floor) / (1.0 - floor), 0.0, 1.0)
    else:
        amount = np.clip(rel, 0.0, 1.0)
    gain = 1.0 - (1.0 - db_to_lin(-float(depth_db))) * amount
    g = gain.astype(x.dtype, copy=False)
    return (x * g[:, None] if x.ndim == 2 else x * g), gain


def parallel_compress(
    x: np.ndarray,
    sr: int,
    blend: float,
    ratio: float = 8.0,
    threshold_offset_db: float = -10.0,
    attack_ms: float = 2.0,
    release_ms: float = 80.0,
    level_from: str = "whole",
    law: str = "centred_box_v1",
) -> np.ndarray:
    """New-York compression: a crushed copy under the dry signal.

    The copy is compressed hard and brought back to the dry level, then mixed
    in at ``blend`` of that level.  The sum is returned at the dry signal's
    RMS, so the bus keeps its place in the balance and only its shape changes:
    quiet detail and sustain come up against the transients.
    """
    if blend <= 1e-3:
        return x
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= EPS:
        return x
    # "active": the threshold follows the level where the source plays (50 ms
    # windows within 30 dB of its loudest), so a vocal with long gaps is not
    # compressed harder than the same phrases sung back to back
    ref = active_rms(x, sr) if level_from == "active" else rms
    crushed = compress(x, sr, 1.0, threshold_db=lin_to_db(ref) + threshold_offset_db,
                       ratio_max=ratio, attack_ms=attack_ms, release_ms=release_ms,
                       makeup_mode="none", law=law)
    c_rms = float(np.sqrt(np.mean(crushed.astype(np.float64) ** 2)))
    if c_rms <= EPS:
        return x
    y = x + crushed * (blend * rms / c_rms)
    y_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return (y * (rms / y_rms)).astype(x.dtype, copy=False)


def active_rms(x: np.ndarray, sr: int, window_s: float = 0.05, range_db: float = 30.0) -> float:
    """RMS over the 50 ms windows within ``range_db`` of the loudest one."""
    w = max(1, int(window_s * sr))
    n = (x.shape[0] // w) * w
    if n == 0:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    xx = x[:n].astype(np.float64)
    p = np.mean(xx.reshape(-1, w, *x.shape[1:]) ** 2, axis=tuple(range(1, xx.ndim + 1)))
    top = float(np.max(p))
    if top <= EPS * EPS:
        return 0.0
    keep = p >= top * 10.0 ** (-range_db / 10.0)
    return float(np.sqrt(np.mean(p[keep])))


def onset_envelope(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Spectral flux every 10 ms: how much new energy each frame brings."""
    mono = np.mean(x, axis=1) if x.ndim == 2 else x
    hop = max(1, int(round(sr * 0.01)))
    win = 1 << int(math.ceil(math.log2(sr * 0.04)))
    frames = 1 + (mono.shape[0] - win) // hop if mono.shape[0] >= win else 0
    if frames < 2:
        return np.zeros(0), sr / hop
    window = np.hanning(win)
    out = np.empty(frames)
    prev = None
    step = 2048  # frames per block keeps the STFT's memory small on long songs
    view = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop]
    for start in range(0, frames, step):
        block = view[start:start + step] * window
        mag = np.log1p(100.0 * np.abs(np.fft.rfft(block, axis=1)))
        if prev is None:
            diff = np.vstack([np.zeros((1, mag.shape[1])), np.diff(mag, axis=0)])
        else:
            diff = np.diff(np.vstack([prev, mag]), axis=0)
        out[start:start + mag.shape[0]] = np.maximum(diff, 0.0).sum(axis=1)
        prev = mag[-1:]
    return out, sr / hop


#: the tempo prior: a log-normal around 120 BPM, one octave wide.  It only
#: decides between metrical levels that fit the onsets about equally well.
TEMPO_PRIOR_BPM = 120.0
TEMPO_RANGE_BPM = (60.0, 200.0)
#: tempi a beat is easily mistaken for, other than double and half
TEMPO_RIVAL_RATIOS = (2.0 / 3.0, 1.5, 0.75, 4.0 / 3.0)
#: a rival scoring this close to the answer makes the tempo unclear
TEMPO_RIVAL_SHARE = 0.9


def _tempo_scores(env: np.ndarray, fps: float, bpms: np.ndarray) -> np.ndarray:
    e = env - _box_smooth(env, max(3, int(fps)))  # local mean off: keep the pulses
    e = np.maximum(e, 0.0)
    n = e.shape[0]
    spec = np.fft.rfft(e, _next_fast_len(2 * n))
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    if acf[0] <= EPS:
        return np.zeros_like(bpms)
    acf = acf / acf[0]
    lags = np.arange(n, dtype=np.float64)
    score = np.zeros_like(bpms)
    # the beat and its multiples, plus its half: a real beat is usually
    # subdivided, while a period of three eighth-notes (which the multiples
    # alone cannot tell from the beat) has nothing at its half
    for k, w in ((0.5, 0.5), (1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
        score += w * np.interp(k * 60.0 * fps / bpms, lags, acf, right=0.0)
    return score / 2.58


def estimate_tempo(x: np.ndarray, sr: int) -> dict | None:
    """Beats per minute from the onsets, or None when there is no steady pulse.

    Autocorrelation of the onset envelope, scored as a comb (the beat and its
    multiples), on a 0.1 BPM grid.  The answer has to hold in both halves of
    the song on its own: a tempo that only one half agrees with is not one a
    delay should be locked to.
    """
    env, fps = onset_envelope(x, sr)
    if env.shape[0] < fps * 8:
        return None
    bpms = np.arange(TEMPO_RANGE_BPM[0], TEMPO_RANGE_BPM[1] + 1e-9, 0.1)
    prior = np.exp(-0.5 * np.log2(bpms / TEMPO_PRIOR_BPM) ** 2)
    whole = _tempo_scores(env, fps, bpms)
    best = int(np.argmax(whole * prior))
    bpm = float(bpms[best])
    strength = float(whole[best])
    # A pulse at 2:3 or 3:4 of the answer that fits almost as well is a
    # different beat a delay could just as well lock to (a dotted or triplet
    # feel): that is not a clear tempo, whatever the strength says.
    rival = 0.0
    for r in TEMPO_RIVAL_RATIOS:
        lo, hi = bpm * r * 0.98, bpm * r * 1.02
        near = (bpms >= lo) & (bpms <= hi)
        if near.any():
            rival = max(rival, float(np.max((whole * prior)[near])))
    ambiguous = rival >= TEMPO_RIVAL_SHARE * float(whole[best] * prior[best])
    half = env.shape[0] // 2
    agree = 0
    for part in (env[:half], env[half:]):
        if part.shape[0] < fps * 4:
            continue
        s = _tempo_scores(part, fps, bpms)
        b = float(bpms[int(np.argmax(s * prior))])
        # the same pulse, or the same pulse counted at double/half speed
        if min(abs(b / bpm - r) for r in (0.5, 1.0, 2.0)) <= 0.02:
            agree += 1
    return {"bpm": round(bpm, 1), "strength": round(strength, 3), "halves_agree": agree,
            "ambiguous": bool(ambiguous)}


def delay_send(
    dry: np.ndarray,
    sr: int,
    time_ms: float,
    feedback: float,
    hpf_hz: float = 300.0,
    lpf_hz: float = 5000.0,
) -> np.ndarray:
    """A delay *send*: the echoes only, band-limited like a real return.

    Repeats at ``time_ms``, each ``feedback`` times the one before, until
    they are 40 dB down.
    """
    d = int(round(sr * time_ms / 1000.0))
    out = np.zeros_like(dry)
    if d <= 0 or d >= dry.shape[0]:
        return out
    gain, k = 1.0, 1
    while gain >= 0.01 and k * d < dry.shape[0]:
        out[k * d:] += dry[:-k * d] * gain
        gain *= float(feedback)
        k += 1
        if feedback <= 0.0:
            break
    return apply_eq(out, sr, [("highpass", hpf_hz, 0.0, 0.707),
                              ("lowpass", min(lpf_hz, sr * 0.45), 0.0, 0.707)])


# ------------------------------------------------- R121 vocal doubler (ADT)

#: R121: the lead vocal's artificial double (ADT: the tape trick behind a
#: "thick" lead without a real double -- two short copies whose delay drifts
#: slowly, so each one's pitch wanders a few cents sharp and flat, hard left
#: and right, low under the voice).  A drifting delay *is* a pitch change
#: (ratio 1 - d'(t)), with no grains and no splices; a short-grain shifter
#: was tried first and resets the phase every grain, so a 7-cent shift never
#: builds up (it measures as a 40 Hz flutter, no pitch change).  Plans
#: without a track's ``double`` block are not doubled.
DOUBLE_LAWS = ("adt_v1",)


def drifting_delay(x: np.ndarray, sr: int, delay_ms: float, cents: float, rate_hz: float,
                   phase: float = 0.0) -> np.ndarray:
    """``x`` read ``delay_ms`` late, the delay swinging sinusoidally at
    ``rate_hz`` just enough that the pitch moves up to ``cents`` either way.

    d(t) = d0 + A sin(2 pi f t + phase), pitch ratio 1 - d'(t), so
    A = (1 - 2**(-cents/1200)) / (2 pi f).  Same length, deterministic,
    linear interpolation, nothing read before the start.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    dev = 1.0 - 2.0 ** (-abs(float(cents)) / 1200.0)
    amp = dev / (2.0 * np.pi * float(rate_hz)) * sr          # samples
    t = np.arange(n, dtype=np.float64)
    d = delay_ms * 1e-3 * sr + amp * np.sin(2.0 * np.pi * rate_hz * t / sr + phase)
    pos = t - d
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 + 1 < n)
    out = np.zeros(n)
    out[ok] = x[i0[ok]] * (1.0 - frac[ok]) + x[i0[ok] + 1] * frac[ok]
    return out


def vocal_double(x: np.ndarray, sr: int, spec: dict) -> np.ndarray:
    """The stereo double of a (stereo) lead vocal, to be added to it.

    Left ``delays_ms[0]`` late, right ``delays_ms[1]`` late; each drifts by
    up to ``cents`` at its own rate (``rates_hz``) and in opposite phase, so
    when one side is sharp the other is flat.  Each side at ``level_db``
    under the voice, band-limited to ``hpf_hz``..``lpf_hz`` (no doubled lows,
    no doubled sibilance).
    """
    law = spec.get("law")
    if law not in DOUBLE_LAWS:
        raise ValueError(f"unknown double law {law!r}")
    m = np.asarray(x, dtype=np.float64).mean(axis=1)
    cents = float(spec["cents"])
    dl, dr = (float(v) for v in spec["delays_ms"])
    rl, rr = (float(v) for v in spec["rates_hz"])
    g = db_to_lin(float(spec["level_db"]))
    y = np.stack([drifting_delay(m, sr, dl, cents, rl, 0.0),
                  drifting_delay(m, sr, dr, cents, rr, np.pi)], 1) * g
    return apply_eq(y, sr, [("highpass", float(spec.get("hpf_hz", 180.0)), 0.0, 0.707),
                            ("lowpass", min(float(spec.get("lpf_hz", 9000.0)), sr * 0.45), 0.0, 0.707)])


# ------------------------------------------- R122 dynamic resonance suppression

#: R122: narrow, moving resonances on the lead vocal (harshness 2-8 kHz,
#: honk around 1 kHz) are pulled down where they stick out of the voice's own
#: spectrum, frame by frame -- a de-esser for every band.  It smooths peaks;
#: it does not tilt the broadband brightness (that decision is the owner's).
#: Plans without a track's ``resonance`` block are not touched.
RESONANCE_LAWS = ("res_suppress_v1",)


def _bin_windows(freqs: np.ndarray, octaves: float, df: float) -> tuple[np.ndarray, np.ndarray]:
    """Per bin, the [lo, hi) bin range of a window ``octaves`` wide around it."""
    k = np.arange(freqs.size)
    half = 2.0 ** (octaves / 2.0)
    lo = np.floor(np.maximum(freqs / half / df, 0)).astype(np.int64)
    hi = np.ceil(freqs * half / df).astype(np.int64) + 1
    lo = np.minimum(lo, k)
    hi = np.maximum(np.minimum(hi, freqs.size), k + 1)
    return lo, hi


def _window_mean(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Mean of each frame's values over every bin's window (cumulative sums)."""
    c = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    return (c[:, hi] - c[:, lo]) / (hi - lo)


def resonance_weight(freqs: np.ndarray) -> np.ndarray:
    """Where the detector listens: 0 below 800 Hz, up to 1 at 2 kHz, 1 to
    8 kHz, down to 0.5 at 12 kHz and 0 at 16 kHz."""
    lf = np.log2(np.maximum(freqs, 1.0))
    return np.interp(lf, np.log2([800.0, 2000.0, 8000.0, 12000.0, 16000.0]), [0.0, 1.0, 1.0, 0.5, 0.0],
                     left=0.0, right=0.0)


def resonance_suppress(x: np.ndarray, sr: int, spec: dict) -> tuple[np.ndarray, dict]:
    """Pull narrow resonances down, then hold the whole song to ``max_mean_cut_db``:
    if the average cut over 2-8 kHz came out deeper, ratio and max_db are
    both scaled by cap/cut and the pass is run once more.  Every step of the
    gain (the excess times the ratio, the max_db ceiling, the smoothing over
    frequency, the attack/release) is linear in that common scale, so every
    cut -- and the average -- shrinks by exactly that factor.  A smoother that
    removes more than the cap changes the voice, not its peaks."""
    y, rec = _resonance_pass(x, sr, spec)
    cap = spec.get("max_mean_cut_db")
    raw = rec.pop("_mean_cut_2k_8k_raw")
    if cap is not None and raw > float(cap) > 0.0:
        scale = float(cap) / raw
        ratio = float(spec.get("ratio", 0.6)) * scale
        gmax = float(spec.get("max_db", 6.0)) * scale
        y, rec2 = _resonance_pass(x, sr, {**spec, "ratio": ratio, "max_db": gmax})
        rec2.pop("_mean_cut_2k_8k_raw")
        rec = {**rec2, "capped_from_db": rec["mean_cut_2k_8k_db"], "ratio_used": round(ratio, 3),
               "max_db_used": round(gmax, 2)}
    return y, rec


def _resonance_pass(x: np.ndarray, sr: int, spec: dict) -> tuple[np.ndarray, dict]:
    """Pull narrow resonances of a (stereo) track down, linked across channels.

    Per frame (STFT, ~43 ms, hop a quarter of that): the power in a
    ``detect_oct`` window around each bin against the power in a
    ``ref_oct`` window with the ``notch_oct`` around the bin left out; what sticks out more than ``threshold_db`` is cut by
    ``ratio`` of the excess times the listening weight, at most ``max_db``,
    smoothed over a sixth of an octave, attack ``attack_ms`` / release
    ``release_ms``; frames more than 40 dB under the track's loudest are left
    alone.  Returns (audio, receipt).
    """
    if spec.get("law") not in RESONANCE_LAWS:
        raise ValueError(f"unknown resonance law {spec.get('law')!r}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    N = 2048 if sr <= 50000 else 4096
    hop = N // 4
    win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N))   # periodic sqrt-Hann
    pad = N
    xp = np.concatenate([np.zeros((pad, ch)), x, np.zeros((pad + hop, ch))])
    frames = (xp.shape[0] - N) // hop + 1
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    df = freqs[1]
    dlo, dhi = _bin_windows(freqs, float(spec.get("detect_oct", 1 / 6)), df)
    rlo, rhi = _bin_windows(freqs, float(spec.get("ref_oct", 1.5)), df)
    slo, shi = _bin_windows(freqs, 1 / 6, df)
    nlo, nhi = _bin_windows(freqs, float(spec.get("notch_oct", 1 / 3)), df)
    w = resonance_weight(freqs)[None, :]
    theta = float(spec.get("threshold_db", 4.0))
    ratio = float(spec.get("ratio", 0.6))
    gmax = float(spec.get("max_db", 6.0))
    a_att = 1.0 - math.exp(-hop / (float(spec.get("attack_ms", 5.0)) * 1e-3 * sr))
    a_rel = 1.0 - math.exp(-hop / (float(spec.get("release_ms", 80.0)) * 1e-3 * sr))
    # the gate: frame levels against the loudest frame
    mono = xp.mean(axis=1)
    idx = np.arange(frames) * hop
    lev = np.array([np.mean(mono[i: i + N] ** 2) for i in idx])
    lev_db = 10.0 * np.log10(lev + 1e-20)
    live = lev_db > lev_db.max() - 40.0
    out = np.zeros_like(xp)
    g_prev = np.zeros(freqs.size)
    cut_sum = np.zeros(freqs.size)
    cut_frames = 0
    peak_cut = []
    chunk = 1024
    for c0 in range(0, frames, chunk):
        c1 = min(frames, c0 + chunk)
        seg = np.stack([xp[i: i + N] for i in idx[c0:c1]])          # (f, N, ch)
        X = np.fft.rfft(seg * win[None, :, None], axis=1)            # (f, bins, ch)
        P = np.mean(np.abs(X) ** 2, axis=2) + 1e-20                  # linked: one detector
        # both sides are power means: a harmonic spectrum's valleys must not
        # pull the reference down (a dB mean sits in the valleys, and then
        # every harmonic "sticks out")
        det = 10.0 * np.log10(np.maximum(_window_mean(P, dlo, dhi), 1e-20))
        # ... and the reference leaves out the neighbourhood it judges, or a
        # resonance would lift its own yardstick
        cP = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
        outer = cP[:, rhi] - cP[:, rlo]
        inner = cP[:, nhi] - cP[:, nlo]
        cnt = np.maximum((rhi - rlo) - (nhi - nlo), 1)
        ref = 10.0 * np.log10(np.maximum((outer - inner) / cnt, 1e-20))
        excess = np.maximum(det - ref - theta, 0.0)
        g = -np.minimum(ratio * w * excess, gmax)
        g = _window_mean(g, slo, shi)
        g[~live[c0:c1]] = 0.0
        sm = np.empty_like(g)
        for j in range(g.shape[0]):
            a = np.where(g[j] < g_prev, a_att, a_rel)
            g_prev = g_prev + (g[j] - g_prev) * a
            sm[j] = g_prev
        lv = live[c0:c1]
        if lv.any():
            cut_sum += -sm[lv].sum(axis=0)
            cut_frames += int(lv.sum())
            peak_cut.append(-sm[lv].min(axis=1))
        Y = X * (10.0 ** (sm / 20.0))[:, :, None]
        y = np.fft.irfft(Y, N, axis=1) * win[None, :, None]
        for j, i in enumerate(idx[c0:c1]):
            out[i: i + N] += y[j]
    y = out[pad: pad + n] / 2.0          # sqrt-Hann^2 = Hann; four of them overlap to 2
    mean_cut = cut_sum / max(cut_frames, 1)
    band = (freqs >= 2000.0) & (freqs <= 8000.0)
    pk = np.concatenate(peak_cut) if peak_cut else np.zeros(1)
    receipt = {"mean_cut_2k_8k_db": round(float(mean_cut[band].mean()), 2),
               "_mean_cut_2k_8k_raw": float(mean_cut[band].mean()),
               "deepest_mean_cut_db": round(float(mean_cut.max()), 2),
               "deepest_mean_cut_hz": round(float(freqs[int(np.argmax(mean_cut))]), 0),
               "p95_frame_peak_cut_db": round(float(np.percentile(pk, 95)), 2)}
    return (y if ch > 1 else y[:, 0]), receipt


# ------------------------------------------- R121 band-selective kick duck
#: R121: under the kick only the bass's low band dips.  The bass bus is split
#: at a crossover just above the kick's fundamental with zero-phase
#: Linkwitz-Riley magnitudes (low |H| = 1 / (1 + (f/fc)^4), high = 1 - low),
#: so the two bands add back to the input exactly: out = x - (1 - g) * low.
#: Where the gain is 1 the bus is untouched to the bit; the bass's
#: articulation above the crossover never moves.
KICK_DUCK_BAND_LAWS = ("kick_duck_band_v1",)
CROSSOVER_PAD_S = 1.0           # zeros either side: the low-pass rings, the FFT must not wrap
KICK_F0_BAND_HZ = (40.0, 90.0)  # where a kick's fundamental lives
KICK_CROSSOVER_RATIO = 1.6      # crossover = 1.6 x the kick's fundamental ...
KICK_CROSSOVER_RANGE_HZ = (90.0, 130.0)   # ... held to 90-130 Hz (published bands: 40-100 Hz,
                                          # the highest about 131 Hz)
KICK_CROSSOVER_DEFAULT_HZ = 120.0


def crossover_lowpass(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    """The low band of a zero-phase LR4-magnitude crossover at ``fc``.

    ``x - crossover_lowpass(x)`` is the high band; the two sum to ``x``.
    """
    n = x.shape[0]
    if n == 0:
        return x.copy()
    pad = int(round(CROSSOVER_PAD_S * sr))
    nf = _next_fast_len(n + 2 * pad)
    shape = (nf,) + x.shape[1:]
    buf = np.zeros(shape, dtype=np.float64)
    buf[pad: pad + n] = x
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    H = 1.0 / (1.0 + (f / float(fc)) ** 4)
    X = np.fft.rfft(buf, nf, axis=0)
    y = np.fft.irfft(X * (H[:, None] if x.ndim == 2 else H), nf, axis=0)
    return y[pad: pad + n].astype(x.dtype, copy=False)


def kick_fundamental_hz(x: np.ndarray, sr: int) -> float | None:
    """The long-term spectral peak of a kick (or a whole kit) in 40-90 Hz.

    ``None`` when the key has nothing there (silent, or no low end)."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    m = np.asarray(m, dtype=np.float64)
    seg = int(sr // 2)                       # 2 Hz bins, Hann, half overlap
    if m.size < seg:
        return None
    win = np.hanning(seg)
    hop = seg // 2
    P = np.zeros(seg // 2 + 1)
    for s in range(0, m.size - seg + 1, hop):
        P += np.abs(np.fft.rfft(m[s: s + seg] * win)) ** 2
    f = np.fft.rfftfreq(seg, 1.0 / sr)
    lo, hi = KICK_F0_BAND_HZ
    band = (f >= lo) & (f <= hi)
    if not band.any() or P[band].max() <= 1e-12 * max(float(P.max()), 1e-30) or P.max() <= 0.0:
        return None
    return float(f[band][int(np.argmax(P[band]))])


def kick_crossover_hz(f0: float | None) -> float:
    """1.6 x the kick's fundamental, held to 90-130 Hz (120 Hz without one)."""
    if f0 is None:
        return KICK_CROSSOVER_DEFAULT_HZ
    lo, hi = KICK_CROSSOVER_RANGE_HZ
    return float(min(max(KICK_CROSSOVER_RATIO * f0, lo), hi))
