"""Analyse a reference track into the numbers the planner can aim at.

The idea is deliberately narrow: a reference does not get copied, it gets
*measured*.  If the user drops in a song they like, what transfers is its
loudness, its spectral shape, its dynamics and its width -- not its notes, not
its arrangement, and not a learned model of its mix decisions.

The spectral shape is stored as 31 third-octave bands, which is what
``schemas/mix_plan.schema.json`` asks for under ``style.reference.tone``.
"""

from __future__ import annotations

import os

import numpy as np

from . import analyze, dsp, sections as secmod

#: standard 1/3-octave band centres, 20 Hz .. 20 kHz (31 bands)
THIRD_OCTAVE_CENTERS = (
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
    16000.0, 20000.0,
)

#: the five regions the presets speak in, as (name, low_hz, high_hz)
REGIONS = (
    ("low", 20.0, 160.0),
    ("low_mid", 160.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("air", 6000.0, 20000.0),
)


class ReferenceError(RuntimeError):
    pass


TONE_METHOD = {
    "id": "stereo_power_hann_2s_v1",
    "channel_aggregation": "mean of channel powers, no mono fold",
    "window_s": 2.0,
    "hop_s": 0.5,
    "common_support_floor_db": 30.0,
    "boundary": "full-support zero-padded edge windows",
    "normalisation": "31-band zero-mean dB; correction/distance re-centred on fixed common support",
}


def spectrum_db(x: np.ndarray, sr: int, centers=THIRD_OCTAVE_CENTERS) -> list[float]:
    """Full-duration, equally covered short-time stereo power shape.

    Channels are squared before averaging.  Four overlapping Hann windows
    cover every sample, including the boundaries; a chorus near an edge must
    not count 20 dB less than the same chorus in the middle.  Only one window
    and its FFT are held at once, independent of song duration.
    """
    audio = np.asarray(x)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] == 0 or sr <= 0:
        raise ReferenceError("tone analysis requires audio channels and a positive sample rate")
    if len(audio) == 0 or not np.any(audio):
        return [0.0 for _ in centers]
    size = max(64, int(round(2.0 * sr)))
    hop = max(1, size // 4)
    window = np.hanning(size)
    power = np.zeros(size // 2 + 1, dtype=np.float64)
    for start in range(-size + hop, len(audio), hop):
        frame = np.zeros((size, audio.shape[1]), dtype=np.float64)
        lo, hi = max(0, start), min(len(audio), start + size)
        if hi <= lo:
            continue
        frame[lo-start:hi-start] = audio[lo:hi]
        if not np.isfinite(frame).all():
            raise ReferenceError("tone analysis requires finite audio")
        transformed = np.fft.rfft(frame * window[:, None], axis=0)
        power += np.mean(np.abs(transformed) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(size, 1.0 / sr)
    total = float(power.sum()) + 1e-30
    levels = []
    for center in centers:
        lo, hi = center * 2**(-1/6), center * 2**(1/6)
        mask = (frequencies >= lo) & (frequencies < hi)
        levels.append(10.0 * np.log10(max(float(power[mask].sum()) / total, 1e-12)))
    shape = np.asarray(levels)
    shape -= shape.mean()
    return [round(float(v), 2) for v in shape]


def common_support(ref_tone: list[float], base_tone: list[float], floor_db: float = 30.0,
                   reference_sr: int | None = None, base_sr: int | None = None) -> list[int]:
    """Choose reliable common bands once, before correction.

    The default30 dB floor rejects <0.1%-of-peak power evidence. Finite
    boundary windows leak into otherwise empty bands near -40 dB; allowing
    those bins to set a global shape offset creates false broad EQ moves.
    """
    a, b = np.asarray(ref_tone, dtype=float), np.asarray(base_tone, dtype=float)
    if a.shape != (31,) or b.shape != (31,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ReferenceError("reference tone must contain 31 finite bands")
    nyquist = min([rate/2 for rate in (reference_sr, base_sr) if rate is not None] or [float("inf")])
    return [i for i, center in enumerate(THIRD_OCTAVE_CENTERS)
            if a[i] >= a.max()-floor_db and b[i] >= b.max()-floor_db
            and center * 2**(1/6) <= nyquist]


def tone_distance(ref_tone: list[float], tone: list[float], indices: list[int]) -> float | None:
    """RMS dB of relative shape error on a FIXED common-band set; not quality."""
    if len(indices) < 2:
        return None
    delta = np.asarray(ref_tone, dtype=float)[indices] - np.asarray(tone, dtype=float)[indices]
    delta -= delta.mean()
    return round(float(np.sqrt(np.mean(delta**2))), 4)


def region_tilt(tone: list[float], centers=THIRD_OCTAVE_CENTERS) -> dict[str, float]:
    """Collapse a 31-band spectrum into the presets' five named regions."""
    out = {}
    for name, lo, hi in REGIONS:
        vals = [v for v, c in zip(tone, centers) if lo <= c < hi]
        out[name] = round(float(np.mean(vals)), 2) if vals else 0.0
    return out


def analyse(path: str, decode_policy: str = analyze.DEFAULT_DECODE_POLICY) -> dict:
    """Measure a reference file into a plan-ready target vector."""
    if not os.path.exists(path):
        raise ReferenceError(f"reference not found: {path}")
    audio, sr = analyze.load_stem_file(path, decode_policy)
    if audio.shape[0] < sr // 2:
        raise ReferenceError(f"{os.path.basename(path)}: shorter than 0.5 s")

    if not np.any(audio) or float(np.max(np.abs(audio))) < 1e-10:
        raise ReferenceError(f"{os.path.basename(path)}: reference is silent")
    mono = audio.mean(axis=1)
    tone = spectrum_db(audio, sr)

    # low-band correlation tells us whether the reference's bottom end is mono.
    # That is the one thing a wide-looking master can still be strict about,
    # and it is the only "should I collapse the bass" evidence we actually have.
    low = dsp.apply_eq(audio, sr, [("lowpass", 120.0, 0.0, 0.707)])
    low_corr = dsp.stereo_correlation(low)

    # Does the reference itself have sections, and how hard does it contrast?
    # This is what scales our own section treatment: a reference whose chorus
    # is only 1 dB above its verse should not make us widen by 4%.
    ref_sections = secmod.detect(audio, sr)
    contrast_db = 0.0
    width_contrast = 0.0
    if len(ref_sections) > 1:
        levels = [s["level_db"] for s in ref_sections]
        contrast_db = float(max(levels) - min(levels))
        widths = []
        for s in ref_sections:
            lo = int(s["start_s"] * sr)
            hi = min(audio.shape[0], int(s["end_s"] * sr))
            if hi - lo > sr // 2:
                widths.append(dsp.width_ratio(audio[lo:hi]))
        if len(widths) > 1:
            width_contrast = float(max(widths) - min(widths))

    return {
        "file": os.path.basename(path),
        "sections": ref_sections,
        "section_contrast_db": round(contrast_db, 2),
        "section_width_contrast": round(width_contrast, 3),
        "path": os.path.abspath(path),
        "sample_rate": sr,
        "duration_s": round(audio.shape[0] / sr, 3),
        "lufs": round(dsp.loudness_lufs(audio, sr), 2),
        "true_peak_db": round(dsp.true_peak_db(audio, sr), 2),
        "crest_db": round(dsp.crest_db(audio), 2),
        "width": round(dsp.width_ratio(audio), 3),
        "correlation": round(dsp.stereo_correlation(audio), 3),
        "low_correlation": round(low_corr, 3),
        "centroid_hz": round(dsp.spectral_centroid(mono, sr), 1),
        "tail_ratio": round(dsp.tail_ratio(audio, sr), 3),
        "tone": tone,
        "tone_method": dict(TONE_METHOD),
        "tone_tilt": region_tilt(tone),
    }


def corrective_tilt(
    ref_tone: list[float], base_tone: list[float], floor_db: float = 30.0,
    indices: list[int] | None = None,
) -> dict[str, float]:
    """Shape-only EQ difference on common supported bands.

    Each 31-band vector has its own mean, which includes missing/silent bands.
    Re-centre their difference on COMMON support: an extra 8 kHz component
    outside an 8 kHz reference's Nyquist must not ask for +5 dB at both 1 kHz
    and 2.5 kHz.  One common band carries no relative-shape information.
    """
    if indices is None:
        indices = common_support(ref_tone, base_tone, floor_db)
    result = {name: 0.0 for name, _, _ in REGIONS}
    if len(indices) < 2:
        return result
    difference = np.asarray(ref_tone, dtype=float) - np.asarray(base_tone, dtype=float)
    difference -= float(np.mean(difference[indices]))
    for name, lo, hi in REGIONS:
        values = [difference[i] for i in indices if lo <= THIRD_OCTAVE_CENTERS[i] < hi]
        result[name] = round(float(np.mean(values)), 2) if values else 0.0
    return result


def to_plan_block(ref: dict, blend: float) -> dict:
    """The ``style.reference`` object the schema defines."""
    return {
        "mode": "global",
        "file": ref.get("file"),
        "lufs": ref["lufs"],
        "true_peak_db": ref["true_peak_db"],
        "crest_db": ref["crest_db"],
        "width": ref["width"],
        "tone": ref["tone"],
        "tone_method": dict(ref.get("tone_method") or {"id": "unspecified_legacy"}),
        "section_contrast_db": ref.get("section_contrast_db"),
        "section_width_contrast": ref.get("section_width_contrast"),
        "notes": (
            f"{ref['file']} · blended {blend:.2f} · "
            f"corr {ref['correlation']} · low-band mono below 120 Hz: "
            f"{ref['low_correlation'] > 0.9}"
        ),
    }
