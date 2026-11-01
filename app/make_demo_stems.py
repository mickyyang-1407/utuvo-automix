"""Generate a tiny synthetic multi-track session so the mixer can be run
end to end with no external audio.

Nothing here is meant to sound good: each stem just needs a distinct role and
a plausible spectral fingerprint, so the analyser and the planner have
something real to react to.  100 BPM, four bars, A minor.

    python3 make_demo_stems.py --out demo_stems
"""

from __future__ import annotations

import argparse
import os
import wave

import numpy as np

SR = 48000
BPM = 100.0
BEAT = 60.0 / BPM
BARS = 4
BEATS_PER_BAR = 4
DURATION = BARS * BEATS_PER_BAR * BEAT
N = int(DURATION * SR)


def _t() -> np.ndarray:
    return np.arange(N) / SR


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    """Exponential percussive envelope of length n (samples)."""
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    return a * np.exp(-t / max(decay, 1e-4))


def _add(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= N:
        return
    j = min(N, i + sig.shape[0])
    buf[i:j] += gain * sig[: j - i]


def _sine(freq, n=None, sr=SR):
    n = N if n is None else n
    return np.sin(2 * np.pi * freq * np.arange(n) / sr)


def _saw(freq, n=None, sr=SR, harmonics=12):
    n = N if n is None else n
    t = np.arange(n) / sr
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        if freq * h > sr / 2:
            break
        out += np.sin(2 * np.pi * freq * h * t) / h
    return out / np.max(np.abs(out) + 1e-12)


def _stereo(x: np.ndarray, width: float = 0.0) -> np.ndarray:
    """Mono -> stereo using a short-delay side component.

    Delay-based widening (a few ms on the side signal only) keeps the two
    channels positively correlated, like a real close-miked source.  The
    obvious alternative -- adding and subtracting independent noise -- makes
    the channels anti-phase, which then shows up as a negative correlation in
    the mix meters and is a phase problem no mixer should have to fix.
    """
    if width <= 0.0:
        return np.stack([x, x], axis=1)
    k = max(1, int(0.004 * SR))  # 4 ms
    delayed = np.concatenate([np.zeros(k), x[:-k]])
    side = width * 0.35 * delayed
    return np.stack([x + side, x - side], axis=1)


# ------------------------------------------------------------------- voices

def kick() -> np.ndarray:
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    freq = 120.0 * np.exp(-t * 28.0) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * _env(n, 0.001, 0.16)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t * 400.0) * 0.25
    sig = body + click
    buf = np.zeros(N)
    for b in range(BARS * BEATS_PER_BAR):
        if b % 4 in (0, 2) or b % 4 == 2:
            _add(buf, sig, b * BEAT, 0.9)
    return _stereo(buf)


def snare() -> np.ndarray:
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n)
    # cheap bandpass: subtract a smoothed copy
    k = int(0.004 * SR)
    sm = np.convolve(noise, np.ones(k) / k, mode="same")
    noise = noise - sm
    tone = np.sin(2 * np.pi * 190.0 * t) * 0.5
    sig = (noise * 0.7 + tone) * _env(n, 0.001, 0.09)
    buf = np.zeros(N)
    for b in range(BARS * BEATS_PER_BAR):
        if b % 4 in (1, 3):
            _add(buf, sig, b * BEAT, 0.7)
    return _stereo(buf, width=0.15)


def hats() -> np.ndarray:
    n = int(0.09 * SR)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    # brighten: first difference acts as a highpass
    noise = np.diff(np.concatenate([[0.0], noise]))
    sig = noise * _env(n, 0.0005, 0.022) * 0.35
    buf = np.zeros(N)
    steps = BARS * BEATS_PER_BAR * 2
    for i in range(steps):
        at = i * BEAT / 2
        _add(buf, sig, at, 1.0 if i % 2 == 0 else 0.6)
    return _stereo(buf, width=0.25)


def bass() -> np.ndarray:
    buf = np.zeros(N)
    notes = [(0, 55.0), (2, 55.0), (4, 73.42), (6, 65.41)]
    pattern = [(b * BEAT * 2, f) for b, f in notes]
    for b in range(BARS):
        for beat, f in pattern:
            at = b * BEATS_PER_BAR * BEAT + beat
            n = int(BEAT * 1.6 * SR)
            sig = _saw(f, n, harmonics=8) * _env(n, 0.01, 0.35)
            _add(buf, sig, at, 0.6)
    return _stereo(buf)


def lead_vocal() -> np.ndarray:
    """A vowel-ish formant stack, so it lands in the presence band."""
    buf = np.zeros(N)
    melody = [440.0, 493.88, 523.25, 493.88]
    for b in range(BARS):
        for k, f in enumerate(melody):
            at = b * BEATS_PER_BAR * BEAT + k * BEAT
            n = int(BEAT * 0.9 * SR)
            t = np.arange(n) / SR
            sig = np.zeros(n)
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2), (5, 0.12)):
                sig += amp * np.sin(2 * np.pi * f * h * t)
            # formant lift around 2.5 kHz
            sig = sig + 0.3 * np.sin(2 * np.pi * 2500.0 * t) * np.sin(2 * np.pi * f * t)
            sig *= _env(n, 0.03, 0.45) / (np.max(np.abs(sig)) + 1e-12)
            _add(buf, sig, at, 0.5)
    return _stereo(buf, width=0.05)


def guitar() -> np.ndarray:
    """Plucked chords, mid-heavy, wide."""
    buf = np.zeros(N)
    chords = [(0, [220.0, 277.18, 329.63]), (2, [196.0, 246.94, 293.66])]
    for b in range(BARS):
        for beat, notes in chords:
            at = b * BEATS_PER_BAR * BEAT + beat * BEAT
            n = int(BEAT * 1.4 * SR)
            t = np.arange(n) / SR
            sig = np.zeros(n)
            for f in notes:
                for h in (1, 2, 3):
                    sig += np.sin(2 * np.pi * f * h * t) / h
            sig *= _env(n, 0.004, 0.30) / (np.max(np.abs(sig)) + 1e-12)
            _add(buf, sig, at, 0.35)
    return _stereo(buf, width=0.30)


def pad() -> np.ndarray:
    t = _t()
    chords = [220.0, 277.18, 329.63]
    sig = np.zeros(N)
    for f in chords:
        for detune in (-0.6, 0.0, 0.6):
            sig += np.sin(2 * np.pi * (f + detune) * t + detune)
    sig /= np.max(np.abs(sig)) + 1e-12
    slow = 0.6 + 0.4 * np.sin(2 * np.pi * 0.12 * t)
    return _stereo(sig * slow * 0.30, width=0.45)


# --------------------------------------------------------------------- main

VOICES = {
    "drums_kick.wav": kick,
    "drums_snare.wav": snare,
    "drums_hats.wav": hats,
    "bass.wav": bass,
    "vocal_lead.wav": lead_vocal,
    "guitar_elec.wav": guitar,
    "pad.wav": pad,
}


def write_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7
    data = np.clip(audio, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_stems")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name, fn in VOICES.items():
        audio = fn()
        path = os.path.join(args.out, name)
        write_wav(path, audio)
        print(f"wrote {path}  {audio.shape[0] / SR:.1f}s")


if __name__ == "__main__":
    main()
