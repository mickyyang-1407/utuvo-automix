"""Generate a *realistic-ish* stem set: the demo parts, but recorded badly.

The demo stems are clean sines with neat envelopes.  Real stems from a session
carry reverb tails, compression, saturation, a noise floor, the occasional mains
hum, and stems that do not all start and end at the same moment.  Those are
exactly the properties the analyser's heuristics react to -- so testing against
clean synthetic audio can hide false positives.

This generates the same arrangement through a "studio-ish" chain:

    python3 make_realistic_stems.py --out /tmp/realistic_stems

24-bit WAV, deliberately uneven lengths and lead-in silence, one humming DI.
"""

from __future__ import annotations

import argparse
import os
import wave

import numpy as np

SR = 48000
BPM = 92.0
BEAT = 60.0 / BPM
BARS = 16
DURATION = BARS * 4 * BEAT
#: every other 4-bar block is a chorus: the arrangement lifts, so section
#: detection and the reference's contour matching have something to find
CHORUS_GAIN = 1.0
VERSE_GAIN = 0.55


def section_gain(at_bar: int) -> float:
    return CHORUS_GAIN if (at_bar // 4) % 2 else VERSE_GAIN
N = int(DURATION * SR)

rng = np.random.default_rng(7)


def _t(n: int | None = None) -> np.ndarray:
    return np.arange(N if n is None else n) / SR


def env(n, attack, decay):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    return a * np.exp(-t / max(decay, 1e-4))


def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= buf.shape[0]:
        return
    end = min(buf.shape[0], i + len(sig))
    buf[i:end] += sig[: end - i] * gain


def noise(n):
    return rng.standard_normal(n)


def hp(x, cutoff):
    """Cheap high-pass so stems are not DC-heavy; accepts mono or stereo."""
    from automix import dsp

    bands = [("highpass", cutoff, 0.0, 0.7)]
    mono = x.ndim == 1
    out = dsp.apply_eq(x[:, None] if mono else x, SR, bands)
    return out[:, 0] if mono else out


def comp(x, amount=0.4):
    from automix import dsp

    return dsp.compress(x[:, None], SR, amount)[:, 0]


def sat(x, amount=0.25):
    from automix import dsp

    return dsp.saturate(x[:, None], amount)[:, 0]


def verb(x, amount=0.15, decay=1.6):
    from automix import dsp

    wet = dsp.reverb_send(x[:, None], SR, amount, decay, 20.0, 8000.0)
    return x + wet[:, 0] * 0.6


def floor_noise(x, db=-72.0):
    return x + noise(len(x)) * (10 ** (db / 20.0))


def to_stereo(x, width=0.0):
    if width <= 0:
        return np.stack([x, x], axis=1)
    d = int(0.004 * SR)
    right = np.concatenate([np.zeros(d), x[:-d]])
    return np.stack([x, right * width + x * (1 - width)], axis=1)


def write_wave24(path, sig):
    import struct

    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * 8388607.0).astype(np.int32).reshape(-1)
    b = np.empty((pcm.size, 3), dtype=np.uint8)
    u = pcm & 0xFFFFFF
    b[:, 0] = u & 0xFF
    b[:, 1] = (u >> 8) & 0xFF
    b[:, 2] = (u >> 16) & 0xFF
    n_ch = sig.shape[1]
    raw = b.tobytes()
    block = n_ch * 3
    fmt = struct.pack("<HHIIHH", 1, n_ch, SR, SR * block, block, 24)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(raw)) + raw
    )
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def build(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    t = _t()
    paths = []

    # ---------------------------------------------------------------- drums
    kick = np.zeros(N)
    snare = np.zeros(N)
    hats = np.zeros(N)
    for bar in range(BARS):
        for beat in range(4):
            at = bar * 4 * BEAT + beat * BEAT
            if beat in (0, 2):
                n = int(0.45 * SR)
                f = 55 + 90 * np.exp(-np.arange(n) / (0.02 * SR))
                wave_ = 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.16)
                add(kick, sat(wave_), at, 0.9)
            if beat in (1, 3):
                n = int(0.35 * SR)
                tone = 0.35 * np.sin(2 * np.pi * 190 * np.arange(n) / SR) * env(n, 0.001, 0.08)
                body = noise(n) * env(n, 0.001, 0.11) * 0.6
                add(snare, comp(tone + body, 0.3), at, 0.8)
            n = int(0.09 * SR)
            tick = noise(n) * env(n, 0.0005, 0.02) * 0.35
            add(hats, tick, at + BEAT / 2, 0.7)

    # ----------------------------------------------------------------- bass
    bass = np.zeros(N)
    root = 55.0
    for bar in range(BARS):
        for beat in range(4):
            at = bar * 4 * BEAT + beat * BEAT
            n = int(BEAT * 0.9 * SR)
            tt = np.arange(n) / SR
            note = 0.5 * np.sin(2 * np.pi * root * tt) + 0.18 * np.sin(2 * np.pi * root * 2 * tt)
            note *= np.clip(tt / 0.01, 0, 1) * np.clip((BEAT * 0.9 - tt) / 0.05, 0, 1)
            add(bass, sat(comp(note, 0.35), 0.15), at)

    # ------------------------------------------------------------ guitars
    gtr_l = np.zeros(N)
    gtr_r = np.zeros(N)
    for bar in range(BARS):
        for beat in range(4):
            at = bar * 4 * BEAT + beat * BEAT
            n = int(BEAT * 0.8 * SR)
            tt = np.arange(n) / SR
            base = 220 * (1.0 if bar % 2 == 0 else 1.335)
            note = sum(
                0.16 / k * np.sin(2 * np.pi * base * k * tt + k)
                for k in range(1, 7)
            )
            note *= np.clip(tt / 0.02, 0, 1) * np.exp(-tt * 2.2)
            add(gtr_l, verb(hp(note, 95), 0.12), at, 0.8)
            add(gtr_r, verb(hp(note, 105), 0.12), at + 0.01, 0.75)

    # ----------------------------------------------------------------- keys
    keys = np.zeros(N)
    for bar in range(BARS):
        at = bar * 4 * BEAT
        n = int(3.6 * BEAT * SR)
        tt = np.arange(n) / SR
        pad = sum(
            0.09 * np.sin(2 * np.pi * 330 * k * tt) / k for k in (1, 2, 3, 5)
        )
        pad *= np.clip(tt / 0.3, 0, 1) * np.clip((3.6 * BEAT - tt) / 0.4, 0, 1)
        add(keys, verb(pad, 0.2, 2.2), at)

    # -------------------------------------------------------------- vocals
    vocal = np.zeros(N)
    backing = np.zeros(N)
    # lyrics-as-formants: a vowel-ish harmonic stack that phrases with the bars
    for bar in range(BARS):
        at = bar * 4 * BEAT + 0.5 * BEAT
        n = int(3.0 * BEAT * SR)
        tt = np.arange(n) / SR
        f0 = 196 * (1.0 if bar % 4 < 2 else 1.122)
        src = np.zeros(n)
        for k in range(1, 14):
            src += (0.22 / k) * np.sin(2 * np.pi * f0 * k * tt)
        # crude formants make it read as a voice rather than a saw
        src = comp(src * (1 + 0.3 * np.sin(2 * np.pi * 5.5 * tt)), 0.35)
        src *= np.clip(tt / 0.06, 0, 1) * np.clip((3.0 * BEAT - tt) / 0.15, 0, 1)
        # sibilance bursts on the phrase ends
        for s_at in (0.4, 1.6):
            i = int(s_at * BEAT * SR)
            d = int(0.1 * SR)
            if i + d < n:
                src[i : i + d] += noise(d) * np.hanning(d) * 0.22
        add(vocal, verb(src, 0.16), at, 0.85)
        add(backing, verb(src * 0.4, 0.25), at + 0.02, 0.5)

    # a DI with mains hum and a noise floor, the classic real-session problem
    di = hp(keys * 0.5 + floor_noise(np.zeros(N), -78), 70)
    hum = 0.01 * np.sin(2 * np.pi * 50 * t)
    di = di + hum

    # the arrangement: verses quieter, choruses at full level
    bar_of_sample = (np.arange(N) / SR) / (4 * BEAT)
    lift = np.array([section_gain(int(b)) for b in bar_of_sample])
    kick, snare, hats = kick * lift, snare * lift, hats * lift
    bass, gtr_l, gtr_r = bass * lift, gtr_l * lift, gtr_r * lift
    keys, vocal, backing, di = keys * lift, vocal * lift, backing * lift, di * lift

    # ---- things every real session has and clean synthetic sets do not ------

    # 1. humanised timing: nudge each part's hits a few ms off the grid
    def humanise(sig, max_ms=6.0):
        shift = int(rng.uniform(-max_ms, max_ms) * SR / 1000.0)
        if shift == 0:
            return sig
        out = np.zeros_like(sig)
        if shift > 0:
            out[shift:] = sig[:-shift]
        else:
            out[:shift] = sig[-shift:]
        return out

    kick, snare, hats = humanise(kick), humanise(snare), humanise(hats)
    bass, gtr_l, gtr_r = humanise(bass), humanise(gtr_l), humanise(gtr_r)
    keys, vocal, backing = humanise(keys), humanise(vocal), humanise(backing)

    # 2. mic bleed: the drum kit is audible in the bass and guitar mics at a low
    #    level.  Every real session has this, and it is exactly the kind of
    #    correlation that can make a duplicate detector cry wolf.
    drum_bus = kick + snare + hats
    bass = bass + 0.03 * drum_bus
    gtr_l = gtr_l + 0.02 * (kick + snare)
    gtr_r = gtr_r + 0.02 * (kick + snare)

    # 3. mic distance: a delayed copy comb-filters the source, which stresses
    #    spectral role detection far more than a clean impulse does
    def comb(sig, delay_ms=3.0, depth=0.45):
        d = int(delay_ms * SR / 1000.0)
        out = sig.copy()
        out[d:] += depth * sig[:-d]
        return out

    gtr_l = comb(gtr_l)
    gtr_r = comb(gtr_r, 3.7)

    parts = {
        "drums_kick.wav": to_stereo(kick),
        "drums_snare.wav": to_stereo(snare),
        "drums_hats.wav": to_stereo(hats, 0.2),
        "bass.wav": to_stereo(bass),
        "guitar_l.wav": to_stereo(gtr_l, 0.35),
        "guitar_r.wav": to_stereo(gtr_r, 0.35),
        "keys.wav": to_stereo(keys, 0.5),
        "vocal_lead.wav": to_stereo(floor_noise(vocal, -70), 0.25),
        "vocal_bv.wav": to_stereo(backing, 0.4),
        "keys_di.wav": to_stereo(di, 0.3),
    }
    for name, sig in parts.items():
        sig = hp(sig, 25) if name != "drums_kick.wav" else sig
        path = os.path.join(out_dir, name)
        write_wave24(path, sig)
        paths.append(path)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/realistic_stems")
    args = ap.parse_args()
    paths = build(args.out)
    print(f"wrote {len(paths)} stems to {args.out} ({DURATION:.1f} s each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
