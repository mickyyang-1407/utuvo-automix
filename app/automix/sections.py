"""Find the sections of a song by its own dynamics.

No downbeat detection, no bar grid, no ML: a section boundary in a finished
mix almost always shows up as a sustained step in short-term loudness, and that
is measurable.  The result is used to vary *global* character (width, reverb
send) between verse and chorus -- the smallest change that makes a static mix
stop sounding static.

Deliberately conservative: sections shorter than a phrase are merged away, and
a song with no clear level contrast returns a single section, which means the
renderer does nothing.
"""

from __future__ import annotations

import numpy as np

from . import dsp

#: a section boundary must change the level by at least this much ...
MIN_JUMP_DB = 2.5
#: ... and the resulting sections must be at least this long
MIN_SECTION_S = 8.0
#: compare the level in windows this wide on either side of a candidate
HALF_WINDOW_S = 4.0
#: and skip this much either side of the candidate, so the windows do not
#: straddle the very boundary they are trying to detect (that halved every
#: measured step and hid anything below ~5 dB)
GUARD_S = 1.0
#: below this length there is not enough song to have sections
MIN_TOTAL_S = 30.0


def find_sections(mix: np.ndarray, sr: int) -> list[dict]:
    """Return ``[{"start_s", "end_s", "level_db"}, ...]`` covering the audio."""
    n = mix.shape[0]
    total_s = n / float(sr)
    if total_s < MIN_TOTAL_S:
        return []

    times, db = dsp.short_term_db(mix, sr, win_s=1.0, hop_s=0.25)
    if times.size < 12:
        return []

    # smooth a little so a single loud bar does not look like a section
    kernel = np.ones(8) / 8.0
    smoothed = np.convolve(np.pad(db, (4, 3), mode="edge"), kernel, mode="valid")[: db.size]

    half = int(HALF_WINDOW_S / 0.25)
    guard = int(GUARD_S / 0.25)
    candidates: list[tuple[float, float]] = []
    for i in range(half + guard, smoothed.size - half - guard):
        before = float(np.median(smoothed[i - half - guard : i - guard]))
        after = float(np.median(smoothed[i + guard : i + half + guard]))
        jump = after - before
        if abs(jump) >= MIN_JUMP_DB:
            candidates.append((float(times[i]), abs(jump)))

    # keep the strongest boundaries first, then drop any that would create a
    # section shorter than MIN_SECTION_S
    candidates.sort(key=lambda c: -c[1])
    bounds: list[float] = []
    for t, _ in candidates:
        if t < MIN_SECTION_S or total_s - t < MIN_SECTION_S:
            continue
        if all(abs(t - b) >= MIN_SECTION_S for b in bounds):
            bounds.append(t)
    bounds.sort()

    edges = [0.0] + bounds + [total_s]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = int(a * sr)
        hi = min(n, int(b * sr))
        if hi <= lo:
            continue
        seg = mix[lo:hi]
        level = dsp.lin_to_db(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
        out.append({"start_s": round(a, 2), "end_s": round(b, 2), "level_db": round(level, 2)})
    return out


def label(sections: list[dict]) -> list[dict]:
    """Name the quiet ones 'verse' and the loud ones 'chorus'.

    Only names, not meaning: if the song has three distinct levels the middle
    one is 'other'.  The renderer only uses the ordering, so a wrong name is
    cosmetic -- the quiet parts get the quiet treatment either way.
    """
    if not sections:
        return sections
    levels = [s["level_db"] for s in sections]
    if max(levels) - min(levels) < MIN_JUMP_DB:
        for s in sections:
            s["label"] = "other"
        return sections

    lo, hi = min(levels), max(levels)
    for s in sections:
        pos = (s["level_db"] - lo) / max(hi - lo, 1e-9)
        if pos >= 0.66:
            s["label"] = "chorus"
        elif pos <= 0.33:
            s["label"] = "verse"
        else:
            s["label"] = "other"
    return sections


def detect(mix: np.ndarray, sr: int) -> list[dict]:
    return label(find_sections(mix, sr))
