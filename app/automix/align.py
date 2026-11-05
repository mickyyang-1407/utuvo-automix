"""Drum microphones lined up in time and in polarity (R120, law ``gccphat_v1``).

Several microphones on one kit hear every hit a few milliseconds apart: the
kick's inside and outside mic, the snare's top and bottom mic, the close mics
and the overheads above them.  Summed, the late copies comb-filter against the
early ones (a hollow kick, a thin snare, a smeared image), and a mic that
faces the other way (snare bottom) cancels the low end outright.

What this module does, and only when the evidence is clear:

* kick mics with each other, snare mics with each other: the later ones are
  moved earlier onto the earliest one (with three or more mics on one drum,
  onto the earliest one still within 10 ms of the latest -- a mic earlier
  than that would need a delay and keeps its time), and a mic whose
  polarity is reversed is flipped;
* the overheads are moved earlier onto the snare's close mic, keeping the
  close mic a few samples ahead (a small natural lead -- fully aligned kits
  sound like triggered samples);
* the snare and the kick are flipped if they are reversed against the
  overheads (the overheads are the kit's polarity reference).

What it never does:

* room mics, hi-hat and cymbal close mics are left alone (room mics carry
  the depth); tom mics are only checked for polarity against the overheads,
  never moved in time (they are mostly bleed between their hits);
* a pair that does not agree with itself -- the lag measured on four groups
  of hits disagrees, the correlation peak does not stand out, the lag is
  longer than any kit -- is left exactly as recorded, and the plan says why;
* a polarity is only flipped when the correlation's sign *and* the low-band
  energy of the sum both say so;
* a declared capture group (the user's own statement) is never touched.

Estimator: GCC-PHAT (Knapp & Carter 1976) on frames that start just before
the hits of the instrument in its own band, summed over up to 64 hits, read
at 1/16 sample.  The plan stores, per moved track, ``align = {law,
advance_ms, polarity}``; the renderer applies exactly that.  A plan without
the field renders as it always did.
"""

from __future__ import annotations

import math

import numpy as np

ALIGN_LAW = "gccphat_v1"
MODES = ("full", "polarity", "off")

#: inter-mic delays on a kit are under 10 ms; a longer window locks onto the
#: repeating pattern instead of the mic distance
MAX_LAG_MS = 10.0
#: interpolation of the correlation (1/16 sample)
SUBSAMPLE = 16
MIN_ONSETS = 8
MAX_ONSETS = 64
ONSET_SPACING_S = 0.25
#: confidence gate: the peak must stand out from everything outside its lobe
PSR_MIN = 2.0
#: the lags measured on four interleaved groups of hits must agree
AGREE_MS = 0.1
#: smaller than this is "already aligned": nothing is moved
ALREADY_ALIGNED_MS = 0.05
#: the close mic keeps this much lead over the overheads (3 samples at 48 kHz)
CLOSE_LEAD_MS = 0.0625
#: the low band of the sum must be this much bigger the other way round
POLARITY_MARGIN_DB = 1.0

#: per instrument: the band the pair shares, the band the polarity is judged
#: in, and the frame length in seconds
#: hits are found in a narrower band of the instrument's own (the kick's sub,
#: the snare's crack), so the other drum's bleed does not count as a hit
SOURCES = {
    "kick": {"band": (40.0, 2000.0), "onset": (40.0, 120.0), "confirm": (40.0, 250.0), "frame_s": 0.17},
    "snare": {"band": (150.0, 8000.0), "onset": (800.0, 6000.0), "confirm": (120.0, 500.0), "frame_s": 0.085},
    "toms": {"band": (60.0, 3000.0), "onset": (60.0, 400.0), "confirm": (60.0, 300.0), "frame_s": 0.17},
}


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 inside the band, raised-cosine skirts a third of an octave wide."""
    m = np.zeros_like(freqs)
    lo_in, hi_in = lo * 2 ** (1 / 6), hi * 2 ** (-1 / 6)
    lo_out, hi_out = lo * 2 ** (-1 / 6), hi * 2 ** (1 / 6)
    m[(freqs >= lo_in) & (freqs <= hi_in)] = 1.0
    up = (freqs > lo_out) & (freqs < lo_in)
    m[up] = 0.5 - 0.5 * np.cos(np.pi * np.log(freqs[up] / lo_out) / np.log(lo_in / lo_out))
    dn = (freqs > hi_in) & (freqs < hi_out)
    m[dn] = 0.5 + 0.5 * np.cos(np.pi * np.log(freqs[dn] / hi_in) / np.log(hi_out / hi_in))
    return m


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band filter of a whole mono signal (analysis only)."""
    n = x.shape[0]
    X = np.fft.rfft(x)
    X *= _band_mask(np.fft.rfftfreq(n, 1.0 / sr), lo, hi)
    return np.fft.irfft(X, n)


def onsets(xb: np.ndarray, sr: int, max_onsets: int = MAX_ONSETS) -> np.ndarray:
    """Sample positions of hits in a band-limited signal.

    5 ms energy frames; a hit is a frame at least 9 dB above the loudest of
    the 30 ms before it and within 25 dB of the loudest frame.  Hits closer
    than ``ONSET_SPACING_S`` keep the louder one; more than ``max_onsets``
    keep an even spread over the whole song.
    """
    hop = max(1, int(round(0.005 * sr)))
    nfr = xb.shape[0] // hop
    if nfr < 8:
        return np.zeros(0, dtype=np.int64)
    e = (xb[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1)
    e_db = 10.0 * np.log10(e + 1e-20)
    top = float(e_db.max())
    if top < -100.0:
        return np.zeros(0, dtype=np.int64)
    look = 6
    past = np.full(nfr, -200.0)
    for k in range(1, look + 1):
        past[k:] = np.maximum(past[k:], e_db[:-k])
    cand = np.flatnonzero((e_db - past >= 9.0) & (e_db >= top - 25.0))
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    # one hit per spacing window: the loudest wins
    spacing = max(1, int(round(ONSET_SPACING_S * sr / hop)))
    order = cand[np.argsort(-e_db[cand], kind="stable")]
    taken: list[int] = []
    blocked = np.zeros(nfr, dtype=bool)
    for f in order:
        if blocked[f]:
            continue
        taken.append(int(f))
        blocked[max(0, f - spacing): f + spacing + 1] = True
    taken.sort()
    if len(taken) > max_onsets:
        pick = np.unique(np.round(np.linspace(0, len(taken) - 1, max_onsets)).astype(int))
        taken = [taken[i] for i in pick]
    return np.asarray(taken, dtype=np.int64) * hop


def source_hits(x: np.ndarray, sr: int, source: str) -> np.ndarray:
    """The hits of one instrument: hits in its band that also rise in its own
    onset band (the kick's sub, the snare's crack) -- a snare hit bleeding
    into the kick mic rises in the kick's band but not in its sub."""
    x = _mono(x)
    lo, hi = SOURCES[source]["band"]
    cand = onsets(bandpass(x, sr, lo, hi), sr, max_onsets=4 * MAX_ONSETS)
    if cand.size == 0:
        return cand
    olo, ohi = SOURCES[source]["onset"]
    xo = bandpass(x, sr, olo, ohi) ** 2
    after = int(0.06 * sr)
    b0, b1 = int(0.12 * sr), int(0.04 * sr)
    keep = []
    for h in cand:
        if h - b0 < 0 or h + after > xo.shape[0]:
            continue
        e_after = float(xo[h: h + after].mean())
        e_before = float(xo[h - b0: h - b1].mean())
        if e_after >= 4.0 * e_before and e_after > 0.0:
            keep.append(h)
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size > MAX_ONSETS:
        keep = keep[np.unique(np.round(np.linspace(0, keep.size - 1, MAX_ONSETS)).astype(int))]
    return keep


def _frame_len(sr: int, seconds: float) -> int:
    return int(2 ** round(math.log2(max(256.0, seconds * sr))))


def _frames(ref: np.ndarray, mem: np.ndarray, hits: np.ndarray, n: int):
    """Hann frames with the hit a quarter of the way in, for both signals."""
    win = np.hanning(n)
    R, X = [], []
    for h in hits:
        s = int(h) - n // 4
        if s < 0 or s + n > ref.shape[0] or s + n > mem.shape[0]:
            continue
        R.append(np.fft.rfft(ref[s: s + n] * win))
        X.append(np.fft.rfft(mem[s: s + n] * win))
    if not R:
        return None, None
    return np.asarray(R), np.asarray(X)


def _peak(P: np.ndarray, n: int, sr: int, lobe_ms: float) -> tuple[float, float, float]:
    """(lag in samples, signed peak value, peak-to-sidelobe ratio)."""
    r = np.fft.irfft(P, n * SUBSAMPLE)
    maxlag = int(round(MAX_LAG_MS * 1e-3 * sr * SUBSAMPLE))
    idx = np.concatenate([np.arange(0, maxlag + 1), np.arange(r.shape[0] - maxlag, r.shape[0])])
    lags = np.where(idx <= maxlag, idx, idx - r.shape[0]).astype(np.float64)
    vals = r[idx]
    k = int(np.argmax(np.abs(vals)))
    lag = lags[k]
    peak = float(vals[k])
    lobe = lobe_ms * 1e-3 * sr * SUBSAMPLE
    outside = np.abs(lags - lag) > lobe
    side = float(np.max(np.abs(vals[outside]))) if np.any(outside) else 0.0
    if abs(peak) <= 0.0:
        psr = 0.0
    else:
        psr = abs(peak) / side if side > 0 else float("inf")
    return lag / SUBSAMPLE, peak, psr


def estimate(ref: np.ndarray, mem: np.ndarray, sr: int, source: str,
             hits: np.ndarray | None = None) -> dict:
    """Lag (ms, positive = ``mem`` arrives later) and polarity of ``mem``
    against ``ref`` in the instrument's band, with the evidence behind it.

    ``ok`` is False when the evidence is not clear; ``reason`` says why.
    """
    spec = SOURCES[source]
    lo, hi = spec["band"]
    ref = _mono(ref)
    mem = _mono(mem)
    m = min(ref.shape[0], mem.shape[0])
    ref, mem = ref[:m], mem[:m]
    if hits is None:
        hits = source_hits(ref, sr, source)
    out = {"source": source, "onsets": int(len(hits)), "ok": False}
    if len(hits) < MIN_ONSETS:
        out["reason"] = f"only {len(hits)} clear hits (need {MIN_ONSETS})"
        return out
    n = _frame_len(sr, spec["frame_s"])
    R, X = _frames(ref, mem, hits, n)
    if R is None or R.shape[0] < MIN_ONSETS:
        out["reason"] = "too few complete frames"
        return out
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = _band_mask(freqs, lo, hi)
    # a member that is silent where the reference hits has nothing to
    # correlate: refuse it (0 / 0 must not read as an infinitely clear peak)
    e_ref = float(np.sum(np.abs(R) ** 2 * mask))
    e_mem = float(np.sum(np.abs(X) ** 2 * mask))
    if e_ref <= 0.0 or e_mem <= 1e-6 * e_ref:
        out["reason"] = "silent where the reference hits (more than 60 dB under it)"
        return out
    G = np.conj(R) * X
    mag = np.abs(G)
    floor = 1e-12 * max(float(mag.max()), 1e-30)
    P = G / (mag + floor) * mask
    lobe_ms = max(0.5, 1.5e3 / (hi - lo))
    lag, peak, psr = _peak(P.sum(axis=0), n, sr, lobe_ms)
    groups = []
    for g in range(4):
        sel = P[g::4]
        if sel.shape[0] >= 2:
            groups.append(_peak(sel.sum(axis=0), n, sr, lobe_ms)[0])
    spread_ms = (max(groups) - min(groups)) * 1e3 / sr if len(groups) >= 2 else float("inf")
    lag_ms = lag * 1e3 / sr
    # polarity, judged on the low band of the sum with the lag taken out
    clo, chi = spec["confirm"]
    cb = (freqs >= clo) & (freqs <= chi)
    ramp = np.exp(2j * np.pi * freqs[cb] * (lag / sr))  # moves mem earlier by lag
    Xs = X[:, cb] * ramp
    e_pos = float(np.sum(np.abs(R[:, cb] + Xs) ** 2))
    e_neg = float(np.sum(np.abs(R[:, cb] - Xs) ** 2))
    flip_db = 10.0 * math.log10((e_neg + 1e-30) / (e_pos + 1e-30))
    out.update({
        "lag_ms": round(lag_ms, 4),
        "cm": round(lag_ms * 34.3, 1),
        "psr": round(psr, 2) if math.isfinite(psr) else 99.0,
        "spread_ms": round(spread_ms, 4) if math.isfinite(spread_ms) else 99.0,
        "sign": 1 if peak >= 0 else -1,
        "flip_gain_db": round(flip_db, 2),
        "frames": int(R.shape[0]),
    })
    if psr < PSR_MIN:
        out["reason"] = f"correlation peak does not stand out (ratio {psr:.2f} < {PSR_MIN})"
        return out
    if spread_ms > AGREE_MS:
        out["reason"] = f"hits disagree about the lag (spread {spread_ms:.3f} ms > {AGREE_MS} ms)"
        return out
    if abs(lag_ms) >= MAX_LAG_MS - 0.05:
        out["reason"] = "lag at the edge of the search window"
        return out
    out["ok"] = True
    if out["sign"] < 0 and flip_db >= POLARITY_MARGIN_DB:
        out["polarity"] = -1
    elif out["sign"] > 0 and flip_db <= -POLARITY_MARGIN_DB:
        out["polarity"] = 1
    else:
        out["polarity"] = 0  # the two tests disagree, or neither is clear: do not flip
    return out


def advance(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Move ``x`` earlier by ``ms`` (fractional, exact for band-limited
    signals): a phase ramp on one zero-padded FFT.  The first ``ms`` of the
    track fall off the start; nothing wraps round from the end."""
    if ms == 0.0:
        return x
    shift = ms * 1e-3 * sr
    n = x.shape[0]
    pad = int(math.ceil(abs(shift))) + 64
    from .dsp import _next_fast_len  # one definition of "fast length"
    nf = _next_fast_len(n + pad)
    f = np.fft.rfftfreq(nf)
    ramp = np.exp(2j * np.pi * f * shift)
    X = np.fft.rfft(x, nf, axis=0)
    y = np.fft.irfft(X * (ramp[:, None] if x.ndim == 2 else ramp), nf, axis=0)
    return y[:n]


def apply(x: np.ndarray, sr: int, spec: dict) -> np.ndarray:
    """What the renderer does with a track's ``align`` block."""
    law = spec.get("law")
    if law != ALIGN_LAW:
        raise ValueError(f"unknown align law {law!r}")
    missing = [k for k in ("advance_ms", "polarity") if k not in spec]
    if missing:
        raise ValueError(f"align block is missing {', '.join(missing)}")
    ms = spec["advance_ms"]
    pol = spec["polarity"]
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not (0.0 <= float(ms) <= MAX_LAG_MS):
        raise ValueError(f"align advance_ms must be 0..{MAX_LAG_MS} ms, got {ms!r}")
    if isinstance(pol, bool) or pol not in (1, -1):
        raise ValueError(f"align polarity must be 1 or -1, got {pol!r}")
    y = advance(x, sr, float(ms))
    if pol == -1:
        y = -y
    return y


# ------------------------------------------------------------------ planning


def _loudest(members: list[tuple[str, np.ndarray]], sr: int, source: str) -> int:
    lo, hi = SOURCES[source]["band"]
    rms = [float(np.mean(bandpass(_mono(a), sr, lo, hi) ** 2)) for _, a in members]
    return int(np.argmax(rms))


def _shifted(a: np.ndarray, sr: int, adv_ms: float, pol: int) -> np.ndarray:
    y = advance(_mono(a), sr, adv_ms) if adv_ms else _mono(a)
    return -y if pol == -1 else y


def plan_alignment(tracks: list[dict], audio_by_id: dict, sr: int, mode: str = "full",
                   exclude: set | None = None) -> dict:
    """Decide the moves.  Returns ``{"moves": {id: {advance_ms, polarity}},
    "decisions": [...]}``; ``decisions`` is the receipt (one row per mic
    looked at, moved or not, with the reason)."""
    if mode not in MODES:
        raise ValueError(f"unknown drum align mode {mode!r}")
    exclude = exclude or set()
    decisions: list[dict] = []
    moves: dict[str, dict] = {}
    if mode == "off":
        return {"moves": moves, "decisions": decisions}
    by_role: dict[str, list[str]] = {}
    for t in tracks:
        if t["id"] in exclude or t["id"] not in audio_by_id:
            continue
        by_role.setdefault(t["role"], []).append(t["id"])

    adv: dict[str, float] = {}
    pol: dict[str, int] = {}
    row_of: dict[str, dict] = {}

    def flip_drum(role: str) -> None:
        """Flip every mic of one drum; a mic that could not be lined up says
        so in its own row (it was changed after all)."""
        for i in by_role.get(role) or []:
            pol[i] = -pol.get(i, 1)
            r = row_of.get(i)
            if r is not None and r.get("decision") in ("unchanged", "time kept"):
                r["decision"] = "flipped with its drum (time as recorded)"
                r["note"] = r.pop("reason", None)
    pivots: dict[str, str] = {}
    hits_of: dict[str, np.ndarray] = {}

    # 1. mics on one drum, with each other
    for source in ("kick", "snare"):
        ids = by_role.get(source) or []
        if not ids:
            continue
        members = [(i, audio_by_id[i]) for i in ids]
        p = _loudest(members, sr, source) if len(members) > 1 else 0
        pid = members[p][0]
        pivots[source] = pid
        hits = source_hits(audio_by_id[pid], sr, source)
        hits_of[source] = hits
        lags = {pid: 0.0}
        rows = {}
        for mid, _ in members:
            # every mic of the drum is listed: one that is not lined up keeps
            # its recorded polarity, and a flip of the whole drum (below,
            # against the overheads) must flip it with the others
            pol[mid] = 1
        for mid, a in members:
            if mid == pid:
                continue
            est = estimate(audio_by_id[pid], a, sr, source, hits)
            row = {"track": mid, "role": source, "against": pid, **est}
            row_of[mid] = row
            if not est["ok"]:
                row["decision"] = "unchanged"
                decisions.append(row)
                continue
            lags[mid] = est["lag_ms"]
            pol[mid] = -1 if est["polarity"] == -1 else 1
            row["decision"] = "pending"
            rows[mid] = row
            decisions.append(row)
        # everything moves earlier onto the earliest mic -- but never by more
        # than the search window: with three mics on one drum the spread
        # between the earliest and the latest can exceed what any single pair
        # may (pivot 0, one at -2 ms, one at +9 ms).  The anchor is then the
        # earliest mic still within the window of the latest; a mic earlier
        # than that would need a delay and is left in time as recorded.  The
        # pivot (lag 0) always stays inside, because every pair lag is.
        limit = MAX_LAG_MS - 0.05
        anchor = max(min(lags.values()), max(lags.values()) - limit)
        for mid, lag in lags.items():
            if lag < anchor:
                adv[mid] = 0.0
                if mid in rows and mode == "full":
                    rows[mid]["decision"] = "time kept"
                    rows[mid]["reason"] = (f"the mics on this drum spread over more than {limit:.2f} ms; "
                                           "this one would need a delay, so it stays in time as recorded")
                continue
            a_ms = lag - anchor if mode == "full" else 0.0
            adv[mid] = a_ms if a_ms >= ALREADY_ALIGNED_MS else 0.0
        for row in decisions:
            if row.get("decision") == "pending":
                row["decision"] = "aligned"

    # 2. the overheads onto the snare's close mic (the kick's if there is no snare)
    oh_ids = by_role.get("overheads") or []
    ref_source = "snare" if "snare" in pivots else ("kick" if "kick" in pivots else None)
    oh_adv = 0.0
    if oh_ids and ref_source:
        ref_id = pivots[ref_source]
        ref_sig = _shifted(audio_by_id[ref_id], sr, adv.get(ref_id, 0.0), 1)
        m = min(len(_mono(audio_by_id[i])) for i in oh_ids)
        oh_sum = sum(_mono(audio_by_id[i])[:m] for i in oh_ids)
        hits = hits_of[ref_source]
        if adv.get(ref_id, 0.0):
            hits = hits - int(round(adv[ref_id] * 1e-3 * sr))
            hits = hits[hits > 0]
        est = estimate(ref_sig, oh_sum, sr, ref_source, hits)
        row = {"track": "+".join(oh_ids), "role": "overheads", "against": ref_id, **est}
        if not est["ok"]:
            row["decision"] = "unchanged"
        else:
            # time: the overheads move earlier onto the close mic (full mode)
            if est["lag_ms"] < -ALREADY_ALIGNED_MS:
                row["decision"] = "time kept"
                row["reason"] = ("the overheads arrive before the close mic: the session was "
                                 "already shifted, so its timing is left as it is")
            elif mode != "full":
                row["decision"] = "time kept (polarity only)"
            elif est["lag_ms"] - CLOSE_LEAD_MS >= ALREADY_ALIGNED_MS:
                oh_adv = est["lag_ms"] - CLOSE_LEAD_MS
                row["decision"] = "aligned"
            else:
                row["decision"] = "already aligned"
            # polarity, whatever the time decision: the overheads are the
            # kit's polarity reference, so a reversed drum flips -- every mic
            # on it, the ones that could not be lined up too
            if est["polarity"] == -1:
                flip_drum(ref_source)
                row["flipped"] = ref_source
        decisions.append(row)
        for i in oh_ids:
            adv[i] = oh_adv
            pol.setdefault(i, 1)

        # 3. the kick against the overheads: polarity only
        if ref_source == "snare" and "kick" in pivots:
            kid = pivots["kick"]
            k_sig = _shifted(audio_by_id[kid], sr, adv.get(kid, 0.0), pol.get(kid, 1))
            o_sig = advance(oh_sum, sr, oh_adv) if oh_adv else oh_sum
            khits = hits_of["kick"]
            if adv.get(kid, 0.0):
                khits = khits - int(round(adv[kid] * 1e-3 * sr))
                khits = khits[khits > 0]
            kest = estimate(k_sig, o_sig, sr, "kick", khits)
            krow = {"track": "+".join(oh_ids), "role": "overheads", "against": kid,
                    "check": "kick polarity", **kest}
            if kest["ok"] and kest["polarity"] == -1:
                flip_drum("kick")
                krow["decision"] = "kick flipped"
            else:
                krow["decision"] = "kick polarity kept"
            decisions.append(krow)

    # 4. tom mics: polarity against the overheads, never moved in time (a
    #    tom mic is mostly bleed in between its hits, and moving it changes
    #    the snare and the kick it hears)
    for tid in by_role.get("toms") or []:
        if not oh_ids:
            decisions.append({"track": tid, "role": "toms", "decision": "unchanged",
                              "reason": "no overheads to judge the polarity against"})
            continue
        m = min(len(_mono(audio_by_id[i])) for i in oh_ids)
        o_sum = sum(_mono(audio_by_id[i])[:m] for i in oh_ids)
        o_sig = advance(o_sum, sr, oh_adv) if oh_adv else o_sum
        test = estimate(audio_by_id[tid], o_sig, sr, "toms")
        row = {"track": tid, "role": "toms", "against": "+".join(oh_ids), "check": "polarity only", **test}
        if test["ok"] and test["polarity"] == -1:
            pol[tid] = -1
            row["decision"] = "flipped (time kept)"
        else:
            row["decision"] = "unchanged"
            row.setdefault("reason", "polarity agrees with the overheads, or is not clear")
        decisions.append(row)
    for tid in set(adv) | set(pol):
        a_ms = float(adv.get(tid, 0.0))
        p = int(pol.get(tid, 1))
        if a_ms > 0.0 or p == -1:
            moves[tid] = {"law": ALIGN_LAW, "advance_ms": round(a_ms, 4), "polarity": p}
    for tid in by_role.get("room") or []:
        decisions.append({"track": tid, "role": "room", "decision": "unchanged",
                          "reason": "room mics carry the depth; left as recorded"})
    return {"moves": moves, "decisions": decisions}
