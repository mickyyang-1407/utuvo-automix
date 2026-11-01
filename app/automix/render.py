"""Render a MixPlan to a 24-bit WAV, plus a receipt.

Everything here reads the plan and nothing else: if you change a gain in the
JSON and re-render, you get a different mix.  That is the whole point of
splitting deciding from rendering.

Signal flow, per track::

    gain -> HPF -> corrective EQ -> compressor -> saturation
         -> width (mid/side) -> pan -> group bus

then each group bus (drum / bass / music / vocal / fx) gets its own bus
compression and width, the reverb bus returns straight to the master, and the
master does: bus comp -> tonal tilt EQ -> mono-below -> limiter -> loudness
target -> true-peak check.
"""

from __future__ import annotations

import json
import os
import re
import wave

import numpy as np

from . import analyze, dsp, unmask
from .plan import ROLE_GROUP, PlanError, _manual_reverb_amount

#: plan vocabulary -> dsp vocabulary
KIND_MAP = {
    "bell": "peaking",
    "low_shelf": "lowshelf",
    "high_shelf": "highshelf",
    "high_pass": "highpass",
    "low_pass": "lowpass",
    # a notch is a very narrow, very deep bell as far as the biquad is concerned
    "notch": "peaking",
}

BUS_OF_ROLE = ROLE_GROUP

#: leave this much sample-peak headroom before the master chain
PREMASTER_HEADROOM_DB = -6.0


class RenderError(RuntimeError):
    pass


def _eq_bands(bands: list[dict]) -> list[tuple]:
    out = []
    for b in bands or []:
        if not b.get("enabled", True):
            continue
        kind = KIND_MAP.get(b["type"], "peaking")
        gain = float(b["gain_db"])
        q = float(b.get("q", 1.0))
        if b["type"] == "notch":
            gain, q = -12.0, 8.0
        out.append((kind, float(b["freq"]), gain, q))
    return out


def _set_width(x: np.ndarray, width: float) -> np.ndarray:
    """Mid/side width: 1.0 is untouched, 0.0 collapses to mono, 2.0 doubles."""
    if abs(width - 1.0) < 1e-3:
        return x
    m = x.mean(axis=1, keepdims=True)
    s = (x[:, :1] - x[:, 1:2]) * 0.5
    return m + s * width * np.array([1.0, -1.0])


def _match_width(x: np.ndarray, target: float, limit: float = 2.0) -> np.ndarray:
    """Scale the side signal until the measured width hits ``target``.

    ``width_ratio`` is a side/mid RMS ratio, not a fader position, so a plan
    that says "width 1.4" gets you nothing predictable.  Measuring the signal
    that is actually in hand and solving for the multiplier is the only way to
    land on a number the receipt can then verify.
    """
    cur = dsp.width_ratio(x)
    if cur <= 1e-6 or target <= 0.0:
        return x
    factor = min(max(target / cur, 0.0), limit)
    return _set_width(x, factor)


def _pan(x: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan of a stereo source (0.0 leaves it alone)."""
    if abs(pan) < 1e-3:
        return x
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * np.pi / 4.0  # 45 deg at centre
    lg = np.cos(theta) / np.cos(np.pi / 4.0)
    rg = np.sin(theta) / np.sin(np.pi / 4.0)
    return np.stack([x[:, 0] * lg, x[:, 1] * rg], axis=1)


def _section_env(
    sections: list, n: int, sr: int, key: str, default: float = 1.0
) -> np.ndarray | None:
    """Piecewise-constant envelope from the plan's sections, ramped at edges.

    ``default`` matters: scale-like keys (width, send) default to 1.0, while
    additive ones (air in dB) must default to 0.0.  Getting this wrong adds
    +1 dB of air to any section that does not carry the key.
    """
    if not sections:
        return None
    env = np.full(n, float(default), dtype=np.float64)
    touched = False
    for sec in sections:
        a = max(0, int(float(sec["start_s"]) * sr))
        b = min(n, int(float(sec["end_s"]) * sr))
        if b <= a:
            continue
        env[a:b] = float(sec.get(key, default))
        touched = True
    if not touched:
        return None
    # a half-second ramp; a hard step in width or send is an audible click.
    # NB: this must be the cumsum box filter, not np.convolve with a 24000-wide
    # kernel -- the direct convolution is O(n * w), which turned a 36 s render
    # from seconds into tens of seconds.
    w = max(1, int(0.5 * sr))
    if w > 1 and n > w:
        env = dsp._box_smooth(env, w)
    return env


def _set_width_env(x: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Mid/side width with a per-sample multiplier."""
    m = x.mean(axis=1, keepdims=True)
    s = (x[:, :1] - x[:, 1:2]) * 0.5
    side = s * env[:, None] * np.array([1.0, -1.0], dtype=x.dtype)
    return m + side


def _apply_ride(x: np.ndarray, sr: int, points: list) -> np.ndarray:
    """Apply a piecewise-linear gain automation from the plan.

    ``points`` is ``[[seconds, dB], ...]``.  The curve is interpolated to sample
    resolution and multiplied in -- no smoothing here, because the planner has
    already smoothed it and anything extra would lag the ride.
    """
    if not points:
        return x
    n = x.shape[0]
    ts = np.array([0.0] + [float(p[0]) for p in points] + [n / float(sr)])
    gs = np.array(
        [float(points[0][1])]
        + [float(p[1]) for p in points]
        + [float(points[-1][1])]
    )
    idx = np.arange(n) / float(sr)
    gain_db = np.interp(idx, ts, gs)
    gain = (10.0 ** (gain_db / 20.0)).astype(x.dtype, copy=False)
    return x * gain[:, None] if x.ndim == 2 else x * gain


def _band_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """The level of one frequency band, in dB."""
    mono = x.mean(axis=1).astype(np.float64)
    n = min(len(mono), 1 << 21)
    if n < 64:
        return -120.0
    spec = np.abs(np.fft.rfft(mono[:n])) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return -120.0
    return 10 * float(np.log10(float(spec[m].sum()) + 1e-30))


def _lr_db(x: np.ndarray) -> float:
    """How much louder the left channel is than the right, in dB."""
    l = float(np.mean(x[:, 0].astype(np.float64) ** 2))
    r = float(np.mean(x[:, 1].astype(np.float64) ** 2))
    return 10 * float(np.log10((l + 1e-24) / (r + 1e-24)))


def _rms_db(x: np.ndarray) -> float:
    return float(dsp.lin_to_db(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12))


def _window_lufs(x: np.ndarray, sr: int, start: int, end: int) -> float:
    """Use the source's frozen window and the analyser's short-source method."""
    span = x[start:end]
    return float(dsp.momentary_lufs(span, sr) if len(span) < int(0.4 * sr)
                 else dsp.loudness_lufs(span, sr))


def _level_match(x: np.ndarray, sr: int, spec: dict, source_frames: int, gain_db: float = 0.0) -> tuple[np.ndarray, dict]:
    start, end = spec["start_frame"], spec["end_frame"]
    if spec.get("mode") != "active_lufs_blend" or not 0 <= start < end <= source_frames:
        raise RenderError("invalid level_match mode/source window; check the plan and source audio")
    source_level, amount, cap = (float(spec[k]) for k in ("source_lufs", "amount", "max_gain_db"))
    target = source_level + gain_db
    if not all(np.isfinite(v) for v in (target, amount, cap)) or not 0 <= amount <= 1 or not 0 <= cap <= 6:
        raise RenderError("invalid level_match target/amount/cap")
    before = _window_lufs(x, sr, start, end)
    record = {"before_lufs": before, "target_lufs": target, "amount": amount,
              "start_frame": start, "end_frame": end, "applied_db": 0.0,
              "wanted_db": 0.0, "after_lufs": before, "capped": False, "skipped": None}
    if end - start < int(0.02 * sr) or not np.isfinite(before) or before <= -70.0:
        record["skipped"] = "source window too short or processed signal below loudness floor"
        return x, record
    wanted = amount * (target - before)
    applied = float(np.clip(wanted, -cap, cap))
    x = x * dsp.db_to_lin(applied)
    record.update(wanted_db=wanted, applied_db=applied,
                  after_lufs=_window_lufs(x, sr, start, end), capped=abs(wanted-applied) > 1e-6)
    return x, record


def _stereo_probe_db(x: np.ndarray, sr: int, frequency: float) -> float | None:
    if len(x) < 64:
        return None
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x*window[:, None], axis=0))**2
    f = np.fft.rfftfreq(len(x), 1/sr)
    mask = (f >= frequency*.98) & (f <= frequency*1.02)
    power = float(np.sum(spectrum[mask]))/(len(x)**2*np.mean(window**2))
    return 10*float(np.log10(power)) if power > 1e-20 else None


def _dynamic_eq(x: np.ndarray, sr: int, spec: dict) -> tuple[np.ndarray, dict]:
    """Execute only the plan's frozen crossfade; no activity/role decisions."""
    band = spec['band']
    points = np.asarray(spec['curve'], dtype=float)
    envelope = np.interp(np.arange(len(x))/sr, points[:, 0], points[:, 1])
    filtered = dsp.apply_eq(x, sr, [('peaking', float(band['freq_hz']), float(band['gain_db']), float(band['q']))])
    active = envelope > 0
    result = x.copy()
    result[active] = x[active]+envelope[active, None]*(filtered[active]-x[active])
    max_mix = float(envelope.max())
    nominal = 20*float(np.log10(1-max_mix+max_mix*dsp.db_to_lin(float(band['gain_db']))))
    record = {'mode': spec['mode'], 'duty_fraction': float(np.mean(active)),
              'center_hz': float(band['freq_hz']),
              'max_mix': max_mix, 'max_nominal_cut_db': -nominal,
              'inactive_max_abs_error': float(np.max(np.abs(result[~active]-x[~active]))) if np.any(~active) else 0.0,
              'probes': [], 'measurement_stage': 'dynamic EQ before compressor and loudness match'}
    # A narrow-band physical probe only in stable curve plateaus.  Time-varying
    # boundaries and absent input bands cannot substantiate a precise dB claim.
    boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(envelope)) > 1e-9)+1, len(x)]
    spans = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])
             if b-a >= .15*sr and envelope[a] > 1e-4]
    for start, end in sorted(spans, key=lambda p: p[1]-p[0], reverse=True)[:3]:
        start += int(.01*sr)
        end -= int(.01*sr)
        before = _stereo_probe_db(x[start:end], sr, float(band['freq_hz']))
        after = _stereo_probe_db(result[start:end], sr, float(band['freq_hz']))
        mix = float(envelope[start])
        expected = 20*float(np.log10(1-mix+mix*dsp.db_to_lin(float(band['gain_db']))))
        if before is not None and after is not None and before > -100:
            record['probes'].append({'start_frame': start, 'end_frame': end, 'mix': mix,
                                     'expected_center_db': expected, 'measured_db': after-before,
                                     'before_band_dbfs': before})
    record['probe_status'] = 'measured stable active windows' if record['probes'] else 'no stable supported active probe; no precise attenuation claim'
    return result, record


def _apply_comp(x: np.ndarray, sr: int, spec: dict | None) -> np.ndarray:
    if not spec:
        return x
    # plans without a ratio law were written before it existed and replay
    # with the reading they were rendered with (see plan.COMP_RATIO_LAWS)
    if spec.get("ratio_law", "legacy_amount_v1") == "literal_v1":
        ratio = float(spec.get("ratio", 1.0))
        if ratio <= 1.0:
            return x
        amount, ratio_max = 1.0, ratio
    else:
        amount = min(1.0, max(0.0, (float(spec.get("ratio", 1.0)) - 1.0) / 5.0))
        ratio_max = max(1.01, float(spec.get("ratio", 2.0)))
    threshold = float(spec.get("threshold_db", -24.0))
    if spec.get("threshold_offset_db") is not None:
        # place the threshold relative to what this signal actually contains;
        # a bus's real peaks are not predictable from its tracks' loudness
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
        threshold = round(float(dsp.lin_to_db(rms)) + float(spec["threshold_offset_db"]), 2)
    makeup = spec.get("makeup") or {}
    return dsp.compress(
        x,
        sr,
        amount,
        threshold_db=threshold,
        ratio_max=ratio_max,
        attack_ms=float(spec.get("attack_ms", 12.0)),
        release_ms=float(spec.get("release_ms", 120.0)),
        makeup_mode=str(makeup.get("mode", "half_average_reduction")),
        makeup_cap_db=float(makeup.get("cap_db", 12.0)),
    )


def render(
    stems: list,
    plan: dict,
    reverb_ir_kwargs: dict | None = None,
    collect: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (stereo mix, report dict).  Writing files is the caller's job.

    ``collect``, when a dict is passed, is filled with the per-track processed
    signals and the reverb return -- the pieces a user would want in a DAW.
    They exclude the bus and master processing, which the DAW session should
    redo (or not) on its own.
    """
    rev_plan = dict(plan.get("reverb") or {})
    try:
        manual_amount = _manual_reverb_amount(rev_plan.get("manual_amount", 1.0))
    except PlanError as exc:
        raise RenderError(str(exc)) from exc
    sr = int(plan["source"]["sample_rate"])
    n = max(s.audio.shape[0] for s in stems)
    by_file = {s.name: s for s in stems}

    track_index = {t["id"]: t for t in plan["tracks"]}
    if sum(t.get('dynamic_eq') is not None for t in plan['tracks']) > 2:
        raise RenderError('dynamic vocal unmask supports at most2 targets')
    priority_ids = {t['id'] for t in plan['tracks'] if t['role'] == 'lead_vocal'}
    # The plan is the source of truth, so the reverb shape comes from it.
    # Explicit arguments may still override (the CLI passes the same values;
    # tests pass their own).  Before this, the renderer only read the
    # arguments, so the plan's reverb block -- including each style's
    # high_cut_hz -- was written out and then ignored: every style got the
    # same 8 kHz reverb.
    rev = dict(plan.get("reverb") or {})
    rev.update(reverb_ir_kwargs or {})
    decay = float(rev.get("decay_s", 1.6))
    predelay = float(rev.get("predelay_ms", rev.get("pre_delay_ms", 20.0)))
    high_cut = float(rev.get("high_cut_hz", 8000.0))

    buses: dict[str, np.ndarray] = {}
    reverb_in = np.zeros((n, 2))
    secs = plan.get("sections") or []
    send_env = _section_env(secs, n, sr, "send_scale")
    per_track = []
    # a stem the plan does not mention is left out -- but it must never be
    # left out *silently*: a missing track in a mix is the kind of mistake
    # nobody notices until the client complains
    skipped: list[str] = []
    render_warnings: list[str] = []
    if collect is not None:
        collect["stems"] = {}  # only this render's included tracks may be exported
        collect.pop("reverb", None)  # a reused collection must not retain an older return

    for st in stems:
        # the track id must come from the one place that defines it: the
        # planner wrote the plan with analyze.track_id(), and a second copy
        # of that rule here silently disagreed with it once (a file name with
        # a newline), which left the renderer skipping every stem and
        # producing a silent mix
        tid = analyze.track_id(st.name)
        t = track_index.get(tid)
        if t is None:
            skipped.append(st.name)
            continue  # plan does not mention it: leave it out, loudly reported
        x = st.audio
        if x.shape[0] < n:
            x = np.vstack([x, np.zeros((n - x.shape[0], 2))])

        # Measure what every step does to the level, so the receipt can say
        # why the stem ends up at the level it ends up at.  The plan promises
        # a gain, but the compressor's make-up and the corrective EQ move the
        # level too: without this the plan's number and the audio's number
        # simply do not add up, and the difference looks like a bug.
        chain: list[dict] = []

        def _mark(step: str, before: np.ndarray) -> None:
            after = x  # noqa: F821 -- rebound below
            chain.append(
                {
                    "step": step,
                    "delta_db": round(_rms_db(after) - _rms_db(before), 2),
                }
            )

        x = x * dsp.db_to_lin(float(t.get("gain_db", 0.0)))
        chain.append({"step": "gain", "delta_db": round(float(t.get("gain_db", 0.0)), 2)})
        ride = t.get("ride")
        if ride:
            before = x.copy()
            x = _apply_ride(x, sr, ride)
            _mark("ride", before)

        hpf = float(t.get("hpf_hz", 20.0))
        if hpf > 20.5:
            before = x.copy()
            # measured here, at the point the filter is applied: later steps
            # (the compressor's make-up, the saturation) move every band, so
            # comparing the input and the output afterwards cannot tell what
            # the high-pass did
            lo = max(12.0, hpf / 3.0)
            att_before = _band_db(before, sr, lo, lo * 1.6)
            x = dsp.apply_eq(x, sr, [("highpass", hpf, 0.0, 0.707)])
            att_after = _band_db(x, sr, lo, lo * 1.6)
            hpf_atten = round(att_after - att_before, 2)
            _mark(f"high-pass {hpf:.0f} Hz", before)

        bands = _eq_bands(t.get("eq", []))
        eq_atten: list[dict] = []
        if bands:
            before = x.copy()
            # measured here, where the filter is applied: a bell is wide (at
            # q = 1 the skirt reaches well past the neighbouring band), so
            # comparing the input and the output afterwards cannot tell how
            # much the bell really cut
            x = dsp.apply_eq(x, sr, bands)
            for b in t.get("eq") or []:
                f = float(b.get("freq") or 0)
                if f <= 0 or not b.get("enabled", True):
                    continue
                lo, hi = f * 0.85, f * 1.18
                eq_atten.append(
                    {
                        "freq": f,
                        "gain_db": float(b.get("gain_db") or 0.0),
                        "measured_db": round(
                            _band_db(x, sr, lo, hi) - _band_db(before, sr, lo, hi), 2
                        ),
                    }
                )
            _mark("corrective EQ", before)

        dynamic = None
        if t.get('dynamic_eq') is not None:
            try:
                unmask.validate(t['dynamic_eq'], t['role'], sr, n/sr, priority_ids)
            except (ValueError, TypeError, KeyError) as exc:
                raise RenderError(f'invalid dynamic_eq: {exc}') from exc
            before = x
            x, dynamic = _dynamic_eq(x, sr, t['dynamic_eq'])
            _mark('dynamic vocal EQ', before)

        before = x.copy()
        crest_before = analyze._envelope_crest_db(x, sr)
        x = _apply_comp(x, sr, t.get("comp"))
        crest_after = analyze._envelope_crest_db(x, sr)
        _mark("compressor", before)

        deess = t.get("deess")
        if deess:
            # after compression: squashing a voice makes its sibilance *more*
            # obvious, so the de-esser belongs after the compressor
            before = x.copy()
            x = dsp.deess(
                x,
                sr,
                amount=float(deess.get("amount", 0.0)),
                freq=float(deess.get("freq_hz", 6000.0)),
            )
            _mark("de-esser", before)

        sat = float(t.get("saturation", 0.0))
        if sat > 1e-3:
            before = x.copy()
            x = dsp.saturate(x, sat)
            _mark("saturation", before)

        w = float(t.get("width", 1.0))
        before = x.copy()
        x = _set_width(x, w)
        _mark("width", before)
        before = x.copy()
        # the L/R balance is measured here: the width step above changes it,
        # so the mix's final left/right ratio is not the pan's alone
        lr_before = _lr_db(x)
        x = _pan(x, float(t.get("pan", 0.0)))
        lr_after = _lr_db(x)
        _mark("pan", before)

        matched = None
        if t.get("level_match") is not None:
            before = x
            x, matched = _level_match(x, sr, t["level_match"], len(st.audio), float(t.get("gain_db", 0.0)))
            _mark("loudness match", before)
            if matched["capped"]:
                render_warnings.append(
                    f"{tid}: post-track loudness correction capped to {matched['applied_db']:+.2f} dB "
                    f"(wanted {matched['wanted_db']:+.2f}); source balance is not fully preserved"
                )
            if matched["skipped"]:
                render_warnings.append(f"{tid}: loudness match skipped: {matched['skipped']}")

        send = float((t.get("sends") or {}).get("plate", 0.0))
        if send > 1e-4:
            # accumulate the send *inputs*; the convolution happens once below.
            # Reverb is linear, so conv(sum(x*send)) == sum(conv(x)*send) --
            # but one FFT instead of one per track, which on a 3-minute session
            # is 12 convolutions of an 8.8M-point signal turned into one.
            if send_env is not None:
                reverb_in += x * (send * send_env)[:, None].astype(x.dtype)
            else:
                reverb_in += x * send

        if collect is not None:
            collect.setdefault("stems", {})[tid] = x
        group = BUS_OF_ROLE.get(t["role"], "music")
        buses.setdefault(group, np.zeros((n, 2)))
        buses[group] += x

        per_track.append(
            {
                "id": tid,
                "role": t["role"],
                "bus": group,
                "gain_db": float(t.get("gain_db", 0.0)),
                "sends": send,
                # how loud this track actually is in the mix (post-processing,
                # pre-sum): the number behind "who is too loud in here?"
                "contrib_db": round(
                    dsp.lin_to_db(
                        float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
                    ),
                    2,
                ),
                # what every step did to the level, so the plan's gain and the
                # audio's level can be reconciled
                "chain": chain,
                "level_match": matched,
                "dynamic_eq": dynamic,
                "hpf_atten_db": hpf_atten if hpf > 20.5 else None,
                "eq_atten_db": eq_atten,
                "comp_crest": [round(crest_before, 2), round(crest_after, 2)]
                if t.get("comp")
                else None,
                "pan_lr_db": [round(lr_before, 2), round(lr_after, 2)]
                if abs(float(t.get("pan", 0.0))) > 1e-3
                else None,
                "level_change_db": round(
                    _rms_db(x) - _rms_db(st.audio[: len(x)]), 2
                )
                if len(st.audio) >= len(x)
                else None,
            }
        )
        # The receipt is scalar metadata; only an explicit collector owns the
        # completed track from here. Do not carry track scratch into mastering.
        del before, x

    del send_env

    # Capture the convolution return before it joins the fx bus.  That bus may
    # already contain dry tracks with role=fx; exporting the whole bus as a
    # reverb return would count those dry tracks twice in a DAW.
    wet_ratio_db = None
    if reverb_in.any():
        if manual_amount == 0.0:
            true_return = np.zeros_like(reverb_in)
        else:
            true_return = dsp.reverb_send(reverb_in, sr, 1.0, decay, predelay, high_cut)
            if manual_amount != 1.0:
                true_return = true_return * manual_amount
        dry_sum = np.zeros((n, 2))
        for sig in buses.values():
            dry_sum += sig
        dry_rms = float(np.sqrt(np.mean(dry_sum ** 2)))
        wet_rms = float(np.sqrt(np.mean(true_return.astype(np.float64) ** 2)))
        del dry_sum
        if dry_rms > 1e-12 and wet_rms > 0:
            wet_ratio_db = round(20.0 * np.log10(wet_rms / dry_rms), 2)
        elif wet_rms > 0:
            render_warnings.append("pre-bus reverb/dry ratio is undefined because the dry sum cancels to silence")
        if collect is not None and manual_amount != 0.0:
            collect["reverb"] = true_return
        buses.setdefault("fx", np.zeros((n, 2)))
        buses["fx"] += true_return
        del true_return
    del reverb_in

    # ---- group buses
    bus_specs = plan.get("buses", {})
    for name in buses:
        sig = buses[name]
        spec = bus_specs.get(name) or {}
        sig = _apply_comp(sig, sr, spec.get("comp"))
        sat = float(spec.get("saturation", 0.0))
        if sat > 1e-3:
            sig = dsp.saturate(sig, sat)
        sig = _set_width(sig, float(spec.get("width", 1.0)))
        buses[name] = sig
        del sig

    # The fx bus is part of the original mix path (dry FX plus wet return).
    # Its level must not be described as the pure reverb level.
    mix = np.zeros((n, 2))
    wet_bus = buses.pop("fx", None)
    for sig in buses.values():
        mix += sig
        del sig
    if wet_bus is not None:
        mix += wet_bus
    del wet_bus
    # Preserve the post-fx-pop report exactly, then release completed buses
    # before the master meter's large FFT allocations.
    bus_report = {name: bool(np.any(sig)) for name, sig in buses.items()}
    del buses

    # ---- gain staging.
    # Summing seven stems that were each balanced to a style curve lands well
    # above full scale; here that was +2.5 dBFS, which forced the limiter to
    # remove 3.5 dB of peaks.  Hard clipping that hard sprays harmonics up to
    # Nyquist, the inter-sample peak then overshoots by ~3 dB, and the only way
    # left to hit the true-peak ceiling is to make the mix quieter -- which is
    # how a "mastering" stage ends up costing 2 dB of loudness.  Pulling the
    # sum down to a sane ceiling first means the limiter only ever catches
    # music, not arithmetic.  The loudness loop re-normalises afterwards, so
    # this changes headroom, not the result.
    staging_db = 0.0
    pre_peak = float(np.max(np.abs(mix)))
    if pre_peak > dsp.db_to_lin(PREMASTER_HEADROOM_DB):
        staging_db = PREMASTER_HEADROOM_DB - dsp.lin_to_db(pre_peak)
        mix *= dsp.db_to_lin(staging_db)

    # ---- master
    m = bus_specs.get("master") or {}
    mix = _apply_comp(mix, sr, m.get("comp"))

    mb = float(m.get("mono_below_hz", 0.0))
    if mb > 0:
        mix = dsp.mono_below(mix, sr, mb)

    master_bands = _eq_bands(m.get("eq", []))
    if master_bands:
        mix = dsp.apply_eq(mix, sr, master_bands)

    gain_env = _section_env(secs, n, sr, "gain_db", default=0.0)
    if gain_env is not None and float(np.max(np.abs(gain_env))) > 1e-6:
        # section dynamics vs the reference: a slow gain contour, applied before
        # the master chain so the limiter sees the same shape the audience will
        mix = mix * (10.0 ** (gain_env / 20.0)).astype(mix.dtype)[:, None]
    del gain_env

    air_env = _section_env(secs, n, sr, "air_db", default=0.0)
    if air_env is not None:
        # per-section tone, done the cheap way: master EQ is a filter, not a
        # per-sample operation, so render the two extremes once and crossfade
        # between them with the section envelope.  One extra filter pass beats
        # a time-varying EQ by a wide margin in both cost and complexity.
        lo = float(np.min(air_env))
        hi = float(np.max(air_env))
        if hi - lo > 1e-3:
            darker = dsp.apply_eq(mix, sr, [("highshelf", 9000.0, lo, 0.7)])
            brighter = dsp.apply_eq(mix, sr, [("highshelf", 9000.0, hi, 0.7)])
            w = ((air_env - lo) / (hi - lo)).astype(mix.dtype)[:, None]
            mix = darker * (1.0 - w) + brighter * w
            del darker, brighter, w
    del air_env

    width_env = _section_env(secs, n, sr, "width_scale")
    if width_env is not None:
        mix = _set_width_env(mix, width_env)
    elif m.get("width_target"):
        mix = _match_width(mix, float(m["width_target"]), limit=float(m.get("width_limit", 2.0)))
    else:
        mix = _set_width(mix, float(m.get("width", 1.0)))
    del width_env

    target = float(m.get("target_lufs", -14.0))
    ceiling = float(m.get("limiter", {}).get("ceiling_db", m.get("true_peak_db", -1.0)))

    mix, loud, master_gain_db = _hit_target(mix, sr, target, ceiling)

    report = {
        "tracks": per_track,
        "skipped_stems": skipped,
        "warnings": render_warnings,
        "buses": bus_report,
        "lufs": round(dsp.loudness_lufs(mix, sr), 2),
        # the native rate the measurements above were taken at, so a
        # downstream diagnostic measures the same audio at the same rate
        "sample_rate": int(sr),
        "target_lufs": target,
        "true_peak_db": round(dsp.true_peak_db(mix, sr), 2),
        "ceiling_db": ceiling,
        "peak_db": round(dsp.lin_to_db(float(np.max(np.abs(mix)))), 2),
        "crest_db": round(dsp.crest_db(mix), 2),
        "correlation": round(dsp.stereo_correlation(mix), 3),
        "width": round(dsp.width_ratio(mix), 3),
        "wet_ratio_db": wet_ratio_db,
        "wet_ratio_measurement": "pre-bus pure convolution return over all dry tracks",
        "loudness_passes": loud,
        "staging_db": round(staging_db, 2),
        "master_gain_db": round(master_gain_db, 2),
    }
    decision = (plan.get("style") or {}).get("source_balance")
    if decision is not None:
        report["source_balance"] = json.loads(json.dumps(decision))
    return mix, report


def _hit_target(
    mix: np.ndarray, sr: int, target_lufs: float, ceiling_db: float, passes: int = 5
) -> tuple[np.ndarray, int]:
    """Gain to the loudness target, with a true-peak limiter in between.

    Limiting costs loudness, so this is a feedback loop: measure the residual,
    correct the cumulative gain, limit again.  The gain is always applied to
    the *unprocessed* sum.  Applying it to the already-limited output instead
    compounds it (pass 2 would be +7 dB, pass 3 +14 dB...), the limiter then
    fights a signal that is already squashed, and the loop converges on
    "quiet but flattened" -- exactly the failure that produced 6.8 dB crest
    at 1.5 dB under target.
    """
    ref = mix.copy()
    gain_db = 0.0
    used = 0
    for i in range(1, passes + 1):
        used = i
        cur = dsp.loudness_lufs(mix, sr)
        if cur <= -70.0:
            break  # silence: nothing sensible to normalise
        gain_db += target_lufs - cur
        mix = dsp.true_peak_limit(ref * dsp.db_to_lin(gain_db), sr, ceiling_db)
        if abs(dsp.loudness_lufs(mix, sr) - target_lufs) < 0.2:
            break

    # guarantee, not hope: if the target was unreachable without clipping,
    # give up a little loudness rather than the ceiling
    tp = dsp.true_peak_db(mix, sr)
    if tp > ceiling_db:
        mix = mix * dsp.db_to_lin(ceiling_db - tp)
        gain_db += ceiling_db - tp
    # the applied gain is part of the result: without it nobody can say how
    # far the exported stems are from the finished mix
    return mix, used, gain_db



def write_wav24(path: str, audio: np.ndarray, sr: int) -> None:
    """24-bit PCM, which is what every DAW will happily re-import."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = np.clip(audio, -1.0, 1.0)
    # scale to 24-bit range, then split into three little-endian bytes
    v = np.round(data * 8388607.0).astype(np.int32).reshape(-1)
    v = np.clip(v, -8388608, 8388607)
    b = np.empty((v.size, 3), dtype=np.uint8)
    u = v & 0xFFFFFF
    b[:, 0] = u & 0xFF
    b[:, 1] = (u >> 8) & 0xFF
    b[:, 2] = (u >> 16) & 0xFF
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b.tobytes())


def _verdicts(plan: dict, report: dict) -> list[tuple[str, str, str]]:
    """A short pass/warn table for the top of the receipt.

    Everything here is already measured; the point is that a person opening the
    file should be able to see "did it land?" before reading 40 rows of numbers.
    """
    rows: list[tuple[str, str, str]] = []

    def add(name: str, value: str, ok: bool, note: str = "") -> None:
        rows.append((name, value, "ok" if ok else ("warn: " + note if note else "warn")))

    master = (plan.get("buses") or {}).get("master") or {}
    target = float(master.get("target_lufs", report.get("target_lufs", -14.0)))
    got = float(report.get("lufs", -70.0))
    ceiling = float(master.get("true_peak_db", report.get("ceiling_db", -1.0)))
    tp = float(report.get("true_peak_db", 0.0))
    note = "missed target"
    if got < target - 0.5 and tp >= ceiling - 0.2:
        # The mix is as loud as it may be: the peak ceiling is the binding
        # constraint, not a failure to aim.  Say so, with the number.
        note = (
            f"missed target: the peak ceiling ({ceiling:.1f} dBTP) is the limit, "
            f"it would need {target - got:+.2f} dB more"
        )
    skipped = report.get("skipped_stems") or []
    if skipped:
        add(
            "stem coverage",
            f"{len(skipped)} left out",
            False,
            "the plan does not mention: " + ", ".join(skipped[:6]),
        )
    add("loudness", f"{got:.2f} / {target:.1f} LUFS", abs(got - target) <= 0.5, note)
    add("true peak", f"{tp:.2f} / {ceiling:.1f} dBTP", tp <= ceiling + 0.05, "over ceiling")

    if report.get("wet_ratio_db") is not None:
        wet = float(report["wet_ratio_db"])
        add("reverb", f"{wet:+.1f} dB wet/dry (pre-bus)", wet <= -8.0, "very wet")

    tone = report.get("reference_tone")
    if tone:
        before, after = tone["before_db"], tone["after_db"]
        text = (f"{before:.2f} -> {after:.2f} dB, {tone['band_count']} fixed bands"
                if before is not None and after is not None else "insufficient common supported bands")
        add("reference tone distance", text, after is not None and after <= 0.5, tone["diagnostic"])

    if report.get("mono_loss_db") is not None:
        mono = float(report["mono_loss_db"])
        add("mono fold", f"{mono:+.1f} dB", mono <= 3.0, "phase problems")

    quiet = (plan.get("style") or {}).get("crest_target_db")
    if quiet:
        crest = float(report.get("crest_db", 0.0))
        add("dynamics", f"{crest:.1f} / ~{float(quiet):.1f} dB crest", crest >= float(quiet) - 4.0,
            "flatter than the style expects")
    return rows


def write_report(path: str, plan: dict, report: dict, stems: list) -> None:
    """A human-readable receipt: what was done, and what was measured."""
    w = plan.get("style", {})
    lines = [
        f"# Mix receipt — {w.get('name') or w.get('id')}",
        "",
        f"- style: `{w.get('id')}` v{w.get('version')} · intensity {w.get('intensity')}",
        f"- source: {len(stems)} stems · {plan['source']['duration_s']:.2f} s · "
        f"{plan['source']['sample_rate']} Hz",
        f"- target: {report['target_lufs']} LUFS, ceiling {report['ceiling_db']} dBTP",
        "",
        "## Checks",
        "",
        "| check | measured | verdict |",
        "| --- | --- | --- |",
    ]
    manual_amount = dict(plan.get("reverb") or {}).get("manual_amount", 1.0)
    if manual_amount != 1.0:
        lines[5:5] = [f"- added reverb manual amount: {manual_amount*100:g}%; source room is unchanged."]
    decision = w.get("source_balance")
    if decision:
        proof = decision["proof"]
        lines[5:5] = [
            f"- source balance intent: `{decision['requested']}` → `{decision['resolved']}` · "
            f"k={decision['balance_preserve']} · origin `{decision['origin']}`",
            f"- source balance proof: {proof['reason']} · residual {proof.get('residual', proof.get('relative_rms_residual', '—'))} · "
            f"correlation {proof.get('correlation', '—')} · active coverage {proof.get('reference_active_seconds', proof.get('active_coverage_s', '—'))} s",
            "- balance boundary: per-track active-window LUFS before buses; caps/skips are reported; bus/master may change the balance.",
        ]
    if decision and decision["requested"] == "reference_balance":
        if decision["origin"] == "reference_fit":
            details = [
                "- same-song rough balance: raw-source gain-only reconstruction passed; this is not final mix/reference identity.",
                f"- common fit gain: {proof['common_gain_db']:+.3f} dB, intentionally unapplied; canonical midrange gauge.",
            ]
            details += [f"- fitted relative correction `{id}`: {gain:+.3f} dB (included once in current track gain_db)"
                        for id,gain in proof["relative_gain_db"].items()]
        elif decision["origin"] == "explicit":
            details = ["- same-song rough balance: fitting skipped; explicit manual k takes priority."]
        elif decision["origin"] == "proof":
            details = ["- same-song rough balance: source already reconstructs with one common gain; no new per-stem correction."]
        else:
            details = [f"- same-song rough balance: unavailable ({proof['reason']}); fallback to unknown-input k=0.5, no partial correction."]
        index = lines.index("## Checks")
        lines[index:index] = details + [""]
    # What the declared capture groups protected, and where that protection
    # stops.  This is receipt text only: the numbers that carry the
    # protection are already in the plan's tracks.
    receipt = plan.get("capture_groups")
    if receipt and (receipt.get("groups") or []):
        details = ["## Capture groups", ""]
        for group in receipt["groups"]:
            if group.get("mode") == "stereo_pair":
                # A pair is not one common pan: the two members keep their own
                # channel and their own pan.
                members = group.get("members") or []
                left = members[0] if len(members) > 0 else ""
                right = members[1] if len(members) > 1 else ""
                details.append(
                    f"- {group['id']}: stereo pair — left {left}, right {right} · "
                    f"pair position {group['pan']:+.2f} · "
                    f"shared gain {float(group.get('trim_db', 0.0)):+.2f} dB"
                )
            else:
                details.append(
                    f"- {group['id']}: {', '.join(group['protected_tracks'])} · "
                    f"common pan {group['pan']:+.2f}"
                )
            for note in group.get("notes") or []:
                details.append(f"  - {note}")
        if any(g.get("mode") == "stereo_pair" for g in receipt["groups"]):
            details += [
                "- stereo pair: the two members keep their own channel and their own pan; "
                "they are not put on one shared pan position.",
                "- stereo pair: at position 0 the imported left/right relationship is kept "
                "before the common bus/master; moving the position narrows the image and "
                "both members meet on the same side at the extremes; the shared trim moves "
                "both members together.",
                "- stereo pair: this is a stereo image position -- not a balance control, "
                "not an engineer's recipe and not a quality guarantee. An individual pan or "
                "trim override changes the image, and the common bus/master still changes "
                "the finished mix.",
            ]
        details += ["", f"- boundary: {receipt.get('boundary', '')}", ""]
        index = lines.index("## Checks")
        lines[index:index] = details
    for name, value, verdict in _verdicts(plan, report):
        lines.append(f"| {name} | {value} | {verdict} |")

    lines += [
        "",
        "## Result",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| integrated loudness | **{report['lufs']} LUFS** |",
        f"| true peak | {report['true_peak_db']} dBTP |",
        f"| sample peak | {report['peak_db']} dBFS |",
        f"| crest factor | {report['crest_db']} dB |",
        f"| stereo correlation | {report['correlation']} |",
        "| reverb wet/dry (pre-bus pure return / all dry tracks) | "
        + (f"{report['wet_ratio_db']:+.1f} dB |" if report.get("wet_ratio_db") is not None else "— |"),
        "| mono fold-down | "
        + (f"{report['mono_loss_db']:+.2f} dB |" if "mono_loss_db" in report else "— |"),
        f"| width | {report['width']} |",
        f"| loudness passes | {report['loudness_passes']} |",
        f"| master gain | {report.get('master_gain_db', 0.0):+.2f} dB |",
        "",
        "## Tracks",
        "",
        "| track | role | bus | gain | send | level | net |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in report["tracks"]:
        level = f"{t['contrib_db']:+.1f} dB" if t.get("contrib_db") is not None else "—"
        net = f"{t['level_change_db']:+.2f} dB" if t.get("level_change_db") is not None else "—"
        lines.append(
            f"| {t['id']} | {t['role']} | {t['bus']} | {t['gain_db']:+.2f} dB | "
            f"{t['sends']:.3f} | {level} | {net} |"
        )

    lines += ["", "## Why", ""]
    for tr in plan["tracks"]:
        ev = "; ".join(tr.get("evidence", []))
        lines.append(f"- **{tr['id']}** — {ev}")
        eq = tr.get("eq") or []
        if eq:
            desc = ", ".join(f"{b['type']} {b['freq']:.0f} Hz {b['gain_db']:+.1f} dB" for b in eq)
            lines.append(f"  - corrective EQ: {desc}")
        # the level the plan promised and the level the audio got, step by
        # step -- the compressor's make-up is part of why they differ
        rep = next((r for r in report["tracks"] if r["id"] == tr["id"]), None)
        if rep and rep.get("chain"):
            steps = ", ".join(
                f"{c['step']} {c['delta_db']:+.2f} dB" for c in rep["chain"]
            )
            lines.append(f"  - level: {steps} = {rep.get('level_change_db', 0.0):+.2f} dB net")
        if rep and rep.get("level_match"):
            match = rep["level_match"]
            lines.append(
                f"  - pre-bus loudness: {match['before_lufs']:.2f} -> {match['after_lufs']:.2f} LUFS, "
                f"target {match['target_lufs']:.2f}; correction {match['applied_db']:+.2f} dB; "
                f"capped={match['capped']}, skipped={match['skipped']}"
            )
        if rep and rep.get('dynamic_eq'):
            dynamic = rep['dynamic_eq']
            center = dynamic.get('center_hz', tr.get('dynamic_eq', {}).get('band', {}).get('freq_hz', 3200))
            lines.append(f"  - vocal unmask: {100*dynamic['duty_fraction']:.1f}% time, max nominal center cut {dynamic['max_nominal_cut_db']:.2f} dB at {center/1000:.1f} kHz; {dynamic['measurement_stage']}; {dynamic['probe_status']}")
            for probe in dynamic['probes']:
                lines.append(f"  - unmask probe {probe['start_frame']}:{probe['end_frame']}: expected center {probe['expected_center_db']:+.2f} dB, measured narrow-band {probe['measured_db']:+.2f} dB")

    secs = plan.get("sections") or []
    if secs:
        lines += ["", "## Sections", ""]
        lines.append("| section | start | end | width × | send × | air | gain |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for sec in secs:
            lines.append(
                f"| {sec['label']} | {sec['start_s']:.1f} s | {sec['end_s']:.1f} s | "
                f"{sec.get('width_scale', 1.0):.3f} | {sec.get('send_scale', 1.0):.3f} | "
                f"{sec.get('air_db', 0.0):+.2f} dB | {sec.get('gain_db', 0.0):+.2f} dB |"
            )
        info = plan.get("section_contrast") or {}
        if info.get("reference_db") is not None:
            lines.append(
                f"\nReference section contrast: {info['reference_db']} dB → factor "
                f"{info['factor']:.2f}."
            )

    ref = (plan.get("style") or {}).get("reference")
    if ref:
        lines += ["", "## Reference", ""]
        lines.append(f"- file: `{ref.get('file')}`")
        lines.append(f"- loudness {ref.get('lufs')} LUFS · crest {ref.get('crest_db')} dB · "
                     f"width {ref.get('width')}")
        if ref.get("section_contrast_db") is not None:
            lines.append(f"- section contrast: {ref.get('section_contrast_db')} dB")
        scale = float((plan.get("reverb") or {}).get("scale", 1.0))
        lines.append(f"- reverb match scale: ×{scale:.2f}")
        tone = report.get("reference_tone")
        if tone:
            lines.append(f"- tone analysis: {tone['analysis_method']['id']}; {tone['band_count']} fixed common bands; {tone['baseline_kind']}")
            if tone['before_db'] is None or tone['after_db'] is None:
                lines.append("- relative-tone RMS distance: not enough comparable bands (no relative-tone conclusion); at least two common comparable bands are needed.")
            else:
                lines.append(f"- relative-tone RMS distance: {tone['before_db']} -> {tone['after_db']} dB. {tone['diagnostic']}")
            lines.append("- This is physical spectral-shape distance, not a perceptual quality score.")

    trims = [t for t in plan.get("tracks", []) if t.get("trim_db")]
    if trims:
        lines += ["", "## Manual overrides", ""]
        for t in trims:
            lines.append(f"- {t['id']}: {t['trim_db']:+.1f} dB → gain {t['gain_db']:+.2f} dB")

    warns = list(plan.get("warnings") or []) + list(report.get("warnings") or [])
    lines += ["", "## Guards", ""]
    if warns:
        lines += [f"- {x}" for x in warns]
    else:
        lines.append("- no guard was hit; every requested value was applied as-is")

    cmd = f"python3 -m automix.cli --stems <dir> --style {w.get('id')}"
    if w.get("intensity") not in (None, 1.0):
        cmd += f" --intensity {w.get('intensity')}"
    if ref:
        cmd += " --reference <ref.wav> --blend " + str((plan.get("style") or {}).get("blend", 0.7))
    cmd += " --out <dir>"
    lines += [
        "",
        "## Reproduce",
        "",
        "```sh",
        cmd,
        "```",
        "",
        "The full decision set is in `mixplan.json` next to this file.  Edit a gain",
        "and re-render: that gain changes, and the master loudness stage compensates",
        "for the difference across the whole mix -- which is the point of a",
        "loudness target, and not a hidden change to the other tracks.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_plan_json(path: str, plan: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _quantized24(audio: np.ndarray) -> np.ndarray:
    """The exact decoded values written by write_wav24 (PCM divides by 2**23)."""
    return np.round(np.clip(audio, -1.0, 1.0) * 8388607.0) / 8388608.0


def _export_comparison(total: np.ndarray, reference: np.ndarray | None, sr: int) -> dict:
    """Compare aligned stereo audio; gain and spectral shape are separate."""
    def level(x):
        power = float(np.mean(np.asarray(x, dtype=np.float64) ** 2))
        return 10.0 * float(np.log10(power)) if power > 0 else None

    result = {"sum_rms_dbfs": level(total), "sum_lufs": None, "comparison_frames": len(total)}
    if np.any(total):
        loud = _window_lufs(total, sr, 0, len(total))
        result["sum_lufs"] = loud if loud > -70 else None
    if reference is None:
        result["reference_available"] = False
        return result
    if reference.shape != total.shape:
        raise RenderError("export and final mix must have the same aligned stereo frame count")
    ref = _quantized24(np.asarray(reference, dtype=np.float64))
    result["reference_available"] = True
    result["reference_rms_dbfs"] = level(ref)
    loud = _window_lufs(ref, sr, 0, len(ref)) if np.any(ref) else -70.0
    result["reference_lufs"] = loud if loud > -70 else None
    for field in ("rms_dbfs", "lufs"):
        a, b = result["sum_"+field], result["reference_"+field]
        result["reference_minus_sum_"+field] = b-a if a is not None and b is not None else None
    a = np.asarray(total, dtype=np.float64).reshape(-1)
    b = ref.reshape(-1)
    aa, bb, ab = float(np.dot(a,a)), float(np.dot(b,b)), float(np.dot(a,b))
    alpha = ab / aa if aa > 0 else 0.0
    ac, bc = a-a.mean(), b-b.mean()
    denom = float(np.sqrt(np.dot(ac,ac)*np.dot(bc,bc)))
    result["correlation"] = float(np.dot(ac,bc)/denom) if denom > 0 else None
    result["gain_match_linear"] = alpha
    result["gain_match_db"] = 20*float(np.log10(abs(alpha))) if alpha else None
    result["gain_match_polarity"] = -1 if alpha < 0 else 1
    raw_error = float(np.dot(a-b,a-b))
    shape_error = float(np.dot(a*alpha-b,a*alpha-b))
    result["raw_residual_ratio"] = float(np.sqrt(raw_error/bb)) if bb > 0 else None
    result["gain_matched_residual_ratio"] = float(np.sqrt(shape_error/bb)) if bb > 0 else None
    result["gain_matched_residual_db"] = 10*float(np.log10(shape_error/bb)) if bb > 0 and shape_error > 0 else None
    n = len(total)
    frequencies = np.fft.rfftfreq(n, 1.0/sr)
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    def band_energy(x):
        spectrum = np.fft.rfft(x, axis=0)
        return np.sum(np.abs(spectrum)**2, axis=1) * weights / (n*n*2)
    ea, eb = band_energy(total), band_energy(ref)
    result["bands"] = []
    for name, lo, hi in (("low",20,160),("low_mid",160,500),("mid",500,2000),
                          ("presence",2000,6000),("air",6000,sr/2+1)):
        mask = (frequencies >= lo) & (frequencies < hi)
        pa, pb = float(ea[mask].sum()), float(eb[mask].sum())
        da = 10*float(np.log10(pa)) if pa > 1e-30 else None
        db = 10*float(np.log10(pb)) if pb > 1e-30 else None
        gap = db-da if da is not None and db is not None else None
        result["bands"].append({"name":name,"low_hz":lo,"high_hz":min(hi,sr/2),
            "sum_dbfs":da,"reference_dbfs":db,"reference_minus_sum_db":gap,
            "gain_matched_gap_db": gap-20*float(np.log10(abs(alpha)))
                if gap is not None and alpha else None})
    return result


def write_stem_export(
    out_dir: str, p: dict, collected: dict, report: dict | None = None,
    mix: np.ndarray | None = None,
) -> list[dict]:
    """Write safe aligned tracks, their return, and measured DAW differences.

    One common attenuation protects individual files AND their sum.  Metrics
    use the actual quantized files, including the return, at unity DAW faders.
    """
    sr = int(p["source"]["sample_rate"])
    signals = [(str(tid), "track", np.asarray(audio, dtype=np.float64))
               for tid, audio in sorted((collected.get("stems") or {}).items())]
    if not signals:
        raise RenderError("no processed stems to export")
    reverb = collected.get("reverb")
    if reverb is not None and np.any(reverb):
        signals.append(("_reverb", "return", np.asarray(reverb, dtype=np.float64)))
    shape = signals[0][2].shape
    if len(shape) != 2 or shape[1] != 2 or shape[0] == 0:
        raise RenderError("processed export requires nonempty stereo audio")
    if any(audio.shape != shape or not np.all(np.isfinite(audio)) for _,_,audio in signals):
        raise RenderError("processed export requires finite, aligned stereo tracks")
    if mix is not None and (mix.shape != shape or not np.all(np.isfinite(mix))):
        raise RenderError("final mix and exported tracks must be finite and aligned")
    total = np.zeros(shape, dtype=np.float64)
    peak = 0.0
    for _,_,audio in signals:
        total += audio
        peak = max(peak,float(np.max(np.abs(audio))))
    sum_peak = float(np.max(np.abs(total)))
    peak = max(peak,sum_peak)
    # A tiny extra margin covers accumulation of PCM rounding errors.
    gain_db = min(0.0, -1.001 - 20*float(np.log10(peak))) if peak > 0 else 0.0
    gain = dsp.db_to_lin(gain_db)
    raw_sum_rms = _rms_db(total) if np.any(total) else None
    del total
    os.makedirs(out_dir, exist_ok=True)
    written, mapping = [], []
    used = {"_reverb.wav", "mix.wav", "readme.txt", "_master_chain.json", "_export_report.json"}
    total = np.zeros(shape, dtype=np.float64)
    for index, (tid, kind, audio) in enumerate(signals):
        if kind == "return":
            filename = "_reverb.wav"
        else:
            stem = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', tid).strip(' .')[:180] or "track"
            filename = stem + ".wav"
            if filename.casefold() in used:
                filename = f"{stem}__track{index+1}.wav"
            while filename.casefold() in used:
                filename = "track_" + filename
            used.add(filename.casefold())
        path = _export_path(out_dir, filename)
        write_wav24(path, audio * gain, sr)
        decoded, decoded_sr = analyze.load_stem_file(path)
        if decoded_sr != sr or decoded.shape != shape:
            raise RenderError("export readback changed sample rate or frame alignment")
        total += decoded.astype(np.float64)
        entry = {"source_id":tid,"kind":kind,"filename":filename,"frames":shape[0],
                 "sample_rate":sr,"channels":2,"common_gain_db":gain_db,
                 "sample_peak_dbfs":_sample_peak_db(decoded)}
        mapping.append(entry)
        written.append({"name":filename,"bytes":os.path.getsize(path)})
    comparison = _export_comparison(total, mix, sr)
    summary = {"version":1,"files":mapping,"common_gain_db":gain_db,
               "headroom_ceiling_dbfs":-1.0,"sum_sample_peak_dbfs":_sample_peak_db(total),
               "raw_sum_rms_dbfs_before_common_gain":raw_sum_rms, "comparison":comparison,
               "definitions":{
                   "gap":"reference minus exported sum; positive means final mix is louder",
                   "reference":"separate mix.wav, quantized with the same 24-bit writer; never add it to the stem sum",
                   "sum":"actual PCM24 readback, all listed track and return files at unity, start frame 0",
                   "bands":"mean stereo energy ((L squared + R squared) / 2), matching time-domain mean square; absolute band dBFS use this average, no mono cancellation; 20 Hz to Nyquist",
                   "return":"_reverb.wav is the pure convolution return captured before joining the fx bus; dry fx tracks and fx-bus compression/saturation/width are excluded",
                   "gain_matched":"one least-squares scalar applied only for analysis; shape residual is relative to reference energy",
                   "null":"undefined on silence/empty band, or minus infinity for an exactly zero residual; residual_ratio=0 identifies exact zero",
                   "lufs":"whole aligned files, momentary method below 0.4 seconds; null at/below -70 LUFS floor"}}
    if report is not None:
        report["export_summary"] = summary
    master = p["buses"]["master"]
    normalisation = float((report or {}).get("master_gain_db",0.0))
    chain = {"description":"Partial processing settings for manual reference; not a DAW importer or exact reconstruction recipe",
             "limitations":["Track files and the pure convolution return are pre-bus; bus saturation and the full ordered master/section signal path are not reproduced by this summary", "The original fx bus processes dry fx tracks together with the pure reverb return; this summary is not an exact DAW reconstruction recipe", "Compressor detectors and gain reduction depend on audio; matching settings in another DAW need not match DSP"],
             "master":{"eq":master.get("eq",[]),"limiter_ceiling_db":master.get("limiter",{}).get("ceiling_db"),
                       "target_lufs":master.get("target_lufs"),"true_peak_db":master.get("true_peak_db"),
                       "mono_below_hz":master.get("mono_below_hz"),"loudness_gain_db":normalisation},
             "buses":{name:{"comp":bus.get("comp"),"width":bus.get("width"),"saturation":bus.get("saturation")}
                      for name,bus in p["buses"].items() if name != "master"},
             "reverb":p.get("reverb",{}),"export_common_gain_db":gain_db}
    for filename, obj in (("_master_chain.json",chain),("_export_report.json",summary)):
        path=_export_path(out_dir,filename)
        with open(path,"w",encoding="utf-8") as fh:
            json.dump(obj,fh,indent=2,ensure_ascii=False,allow_nan=False)
            fh.write("\n")
        written.append({"name":filename,"bytes":os.path.getsize(path)})
    gap=comparison.get("reference_minus_sum_rms_dbfs")
    note=["automix stem export", "", "Import every WAV listed in _export_report.json at frame 0, unity faders.",
          "The shared reverb return is included in the measured sum. Do not add the separate final mix.wav.",
          "_reverb.wav is the pure convolution return before the fx bus. It excludes dry FX tracks and fx-bus compression, saturation and width.",
          "Each track contains gain, ride, HPF, EQ, compression, de-ess, saturation, width, pan and declared loudness match.",
          "", "NOT included in the track files or pure return: bus/master compression, saturation, EQ, width, mono-below, section/master processing and limiting.",
          "Summing these files does NOT reproduce mix.wav.",
          "_master_chain.json is a PARTIAL settings reference, not an importer or exact reconstruction recipe.",
          f"One common export gain: {gain_db:+.4f} dB; all tracks and return use the same value.",
          "Individual files AND the sum retain at least -1 dBFS sample-peak headroom.",
          "Measured on actual 24-bit files, with return included:"]
    if gap is not None:
        note.append(f"Measured: summing these tracks lands {gap:+.2f} dB against mix.wav (reference minus sum; positive = final mix louder).")
    else:
        note.append("RMS gap is undefined: no reference or silent sum/reference.")
    note += [f"The loudness normalisation alone was {normalisation:+.2f} dB.",
             "LUFS gap, correlation, raw residual, gain-matched shape residual and stereo frequency-band differences are in _export_report.json.",
             "Level matching in that analysis is virtual; it does not change the exported audio.",
             "Null measurements mean silence/empty bands/undefined ratios; an exact-zero residual has ratio 0.", ""]
    path=_export_path(out_dir,"README.txt")
    with open(path,"w",encoding="utf-8") as fh:
        fh.write("\n".join(note))
    written.append({"name":"README.txt","bytes":os.path.getsize(path)})
    return written


def _sample_peak_db(audio: np.ndarray) -> float | None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20*float(np.log10(peak)) if peak > 0 else None


def _export_path(out_dir: str, filename: str) -> str:
    path = os.path.join(out_dir, filename)
    if os.path.basename(filename) != filename or os.path.islink(path):
        raise RenderError("export destination must be a regular child file, not a link")
    if os.path.commonpath([os.path.realpath(out_dir), os.path.realpath(path)]) != os.path.realpath(out_dir):
        raise RenderError("export destination escapes its output directory")
    return path
