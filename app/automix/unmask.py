"""Bounded, auditable vocal-priority EQ planning on fixed stereo-energy frames.

This is an overlap/level heuristic, not a psychoacoustic masking or speech
recognition model. Rendering reads only the frozen band and mix curve.
"""
from __future__ import annotations

import numpy as np
from .analyze import track_id

TARGET_ROLES = frozenset({'guitar_electric', 'guitar_acoustic', 'keys', 'piano',
                          'synth', 'pad', 'strings', 'other'})
MODE = 'frozen_vocal_unmask_v1'
METHOD = 'stereo_mean_square_hann100ms_hop50ms_v1'
MAX_TARGETS = 2


def frame_power(audio: np.ndarray, sr: int, frames: int) -> dict:
    """Calibrated mean-square band power, same clock/window for every track."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError('unmask frame power requires mono or stereo audio')
    size, hop = max(64, round(.1*sr)), max(1, round(.05*sr))
    starts = np.arange(0, frames, hop)
    window = np.hanning(size)
    frequencies = np.fft.rfftfreq(size, 1/sr)
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1
    if size % 2 == 0:
        weights[-1] = 1
    out = {key: np.zeros(len(starts)) for key in ('rms', 'mid', 'presence')}
    for first in range(0, len(starts), 32):
        offsets = starts[first:first+32]
        batch = np.zeros((len(offsets), size, 2))
        for j, start in enumerate(offsets):
            length = min(size, max(0, len(audio)-int(start)))
            batch[j, :length] = audio[start:start+length]
        out['rms'][first:first+len(offsets)] = np.mean(batch**2, axis=(1, 2))
        spectrum = np.abs(np.fft.rfft(batch*window[None, :, None], axis=1))**2
        for key, lo, hi in (('mid', 500, 2000), ('presence', 2000, 6000)):
            mask = (frequencies >= lo) & (frequencies < hi)
            out[key][first:first+len(offsets)] = np.sum(
                spectrum[:, mask, :]*weights[None, mask, None], axis=(1, 2)
            ) / (size*size*2*np.mean(window**2))
    out['times'] = np.minimum((starts+size/2)/sr, frames/sr)
    return out


def _db(power):
    return 10*np.log10(np.maximum(power, 1e-24))


def _curve(gate: np.ndarray, times: np.ndarray, duration: float) -> list[list[float]]:
    """Finite 50 ms attack / 150 ms release ramps, exactly zero after release."""
    points = [[0.0, 0.0]]
    value, previous = 0.0, 0.0
    for time, active in zip(times, gate):
        time = float(time)
        if time <= previous:
            continue
        value = min(1.0, value+(time-previous)/.05) if active else max(0.0, value-(time-previous)/.15)
        # Preserve native interior precision, with exact saturated endpoints.
        # A mathematical50 ms attack can otherwise stop a few ulps below1.
        if value < 1e-12:
            value = 0.0
        elif value > 1.0-1e-12:
            value = 1.0
        points.append([time, value])
        previous = time
    if points[-1][0] < duration:
        points.append([float(duration), points[-1][1]])
    # Remove only collinear points; the remaining curve is the same function.
    compact = []
    for point in points:
        compact.append(point)
        while len(compact) >= 3:
            a, b, c = compact[-3:]
            left = (b[1]-a[1])*(c[0]-b[0])
            right = (c[1]-b[1])*(b[0]-a[0])
            if abs(left-right) > 1e-8:
                break
            compact.pop(-2)
    return compact


def build(stems: list, tracks: list[dict], intensity: float, guards: dict) -> dict:
    """Write at most two frozen target blocks after all planned fader guards."""
    strength = float(np.clip(intensity, 0, 1))
    summary = {'mode': MODE, 'method': METHOD, 'intensity': strength, 'max_targets': MAX_TARGETS,
               'targets': [], 'priority_source_ids': [], 'diagnostics': [],
               'policy': {'frame_s': .1, 'hop_s': .05, 'attack_ms': 50, 'release_ms': 150,
                          'activity_floor_dbfs': -60, 'activity_relative_db': -30,
                          'presence_floor_dbfs': -65, 'vocal_presence_over_mid_db': -18,
                          'competition_margin_db': -10, 'instrument_cap_db': 1.5, 'other_cap_db': .8,
                          'multiple_leads': 'sum of planned stereo powers; no phase cancellation'}}
    if strength <= 0:
        summary['diagnostics'].append('intensity=0: creative vocal unmask disabled')
        return summary
    # The plan id has one owner.  Match explicitly rather than relying on list
    # positions when callers have dropped silent/excluded tracks.
    by_id = {track_id(s.name): s for s in stems}
    leads = sorted((t for t in tracks if t['role'] == 'lead_vocal'), key=lambda t: t['id'])
    summary['priority_source_ids'] = [t['id'] for t in leads]
    if not leads:
        summary['diagnostics'].append('no prioritized lead vocal; no unmask')
        return summary
    sr = stems[0].sr
    if sr/2 <= 3600:
        summary['diagnostics'].append('sample rate does not support the bounded 3.2 kHz band')
        return summary
    frames = max(len(s.audio) for s in stems)
    measured = {}
    def features(track):
        tid = track['id']
        if tid not in measured:
            raw = frame_power(by_id[tid].audio, sr, frames)
            multiplier = 10**(float(track['gain_db'])/10)
            measured[tid] = {k: v*multiplier if k != 'times' else v for k, v in raw.items()}
        return measured[tid]
    vocal = {k: sum((features(t)[k] for t in leads)) for k in ('rms', 'mid', 'presence')}
    times = features(leads[0])['times']
    vr, vm, vp = (_db(vocal[k]) for k in ('rms', 'mid', 'presence'))
    voice_active = (vr > max(-60.0, float(np.percentile(vr, 95))-30.0))
    voice_supported = (vp >= vm-18) & (vp > -65)
    candidates = []
    for track in tracks:
        if track['role'] not in TARGET_ROLES:
            continue
        if any(abs(float(b['freq'])-3200) < 600 for b in track.get('eq', [])):
            summary['diagnostics'].append(f"{track['id']}: existing corrective EQ already owns this band")
            continue
        power = features(track)
        rms, presence = _db(power['rms']), _db(power['presence'])
        active = rms > max(-60.0, float(np.percentile(rms, 95))-30.0)
        margin = presence-vp
        overlap = voice_active & voice_supported & active & (presence > -65) & (margin >= -10)
        if not np.any(overlap):
            summary['diagnostics'].append(f"{track['id']}: no supported overlap/level competition")
            continue
        score = float(np.sum(power['presence'][overlap]))
        median_margin = float(np.median(margin[overlap]))
        depth = .8 if track['role'] == 'other' or median_margin < -3 else 1.5
        depth = min(depth, float(guards.get('max_eq_db', 6)))*strength
        if depth <= 0:
            continue
        block = {'mode': MODE, 'source_ids': summary['priority_source_ids'],
                 'band': {'freq_hz': 3200.0, 'gain_db': -round(depth, 4), 'q': 1.2},
                 'curve': _curve(overlap, times, frames/sr),
                 'analysis': {'method': METHOD, 'overlap_fraction': round(float(np.mean(overlap)), 6),
                              'planned_presence_margin_median_db': round(median_margin, 3),
                              'planned_gain_db': float(track['gain_db']),
                              'planned_vocal_gains_db': {t['id']: float(t['gain_db']) for t in leads},
                              'reason': 'time-aligned source activity and planned 2-6 kHz competition'}}
        candidates.append((score, track['id'], track, block))
    for _, tid, track, block in sorted(candidates, key=lambda item: (-item[0], item[1]))[:MAX_TARGETS]:
        track['dynamic_eq'] = block
        track.setdefault('evidence', []).append(
            f"vocal masking: dynamic {abs(block['band']['gain_db']):.2f} dB max at 3200 Hz; "
            f"supported overlap {100*block['analysis']['overlap_fraction']:.1f}%; frozen pre-compressor curve")
        summary['targets'].append(tid)
    if not summary['targets']:
        summary['diagnostics'].append('no target selected; no creative band applied')
    return summary


def validate(block: dict, role: str, sr: int, duration: float, source_ids: set[str]) -> None:
    """Validate executable data even without the optional jsonschema package."""
    if not isinstance(block, dict) or block.get('mode') != MODE:
        raise ValueError('unsupported dynamic_eq mode')
    if role not in TARGET_ROLES or set(block) != {'mode', 'source_ids', 'band', 'curve', 'analysis'}:
        raise ValueError('dynamic_eq has unsupported target role or settings')
    if not isinstance(block['analysis'], dict):
        raise ValueError('dynamic_eq analysis provenance must be an object')
    band = block.get('band')
    if not isinstance(band, dict) or set(band) != {'freq_hz', 'gain_db', 'q'}:
        raise ValueError('dynamic_eq band must declare only freq_hz/gain_db/q')
    for key, low, high in (('freq_hz', 200.0, min(10000.0, sr/2-1)),
                           ('gain_db', -.8 if role == 'other' else -1.5, 0), ('q', .3, 5)):
        value = band.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'dynamic_eq.{key} is nonfinite or out of range')
    ids = block.get('source_ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in source_ids for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('dynamic_eq requires unique known priority source_ids')
    points = block.get('curve')
    if not isinstance(points, list) or not 2 <= len(points) <= 100000:
        raise ValueError('dynamic_eq curve needs 2..100000 points')
    previous = -1.0
    previous_mix = None
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('dynamic_eq curve points must be [seconds,mix]')
        time, mix = point
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in point):
            raise ValueError('dynamic_eq curve values must be finite numbers')
        if not previous < time <= duration+.001 or time < 0 or not 0 <= mix <= 1:
            raise ValueError('dynamic_eq curve must be ordered, within source time, and mix within0..1')
        if previous_mix is not None:
            slope = (mix-previous_mix)/(time-previous)
            if slope > 1/.05+.001 or slope < -1/.15-.001:
                raise ValueError('dynamic_eq curve exceeds the50ms attack/150ms release slope limit')
        previous = time
        previous_mix = mix
    if points[0][0] != 0:
        raise ValueError('dynamic_eq curve must start at time0')
