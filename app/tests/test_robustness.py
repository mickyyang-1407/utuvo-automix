"""Robustness tests for real-world stem sets.

Everything here comes from a torture run over a deliberately nasty session:
generic filenames, a silent stem, a 150 ms one-shot, a heavily clipped stem, an
anti-phase pad, hot stems, mixed lengths.  Each test pins one of the things that
went wrong that night.
"""

from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _tone(freq, dur, amp=0.3, sr=SR):
    n = int(dur * sr)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)


def _stereo(x):
    return np.stack([x, x], axis=1).astype(np.float32)


def _stem(name, audio, sr=SR, role="other", source="keyword", sr_override=None):
    s = analyze.Stem(name=name, path=name, audio=audio, sr=sr_override or sr)
    s.role = role
    s.role_source = source
    return s


# ------------------------------------------------------------------ measurement

def test_active_span_ignores_silence():
    tone = _tone(440.0, 1.0)
    padded = np.concatenate([np.zeros(2 * SR), tone, np.zeros(2 * SR)])
    a, b = analyze.active_span(_stereo(padded), SR)
    assert abs(a - 2 * SR) < 100
    assert abs(b - 3 * SR) < 100


def test_leading_silence_does_not_make_a_stem_read_quiet():
    """A vocal that enters late must not be gain-staged as if it were quiet."""
    tone = _stereo(_tone(440.0, 2.0))
    late = _stereo(np.concatenate([np.zeros(6 * SR), _tone(440.0, 2.0)]))

    plain = analyze.measure(_stem("v.wav", tone))
    delayed = analyze.measure(_stem("v2.wav", late))

    assert abs(plain["lufs"] - delayed["lufs"]) < 0.2
    assert abs(plain["crest_db"] - delayed["crest_db"]) < 0.2
    assert delayed["silence_ratio"] > 0.7


def test_padding_does_not_change_crest():
    """Aligning to the longest stem must not fake dynamic range."""
    tone = _stereo(_tone(220.0, 1.0))
    padded = np.vstack([tone, np.zeros((3 * SR, 2), dtype=np.float32)])
    a = analyze.measure(_stem("a.wav", tone))
    b = analyze.measure(_stem("b.wav", padded))
    assert abs(a["crest_db"] - b["crest_db"]) < 0.2


def test_short_one_shot_uses_momentary_loudness():
    """150 ms has no 400 ms gating block; it is short, not silent."""
    short = _stereo(_tone(1000.0, 0.15, 0.3))
    m = analyze.measure(_stem("chirp.wav", short))
    assert m["silent"] is False
    assert m["lufs"] > -20.0, m["lufs"]
    # a 0.3-amplitude tone sits near -13 LUFS, not near the -70 silence floor
    assert abs(m["lufs"] - dsp.momentary_lufs(short, SR)) < 0.01


def test_silent_stem_is_flagged_and_dropped_from_the_plan():
    stems = [
        _stem("kick.wav", _stereo(_tone(60.0, 1.0)), role="kick"),
        _stem("room.wav", np.zeros((SR, 2), dtype=np.float32)),
    ]
    for s in stems:
        analyze.measure(s)
    assert sum(1 for s in stems if s.metrics["silent"]) == 1

    preset = planner.load_preset("pop_2020s")
    p = planner.build_plan(stems, preset)
    assert [t["id"] for t in p["tracks"]] == ["kick"]
    assert any("silent" in w for w in p["warnings"])


def test_all_silent_session_is_an_error_not_a_silent_file():
    stems = [_stem("a.wav", np.zeros((SR, 2), dtype=np.float32))]
    for s in stems:
        analyze.measure(s)
    with pytest.raises(planner.PlanError):
        planner.build_plan(stems, planner.load_preset("lofi"))


def test_clipped_source_is_reported():
    hot = np.clip(_tone(330.0, 1.0, amp=3.0), -1, 1)
    stems = [_stem("gtr.wav", _stereo(hot), role="guitar_electric")]
    for s in stems:
        analyze.measure(s)
    assert stems[0].metrics["clipped_ratio"] > 0.4

    p = planner.build_plan(stems, planner.load_preset("rock"))
    assert any("clipped" in w for w in p["warnings"])


def test_anti_phase_stem_is_reported():
    tone = _tone(196.0, 1.0)
    stems = [_stem("pad.wav", np.stack([tone, -tone], axis=1), role="pad")]
    for s in stems:
        analyze.measure(s)
    assert stems[0].metrics["correlation"] < -0.9

    p = planner.build_plan(stems, planner.load_preset("cinematic"))
    assert any("out of phase" in w for w in p["warnings"])


def test_anti_phase_stem_still_gets_a_real_spectrum():
    """The mono sum of an anti-phase stem is zero; its spectrum must not be."""
    tone = _tone(196.0, 1.0)
    m = analyze.measure(_stem("pad.wav", np.stack([tone, -tone], axis=1), role="pad"))
    assert m["centroid_hz"] > 100.0


# --------------------------------------------------------------------- staging

def test_gain_clamp_reports_the_applied_sign():
    """A hot stem must be shown as a negative clamp, not a positive one."""
    hot = _stereo(_tone(330.0, 2.0, amp=0.95))
    quiet = _stereo(_tone(60.0, 2.0, amp=0.02))
    stems = [
        _stem("gtr.wav", hot, role="guitar_electric"),
        _stem("kick.wav", quiet, role="kick"),
    ]
    for s in stems:
        analyze.measure(s)
    p = planner.build_plan(
        stems, planner.load_preset("pop_2020s"), balance_preserve=0.0
    )
    clamped = [w for w in p["warnings"] if "gain clamped" in w]
    assert clamped, p["warnings"]
    for w in clamped:
        applied = float(w.split("clamped to")[1].split("dB")[0])
        wanted = float(w.split("wanted")[1].split("dB")[0])
        # the message must print what was actually applied: the full budget,
        # with the sign of the move that was requested
        assert abs(abs(applied) - planner.MAX_STAGE_DB) < 1e-6, w
        assert applied * wanted > 0, w
        assert abs(wanted) > abs(applied), w


def test_gain_staging_budget_is_respected():
    stems = [_stem("x.wav", _stereo(_tone(200.0, 2.0, amp=0.9)), role="other")]
    for s in stems:
        analyze.measure(s)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    for t in p["tracks"]:
        assert abs(t["gain_db"]) <= planner.MAX_STAGE_DB + 1e-9


# ------------------------------------------------------------------- numerics

def test_float32_chain_stays_float32():
    """Mixing buffers are float32; a filter must not silently promote them."""
    x = _stereo(_tone(440.0, 1.0))
    assert x.dtype == np.float32
    y = dsp.apply_eq(x, SR, [("highpass", 90.0, 0.0, 0.707)])
    assert y.dtype == np.float32
    assert dsp.compress(y, SR, 0.5).dtype == np.float32
    assert dsp.true_peak_limit(y, SR, -1.0).dtype == np.float32
    assert dsp.reverb_send(y, SR, 0.3, 0.5, 5.0, 8000.0).dtype == np.float32


def test_float32_does_not_change_the_meters():
    """Downcasting storage must not move the loudness or peak readings."""
    x64 = np.stack([_tone(997.0, 2.0)] * 2, axis=1)
    x32 = x64.astype(np.float32)
    assert abs(dsp.loudness_lufs(x64, SR) - dsp.loudness_lufs(x32, SR)) < 0.01
    assert abs(dsp.true_peak_db(x64, SR) - dsp.true_peak_db(x32, SR)) < 0.01


def test_block_wise_metrics_match_on_a_long_signal():
    """Chunking must not change the answer: compare 1 s blocks with 8 s blocks."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((SR * 20, 2)) * 0.3).astype(np.float32)
    one = dsp.true_peak_db(x, SR, block_s=8.0)
    small = dsp.true_peak_db(x, SR, block_s=1.0)
    assert abs(one - small) < 0.1, (one, small)

    y1 = dsp.true_peak_limit(x, SR, -1.0, block_s=8.0)
    y2 = dsp.true_peak_limit(x, SR, -1.0, block_s=1.0)
    assert dsp.true_peak_db(y1, SR) <= -1.0
    assert dsp.true_peak_db(y2, SR) <= -1.0
    # the two block sizes should give audibly the same result
    diff = np.max(np.abs(y1 - y2))
    assert diff < 0.05, diff


def test_memory_estimate_and_budget():
    stems = [
        _stem(f"{i}.wav", np.zeros((SR * 10, 2), dtype=np.float32)) for i in range(4)
    ]
    need = analyze.estimate_memory_mb(stems)
    assert need > 0
    analyze.check_memory(stems, budget_mb=need * 2)  # fits: no raise
    with pytest.raises(analyze.StemError):
        analyze.check_memory(stems, budget_mb=need / 2)


# ------------------------------------------------------------------ end to end

def test_torture_session_renders_and_hits_target():
    """The whole nasty set in one go: it must render, and land on target."""
    stems = [
        _stem("kick.wav", _stereo(_tone(60.0, 2.0, 0.6) * np.exp(-np.arange(2 * SR) / SR)), role="kick"),
        _stem("bass.wav", _stereo(_tone(55.0, 2.0, 0.4)), role="bass"),
        _stem("vocal_lead.wav", _stereo(_tone(440.0, 2.0, 0.3)), role="lead_vocal"),
        _stem("Audio 1.wav", _stereo(_tone(330.0, 2.0, 0.3))),
        _stem("room.wav", np.zeros((SR, 2), dtype=np.float32)),
        _stem("pad.wav", np.stack([_tone(196.0, 2.0), -_tone(196.0, 2.0)], axis=1), role="pad"),
        _stem("hot.wav", _stereo(np.clip(_tone(523.0, 2.0, 2.0), -1, 1)), role="synth"),
    ]
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    preset = planner.load_preset("pop_2020s")
    p = planner.build_plan(stems, preset)
    mix, report = render.render(stems, p, {"decay_s": 0.3, "predelay_ms": 0.0})

    assert abs(report["lufs"] - p["buses"]["master"]["target_lufs"]) < 0.6
    assert report["true_peak_db"] <= p["buses"]["master"]["true_peak_db"] + 1e-6
    assert report["crest_db"] > 4.0
    assert len(p["tracks"]) == 6  # the silent stem is gone

def test_crest_note_reports_reference_gap():
    """A large gap from the style reference is reported without a quality verdict."""
    from automix import pipeline

    preset = {"crest_target_db": 12.0}
    plan = {"warnings": []}
    report = {"crest_db": 6.0, "target_lufs": -9.5}
    pipeline._note_crest(plan, preset, report)
    assert any("below the" in w for w in plan["warnings"])

    # and it stays quiet near the reference or without a crest reference
    quiet = {"warnings": []}
    pipeline._note_crest(quiet, preset, {"crest_db": 11.0, "target_lufs": -14.0})
    assert quiet["warnings"] == []
    pipeline._note_crest(quiet, {}, report)
    assert quiet["warnings"] == []


def test_crest_note_does_not_blame_the_target_for_a_reference_gap():
    from copy import deepcopy
    from automix import pipeline

    # R78 full-cue measurements: raw crest 9.48689017558 dB, output
    # 9.32871853763 dB, style reference 15 dB. The 5.7 dB reference gap
    # is not a measurement of the raw-to-output dynamics change.
    preset = {"crest_target_db": 15.0}
    notes = []
    for target in (-17.0, -11.0):
        plan = {"warnings": ["existing note"], "style": {"id": "ambient_space"},
                "buses": {"master": {"target_lufs": target}}}
        report = {"crest_db": 9.32871853763, "target_lufs": target,
                  "true_peak_db": -8.0}
        original_plan, original_report = deepcopy(plan), deepcopy(report)
        pipeline._note_crest(plan, preset, report)
        assert report == original_report
        assert {k: v for k, v in plan.items() if k != "warnings"} == {
            k: v for k, v in original_plan.items() if k != "warnings"}
        assert plan["warnings"][:1] == original_plan["warnings"]
        assert len(plan["warnings"]) == 2
        notes.append(plan["warnings"][-1])
    assert notes[0] == notes[1]
    warning = notes[0]
    assert "output crest 9.3 dB" in warning
    assert "5.7 dB below the 15.0 dB style reference" in warning
    assert "does not measure a loss of dynamics or establish its cause" in warning
    assert "equal loudness" in warning
    assert "LUFS" not in warning
    assert not any(claim in warning.lower() for claim in (
        "hitting", "costs", "punch", "lower the loudness target"))

def _tone_stem(name, freq, dur=2.0, amp=0.5, decay=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = amp * np.sin(2 * np.pi * freq * t)
    if decay:
        sig = sig * np.exp(-t * decay)
    st = analyze.Stem(name=name, path=name, audio=np.stack([sig, sig], 1).astype(np.float32), sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def test_low_end_carve_when_kick_and_bass_share_a_fundamental():
    stems = [
        _tone_stem("kick.wav", 60.0, decay=8.0),
        _tone_stem("bass.wav", 60.0),
        _tone_stem("vocal_lead.wav", 440.0),
    ]
    assert {s.role for s in stems} >= {"kick", "bass"}
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))

    bass = next(t for t in p["tracks"] if t["role"] == "bass")
    carve = [b for b in bass.get("eq", []) if b["gain_db"] < 0 and b["freq"] < 120]
    assert carve, bass
    assert abs(carve[0]["freq"] - 60.0) < 12.0
    assert carve[0]["gain_db"] <= -3.0
    assert any("room for the kick" in w for w in p["warnings"])
    assert any("room for the kick" in e for e in bass["evidence"])


def test_low_end_carve_is_gentler_when_the_fundamentals_differ():
    stems = [_tone_stem("kick.wav", 60.0, decay=8.0), _tone_stem("bass.wav", 90.0)]
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    bass = next(t for t in p["tracks"] if t["role"] == "bass")
    carve = [b for b in bass.get("eq", []) if b["freq"] < 120]
    assert carve
    assert -3.0 < carve[0]["gain_db"] < 0.0


def test_no_carve_without_a_kick():
    stems = [_tone_stem("bass.wav", 60.0), _tone_stem("vocal_lead.wav", 440.0)]
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    bass = next(t for t in p["tracks"] if t["role"] == "bass")
    assert not [b for b in bass.get("eq", []) if b["freq"] < 120]


def test_low_peak_is_zero_when_there_is_no_low_content():
    m = analyze.measure(_tone_stem("vocal_lead.wav", 440.0))
    assert m["low_peak_hz"] == 0.0

# ------------------------------------------------------------------- riding

def _dynamic_stem(name, role, sr=SR, dur=8.0, loud_at=4.0):
    """A sustained source that is quiet for the first half, loud after."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.where(t < loud_at, 0.08, 0.35)
    sig = env * np.sin(2 * np.pi * 440.0 * t)
    st = analyze.Stem(name=name, path=name, audio=np.stack([sig, sig], 1).astype(np.float32), sr=sr)
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def test_ride_curve_follows_sections_and_is_bounded():
    st = _dynamic_stem("vocal_lead.wav", "lead_vocal")
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    ride = p["tracks"][0]["ride"]

    assert 2 <= len(ride) <= 120
    assert all(abs(g) <= planner.RIDE_MAX_DB + 1e-6 for _, g in ride)
    # the quiet verse must be lifted and the loud chorus pulled back
    verse = [g for t, g in ride if t < 3.0]
    chorus = [g for t, g in ride if t > 5.5]
    assert verse and chorus
    assert max(verse) > 1.0
    assert min(chorus) < -1.0


def test_drums_are_not_ridden():
    st = _dynamic_stem("drums_kick.wav", "kick")
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert "ride" not in p["tracks"][0]


def test_ride_narrows_the_short_term_level_spread():
    from automix import dsp, render

    st = _dynamic_stem("vocal_lead.wav", "lead_vocal")
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))

    def spread(x):
        _, db = dsp.short_term_db(x, st.sr)
        active = db[db > db.max() - 40.0]
        return float(np.percentile(active, 90) - np.percentile(active, 10))

    mix, _ = render.render([st], p, {"decay_s": 0.2, "predelay_ms": 0.0})
    assert spread(mix) < spread(st.audio) - 5.0


def test_ride_can_be_switched_off_by_intensity():
    st = _dynamic_stem("vocal_lead.wav", "lead_vocal")
    p = planner.build_plan([st], planner.load_preset("pop_2020s"), intensity=0.0)
    assert "ride" not in p["tracks"][0]

# ------------------------------------------------------- multi-track context

def _synth(name, freq, dur=3.0, amp=0.4, noise=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + 0.3 * np.random.default_rng(1).standard_normal(n)
    return analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
    )


def test_context_promotes_a_lone_low_track_to_bass():
    """A single track cannot know it is the bass; the set can."""
    stems = [_synth("Audio 1.wav", 120.0)]
    analyze.detect_roles(stems)
    assert stems[0].role == "bass"
    assert stems[0].role_source == "spectral+context"


def test_context_does_not_overwrite_an_existing_bass():
    stems = [_synth("Audio 1.wav", 120.0), _synth("bass.wav", 55.0)]
    analyze.detect_roles(stems)
    roles = {s.name: (s.role, s.role_source) for s in stems}
    assert roles["bass.wav"] == ("bass", "keyword")
    assert roles["Audio 1.wav"][0] == "other"


def test_context_does_not_promote_when_the_slot_is_taken():
    stems = [_synth("Audio 1.wav", 9000.0, amp=0.2), _synth("hats.wav", 9000.0, amp=0.2)]
    analyze.detect_roles(stems)
    roles = {s.name: s.role for s in stems}
    assert roles["hats.wav"] == "hats"
    assert roles["Audio 1.wav"] != "hats"


def test_context_never_downgrades_a_confident_role():
    stems = [
        _synth("kick.wav", 60.0),
        _synth("bass.wav", 55.0),
        _synth("vocal_lead.wav", 440.0),
    ]
    analyze.detect_roles(stems)
    assert {s.role for s in stems} >= {"kick", "bass", "lead_vocal"}

# ------------------------------------------------------------------ performance

def test_section_envelope_is_linear_not_quadratic():
    """Guard the exact bug I introduced: a 24000-wide np.convolve is O(n*w).

    That turned a 36 s render from 3 s into 23 s.  A one-minute envelope has to
    cost well under a second; the direct convolution took ~10 s.
    """
    import time as _time

    from automix import render as rnd

    n = SR * 60
    sections = [
        {"start_s": 0.0, "end_s": 30.0, "width_scale": 0.97},
        {"start_s": 30.0, "end_s": 60.0, "width_scale": 1.04},
    ]
    t0 = _time.perf_counter()
    env = rnd._section_env(sections, n, SR, "width_scale")
    elapsed = _time.perf_counter() - t0
    assert env is not None and env.size == n
    assert elapsed < 1.0, f"envelope took {elapsed:.2f}s for 60 s of audio"


def test_render_cost_scales_with_length():
    """Render time must grow roughly with length, not with its square."""
    import time as _time

    def render_seconds(dur):
        stems = [
            _stem("kick.wav", _stereo(_tone(60.0, dur, 0.5) * np.exp(-np.arange(int(dur * SR)) / SR)), role="kick"),
            _stem("bass.wav", _stereo(_tone(55.0, dur, 0.4)), role="bass"),
            _stem("vocal_lead.wav", _stereo(_tone(440.0, dur, 0.3)), role="lead_vocal"),
        ]
        analyze.align(stems)
        for s in stems:
            analyze.measure(s)
        p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
        t0 = _time.perf_counter()
        render.render(stems, p, {"decay_s": 0.3, "predelay_ms": 0.0})
        return _time.perf_counter() - t0

    short = render_seconds(4.0)
    long = render_seconds(16.0)  # 4x the audio
    # 4x the data should be well under 8x the time; quadratic would be ~16x
    assert long < short * 8.0 + 1.0, (short, long)

# ------------------------------------------------------- vocal masking carve

def _harm(name, parts, dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in parts)
    st = analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def _vocal():
    return _harm("vocal_lead.wav", [(220, 0.3), (660, 0.1), (1320, 0.05), (2640, 0.03)])


def _guitar(presence_amp):
    return _harm(
        "guitar_elec.wav",
        [(110, 0.35), (220, 0.2), (660, 0.1), (1320, 0.06), (3300, presence_amp)],
    )


def test_instrument_competing_with_the_vocal_is_carved():
    p = planner.build_plan([_guitar(0.12), _vocal()], planner.load_preset("pop_2020s"))
    guitar = next(t for t in p["tracks"] if t["role"] == "guitar_electric")
    dynamic = guitar.get("dynamic_eq")
    assert dynamic and dynamic["band"]["freq_hz"] == 3200.0
    assert dynamic["band"]["gain_db"] < 0
    assert any("vocal masking" in e for e in guitar["evidence"])


def test_a_quiet_instrument_is_left_alone():
    p = planner.build_plan([_guitar(0.002), _vocal()], planner.load_preset("pop_2020s"))
    guitar = next(t for t in p["tracks"] if t["role"] == "guitar_electric")
    assert not [e for e in guitar.get("evidence", []) if "vocal masking" in e]


def test_no_vocal_means_no_masking_carve():
    p = planner.build_plan([_guitar(0.12)], planner.load_preset("pop_2020s"))
    guitar = p["tracks"][0]
    assert not [e for e in guitar.get("evidence", []) if "vocal masking" in e]


def test_masking_dip_is_audible_at_its_processing_stage():
    guitar, vocal = _guitar(0.12), _vocal()
    p = planner.build_plan([guitar, vocal], planner.load_preset("pop_2020s"))
    track = next(t for t in p["tracks"] if t["role"] == "guitar_electric")
    processed, report = render._dynamic_eq(guitar.audio, SR, track["dynamic_eq"])
    # Inspect a stable active window at this exact step. Compression and the
    # later whole-track matching must not be mistaken for the EQ's effect.
    def presence(x):
        x = x[SR:2*SR].mean(axis=1)
        frequency = np.fft.rfftfreq(len(x), 1/SR)
        return float(np.sum(np.abs(np.fft.rfft(x))**2 * ((frequency>=2900)&(frequency<=3500))))
    assert presence(processed) < presence(guitar.audio)*.98
    assert report["max_nominal_cut_db"] <= 1.5


def test_a_dark_vocal_does_not_carve_instruments():
    """If the voice has nothing in the presence band, there is nothing to protect."""
    dark = _harm("vocal_lead.wav", [(220, 0.3), (660, 0.1)])
    p = planner.build_plan([_guitar(0.12), dark], planner.load_preset("pop_2020s"))
    guitar = next(t for t in p["tracks"] if t["role"] == "guitar_electric")
    assert not [e for e in guitar.get("evidence", []) if "vocal masking" in e]


def test_masking_skips_a_band_that_already_exists():
    """Do not stack a second dip on top of the style EQ's own bell."""
    # This source needs an ordinary corrective presence cut already.
    p = planner.build_plan([_guitar(0.5), _vocal()], planner.load_preset("pop_2020s"))
    guitar = next(t for t in p["tracks"] if t["role"] == "guitar_electric")
    nearby = [b for b in guitar.get("eq", []) if abs(b["freq"] - 3200.0) < 600.0]
    assert len(nearby) == 1
    assert "dynamic_eq" not in guitar

# ------------------------------------------------------------- the receipt

def _rcpt_stem(parts=None):
    """A stem for receipt tests; ``parts`` is [(amp, seconds), ...]."""
    parts = parts or [(0.12, 4.0)]
    chunks = []
    for amp, dur in parts:
        n = int(dur * SR)
        t = np.arange(n) / SR
        chunks.append(amp * np.sin(2 * np.pi * 220.0 * t))
    x = np.concatenate(chunks)
    st = analyze.Stem(
        name="pad.wav", path="p",
        audio=np.stack([x, np.roll(x, 5)], 1).astype(np.float32), sr=SR,
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def _report_text(plan, report=None, stems=None):
    import tempfile

    report = report or {
        "lufs": -10.5, "target_lufs": -10.5, "true_peak_db": -1.2, "peak_db": -1.3,
        "crest_db": 10.0, "correlation": 0.9, "width": 0.1, "loudness_passes": 2,
        "ceiling_db": -1.0,
        "tracks": [
            {"id": t["id"], "role": t["role"], "bus": "music", "gain_db": t["gain_db"],
             "sends": 0.1}
            for t in plan["tracks"]
        ],
    }
    path = tempfile.mktemp(suffix=".md")
    render.write_report(path, plan, report, stems or [])
    return open(path, encoding="utf-8").read()


def test_receipt_lists_sections_when_present(monkeypatch):
    from automix import sections as secmod

    monkeypatch.setattr(secmod, "MIN_TOTAL_S", 6.0)
    monkeypatch.setattr(secmod, "MIN_SECTION_S", 2.0)
    monkeypatch.setattr(secmod, "HALF_WINDOW_S", 1.0)

    st = _rcpt_stem([(0.06, 4), (0.22, 4), (0.06, 4)])
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    text = _report_text(p)
    assert "## Sections" in text
    assert "chorus" in text


def test_receipt_shows_manual_trims():
    st = _rcpt_stem()
    p = planner.build_plan([st], planner.load_preset("pop_2020s"), trims={"pad": 1.2})
    text = _report_text(p)
    assert "## Manual overrides" in text
    assert "+1.2 dB" in text


def test_receipt_does_not_claim_only_one_thing_changes():
    """The old closing line over-claimed: master loudness compensation moves too."""
    st = _rcpt_stem()
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    text = _report_text(p)
    assert "only thing that changes" not in text
    assert "compensates" in text

# ------------------------------------------------------------- mains hum

def _hum_stem(name="guitar_elec.wav", hum=0.01, hum_hz=50.0, dur=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)
    x = 0.25 * env * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t))
    if hum:
        x = x + hum * np.sin(2 * np.pi * hum_hz * t)
    st = analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def test_hum_is_detected_only_when_it_survives_the_gaps():
    assert analyze.detect_hum(_hum_stem(hum=0.0).audio, SR)["hz"] == 0.0
    assert analyze.detect_hum(_hum_stem(hum=0.02).audio, SR)["hz"] == 50.0
    assert analyze.detect_hum(_hum_stem(hum=0.02, hum_hz=60.0).audio, SR)["hz"] == 60.0


def test_hum_notch_is_added_on_a_non_low_end_track():
    st = _hum_stem(hum=0.02)
    assert st.metrics["hum_hz"] == 50.0
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    notches = [b for b in p["tracks"][0].get("eq", []) if abs(b["freq"] - 50.0) < 2]
    assert notches and notches[0]["gain_db"] < 0
    assert any("hum notch" in e for e in p["tracks"][0]["evidence"])


def test_low_end_roles_are_exempt_from_the_hum_notch():
    """A sustained 50 Hz note in the bass is indistinguishable from hum."""
    st = _hum_stem(name="bass.wav", hum=0.02)
    st.role = "bass"
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert not [b for b in p["tracks"][0].get("eq", []) if abs(b["freq"] - 50.0) < 2]
    assert not any("hum notch" in e for e in p["tracks"][0]["evidence"])


def test_hum_notch_actually_removes_the_tone():
    st = _hum_stem(hum=0.02)
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))

    import copy

    def band(x, lo, hi):
        m = x.mean(axis=1)
        spec = np.abs(np.fft.rfft(m)) ** 2
        f = np.fft.rfftfreq(len(m), 1.0 / SR)
        return float(spec[(f >= lo) & (f <= hi)].sum())

    on, _ = render.render([st], p, {"decay_s": 0.3, "predelay_ms": 0.0})
    off_plan = copy.deepcopy(p)
    off_plan["tracks"][0]["eq"] = [
        b for b in off_plan["tracks"][0].get("eq", []) if abs(b["freq"] - 50.0) >= 2
    ]
    off, _ = render.render([st], off_plan, {"decay_s": 0.3, "predelay_ms": 0.0})

    assert 20 * np.log10(band(on, 49, 51) / band(off, 49, 51)) < -6.0

def test_a_noise_floor_is_not_mistaken_for_hum():
    """The realistic-stem bug: steady noise in the gaps looked like a hum line.

    A track with a noise floor and reverb-ish tails keeps energy in every bin
    during its quietest frames, which satisfied "still there in the gaps".  The
    tonality test (the hum bin must stand above its neighbours in the same
    frame) is what separates the two.
    """
    rng = np.random.default_rng(3)
    n = int(6.0 * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)
    body = 0.25 * env * (np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t))
    # a noise floor, and real reverb tails so the gaps are never silent
    from automix import dsp

    noise = rng.standard_normal(n) * 10 ** (-72 / 20)
    tail = dsp.reverb_send(
        np.stack([body, body], 1).astype(np.float32), SR, 0.3, 1.6, 20.0, 8000.0
    )[:, 0]
    dry = body + noise + tail
    assert analyze.detect_hum(np.stack([dry, dry], 1).astype(np.float32), SR)["hz"] == 0.0

    # the same material plus a real hum line must still be caught
    hummed = dry + 0.01 * np.sin(2 * np.pi * 50 * t)
    assert analyze.detect_hum(np.stack([hummed, hummed], 1).astype(np.float32), SR)["hz"] == 50.0


def test_moderate_sibilance_is_left_alone():
    """A vocal that is merely bright should not be de-essed."""
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    body = 0.25 * np.sin(2 * np.pi * 220.0 * t) + 0.12 * np.sin(2 * np.pi * 440.0 * t)
    rng = np.random.default_rng(0)
    sib = np.zeros(n)
    for start in np.arange(0.4, 3.6, 0.8):
        i = int(start * SR)
        d = int(0.12 * SR)
        sib[i : i + d] += rng.standard_normal(d) * np.hanning(d) * 0.25
    audio = np.stack([body + sib, body + sib], axis=1).astype(np.float32)

    st = analyze.Stem(name="vocal_lead.wav", path="v", audio=audio, sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    assert st.metrics["sibilance"] < planner.DEESS_THRESHOLD
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert "deess" not in p["tracks"][0]

# ---------------------------------------------------------- duplicate stems

def test_two_copies_of_one_part_are_flagged():
    base = _harm("keys.wav", [(220, 0.3), (440, 0.1), (880, 0.05)], dur=8.0)
    copy = analyze.Stem(
        name="keys_di.wav", path="k2", audio=base.audio * 0.5, sr=SR
    )
    analyze.detect_roles([copy])
    analyze.measure(copy)
    p = planner.build_plan([base, copy], planner.load_preset("pop_2020s"))
    assert any("carry the same performance" in w for w in p["warnings"])


def test_different_parts_are_not_flagged():
    a = _harm("keys.wav", [(220, 0.3), (440, 0.1)], dur=8.0)
    n = a.audio.shape[0]
    t = np.arange(n) / SR
    other = np.stack(
        [0.3 * np.sin(2 * np.pi * 330 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))] * 2,
        axis=1,
    ).astype(np.float32)
    b = analyze.Stem(name="pad.wav", path="p", audio=other, sr=SR)
    analyze.detect_roles([b])
    analyze.measure(b)
    p = planner.build_plan([a, b], planner.load_preset("pop_2020s"))
    assert not any("carry the same performance" in w for w in p["warnings"])

# ------------------------------------------------------- mixed sample rates

def test_align_resamples_and_keeps_the_pitch():
    """A 44.1 kHz stem in a 48 kHz session is 8.8% fast unless resampled."""
    from automix import dsp

    sr1, sr2 = 44100, 48000
    n1 = int(2 * sr1)
    t1 = np.arange(n1) / sr1
    a = analyze.Stem(
        name="bass.wav", path="b",
        audio=np.stack([0.3 * np.sin(2 * np.pi * 220 * t1)] * 2, 1).astype(np.float32),
        sr=sr1,
    )
    n2 = int(2 * sr2)
    t2 = np.arange(n2) / sr2
    b = analyze.Stem(
        name="vocal_lead.wav", path="v",
        audio=np.stack([0.3 * np.sin(2 * np.pi * 440 * t2)] * 2, 1).astype(np.float32),
        sr=sr2,
    )
    analyze.align([a, b])
    assert a.sr == sr2 and b.sr == sr2
    assert a.audio.shape[0] == b.audio.shape[0]
    assert getattr(a, "resampled_from", None) == sr1

    mono = a.audio.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    f = np.fft.rfftfreq(len(mono), 1 / a.sr)
    peak = float(f[int(np.argmax(spec))])
    assert abs(peak - 220.0) < 3.0, peak


def test_the_plan_says_which_stem_was_resampled():
    sr1 = 44100
    n1 = int(2 * sr1)
    t1 = np.arange(n1) / sr1
    a = analyze.Stem(
        name="bass.wav", path="b",
        audio=np.stack([0.3 * np.sin(2 * np.pi * 110 * t1)] * 2, 1).astype(np.float32),
        sr=sr1,
    )
    n2 = int(2 * 48000)
    t2 = np.arange(n2) / 48000
    b = analyze.Stem(
        name="vocal_lead.wav", path="v",
        audio=np.stack([0.3 * np.sin(2 * np.pi * 440 * t2)] * 2, 1).astype(np.float32),
        sr=48000,
    )
    stems = [a, b]
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert any("resampled" in w for w in p["warnings"])

# ---------------------------------------------------------- mix balance

def test_a_buried_vocal_is_called_out():
    from automix import pipeline
    from automix import plan as planner

    def track(name, f, amp):
        n = int(3 * SR)
        t = np.arange(n) / SR
        x = amp * np.sin(2 * np.pi * f * t)
        st = analyze.Stem(
            name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
        )
        analyze.detect_roles([st])
        analyze.measure(st)
        return st

    stems = [track("vocal_lead.wav", 440.0, 0.004), track("bass.wav", 55.0, 0.3),
             track("guitar_elec.wav", 220.0, 0.3)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan, _, report = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert any("vocal sits" in w for w in plan["warnings"])
    assert all(t.get("contrib_db") is not None for t in report["tracks"])


def test_a_balanced_vocal_is_not_called_out():
    from automix import pipeline
    from automix import plan as planner

    def track(name, f, amp):
        n = int(3 * SR)
        t = np.arange(n) / SR
        x = amp * np.sin(2 * np.pi * f * t)
        st = analyze.Stem(
            name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
        )
        analyze.detect_roles([st])
        analyze.measure(st)
        return st

    stems = [track("vocal_lead.wav", 440.0, 0.3), track("bass.wav", 55.0, 0.15),
             track("guitar_elec.wav", 220.0, 0.1)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert not any("vocal sits" in w for w in plan["warnings"])

# ------------------------------------------------------- mono compatibility

def _stereo_stem(name, left, right):
    return analyze.Stem(
        name=name, path=name, audio=np.stack([left, right], 1).astype(np.float32), sr=SR
    )


def _mix_one(stem):
    from automix import pipeline
    from automix import plan as planner

    stems = [stem]
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return pipeline.mix(stems, planner.load_preset("pop_2020s"))


def test_dual_mono_does_not_lose_level_in_mono():
    """BS.1770 sums channels; the 3 dB constant must be accounted for."""
    n = 3 * SR
    t = np.arange(n) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    plan, _, report = _mix_one(_stereo_stem("pad.wav", tone, tone))
    assert abs(report["mono_loss_db"]) < 1.0
    assert not any("mono fold-down" in w for w in plan["warnings"])


def test_anti_phase_material_is_called_out():
    n = 3 * SR
    t = np.arange(n) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    plan, _, report = _mix_one(_stereo_stem("pad.wav", tone, -tone))
    assert report["mono_loss_db"] > 10.0
    assert any("mono fold-down" in w for w in plan["warnings"])

def test_a_truncated_stem_is_flagged():
    """A stem that stops halfway is almost always a cut-short export."""
    def tone(name, dur, freq):
        n = int(dur * SR)
        t = np.arange(n) / SR
        x = 0.3 * np.sin(2 * np.pi * freq * t)
        st = analyze.Stem(
            name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
        )
        analyze.detect_roles([st])
        analyze.measure(st)
        return st

    stems = [tone("bass.wav", 60.0, 55.0), tone("vocal_lead.wav", 60.0, 440.0),
             tone("guitar_elec.wav", 12.0, 220.0)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert any("cut short" in w and "guitar_elec" in w for w in p["warnings"])


def test_a_stem_that_starts_late_is_not_flagged():
    """Late entry is an arrangement, not a broken export."""
    def tone(name, dur, freq, lead_in=0.0):
        n = int(dur * SR)
        t = np.arange(n) / SR
        x = 0.3 * np.sin(2 * np.pi * freq * t)
        x[: int(lead_in * SR)] = 0.0
        st = analyze.Stem(
            name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
        )
        analyze.detect_roles([st])
        analyze.measure(st)
        return st

    stems = [tone("bass.wav", 40.0, 55.0), tone("vocal_lead.wav", 40.0, 440.0, lead_in=18.0)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert not any("cut short" in w for w in p["warnings"])

def test_at_most_two_instruments_are_ducked_for_the_vocal():
    """Twenty stems used to duck five instruments; the presence band got dull."""
    vocal = _harm("vocal_lead.wav", [(220, 0.3), (660, 0.1), (2640, 0.04)])

    def competitor(name, presence_amp):
        return _harm(
            name,
            [(110, 0.3), (220, 0.2), (660, 0.1), (1320, 0.05), (3300, presence_amp)],
        )

    stems = [
        vocal,
        competitor("guitar_l.wav", 0.14),
        competitor("guitar_r.wav", 0.12),
        competitor("keys.wav", 0.10),
        competitor("pad.wav", 0.08),
    ]
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    carved = [
        t["id"] for t in p["tracks"]
        if any("vocal masking" in e for e in t.get("evidence", []))
    ]
    assert len(carved) == planner.MAX_MASK_CARVES, carved
    # and they are the loudest competitors, not an arbitrary pair
    assert "guitar_l" in carved and "guitar_r" in carved

# ------------------------------------------------------------- pan placement

def _plan_for(stems):
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return planner.build_plan(stems, planner.load_preset("pop_2020s"))


def test_left_and_right_names_are_panned():
    stems = [
        _harm("guitar_l.wav", [(220, 0.3), (660, 0.1)]),
        _harm("guitar_r.wav", [(220, 0.3), (660, 0.1)]),
    ]
    p = _plan_for(stems)
    pans = {t["id"]: t["pan"] for t in p["tracks"]}
    assert pans["guitar_l"] < -0.3
    assert pans["guitar_r"] > 0.3


def test_same_role_tracks_are_spread_not_stacked():
    stems = [
        _harm("pad.wav", [(262, 0.25), (524, 0.1)]),
        _harm("keys.wav", [(330, 0.25), (660, 0.1)]),
    ]
    # force both to the same role by naming
    for st in stems:
        st.role = "pad"
    p = _plan_for(stems)
    pans = [t["pan"] for t in p["tracks"]]
    assert len(set(pans)) == 2, pans


def test_centre_roles_stay_centred_even_when_duplicated():
    stems = [
        _harm("bass.wav", [(55, 0.3), (110, 0.1)]),
        _harm("sub.wav", [(41, 0.3), (82, 0.1)]),
    ]
    for st in stems:
        st.role = "bass"
    p = _plan_for(stems)
    assert all(abs(t["pan"]) < 1e-6 for t in p["tracks"])

def test_different_rhythms_are_not_called_duplicates():
    """Trimming the lead-in used to align unrelated patterns.

    Both envelopes were measured from the *active span*, so every part started
    at its own first hit; a kick and a snare on the same grid then measured
    r=0.998 and were reported as the same performance.
    """
    def hits(first, period, freq):
        dur = 8.0
        n = int(dur * SR)
        t = np.arange(n) / SR
        gate = ((t - first) % period) < 0.08
        gate &= t >= first
        return np.stack([0.4 * np.sin(2 * np.pi * freq * t) * gate] * 2, axis=1).astype(np.float32)

    a = analyze.Stem(name="drums_kick.wav", path="k", audio=hits(0.35, 1.0, 80.0), sr=SR)
    b = analyze.Stem(name="drums_snare.wav", path="s", audio=hits(0.72, 1.0, 200.0), sr=SR)
    for st in (a, b):
        analyze.detect_roles([st])
        analyze.measure(st)

    corr = float(np.corrcoef(a.metrics["envelope"][:16], b.metrics["envelope"][:16])[0, 1])
    assert corr < 0.9, corr
    p = planner.build_plan([a, b], planner.load_preset("pop_2020s"))
    assert not any("same performance" in w for w in p["warnings"])

def _vocal_pair(bv_amp, lead_amp=0.25):
    def mk(name, amp):
        n = int(3 * SR)
        t = np.arange(n) / SR
        x = amp * np.sin(2 * np.pi * 440 * t)
        st = analyze.Stem(
            name=name, path=name, audio=np.stack([x, x], 1).astype(np.float32), sr=SR
        )
        st.role = "lead_vocal" if "lead" in name else "backing_vocal"
        st.role_source = "keyword"
        analyze.measure(st)
        return st

    return [mk("vocal_lead.wav", lead_amp), mk("vocal_bv.wav", bv_amp)]


def test_a_loud_backing_vocal_is_held_behind_the_lead():
    """Compare actual planned LUFS, rather than unrelated gain values."""
    stems = _vocal_pair(0.5)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    levels = {t["id"]: t["level_match"]["source_lufs"] + t["gain_db"] for t in p["tracks"]}
    assert levels["vocal_bv"] <= levels["vocal_lead"] - 3.99
    bv = next(t for t in p["tracks"] if t["id"] == "vocal_bv")
    assert any("below the lead vocal" in e for e in bv["evidence"])
    assert any("planned lead-minus-backing gap" in w for w in p["warnings"])


def test_a_backing_vocal_that_is_already_quiet_is_left_alone():
    stems = _vocal_pair(0.02)
    preset = planner.load_preset("pop_2020s")
    p = planner.build_plan(stems, preset, balance_preserve=1)
    bv = next(t for t in p["tracks"] if t["id"] == "vocal_bv")
    assert bv["gain_db"] == 0.0
    assert not any("below the lead vocal" in e for e in bv["evidence"])
    assert not any("behind the lead" in w for w in p["warnings"])


def test_the_preset_pre_delay_reaches_the_reverb():
    """Every style used to get the same 20 ms because nothing copied it."""
    from automix import dsp

    irs = {}
    for style in ("trap", "cinematic"):
        preset = planner.load_preset(style)
        wanted = float(preset["reverb"]["pre_delay_ms"])
        ir = dsp.make_ir(48000, 1.6, wanted, 9000.0)
        mag = np.abs(ir).max(axis=1)
        onset = int(np.argmax(mag > 0.02 * float(mag.max()))) / 48000 * 1000
        assert abs(onset - wanted) < 1.0, (style, onset, wanted)
        irs[style] = onset
    assert irs["cinematic"] > irs["trap"] + 10

def test_a_mostly_empty_stem_is_flagged():
    """Usually a mistaken export; silence_ratio existed but nothing read it."""
    n = int(20 * SR)
    x = np.zeros(n, dtype=np.float32)
    x[: int(2 * SR)] = 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(int(2 * SR)) / SR)
    stems = [analyze.Stem(name="pad.wav", path="p", audio=np.stack([x, x], 1), sr=SR)]
    analyze.detect_roles(stems)
    analyze.measure(stems[0])
    assert stems[0].metrics["silence_ratio"] > 0.8

    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert any("is silence" in w for w in p["warnings"]), p["warnings"]


def test_a_few_silent_gaps_are_not_flagged():
    n = int(12 * SR)
    t = np.arange(n) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (((t % 1.0) < 0.6))).astype(np.float32)
    stems = [analyze.Stem(name="guitar_elec.wav", path="g", audio=np.stack([x, x], 1), sr=SR)]
    analyze.detect_roles(stems)
    analyze.measure(stems[0])
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert not any("is silence" in w for w in p["warnings"]), p["warnings"]

def _reference_hum(seg, sr):
    """The obvious way to do it: one FFT per frame, per candidate.

    detect_hum batches its frame spectra instead.  This is the yardstick the
    fast path has to match -- same answers, not just "plausible" ones.
    """
    if seg.shape[0] < int(0.5 * sr):
        return {"hz": 0.0, "quiet_db": -120.0, "loud_db": -120.0}
    mono = seg.mean(axis=1).astype(np.float64)
    frame = int(0.25 * sr)
    hop = int(0.125 * sr)
    n_frames = 1 + (mono.shape[0] - frame) // hop
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    best = {"hz": 0.0, "quiet_db": -120.0, "loud_db": -120.0}
    for hz in (50.0, 60.0):
        idx = np.argmin(np.abs(freqs - hz))
        lo, hi = max(0, idx - 1), min(freqs.size, idx + 2)
        near = (freqs >= hz - 20.0) & (freqs <= hz + 20.0) & (np.abs(freqs - hz) > 3.0)
        hum = np.zeros(n_frames)
        total = np.zeros(n_frames)
        specs = []
        for i in range(n_frames):
            spec = np.abs(np.fft.rfft(mono[i * hop : i * hop + frame] * window)) ** 2
            specs.append(spec)
            hum[i] = float(spec[lo:hi].sum())
            total[i] = float(spec.sum()) + 1e-20
        quiet_idx = np.argsort(total)[: max(1, n_frames // 10)]
        ratios, audible = [], []
        for i in quiet_idx:
            spec = specs[i]
            bd = float(spec[lo:hi].mean())
            fl = float(np.median(spec[near])) if np.any(near) else 0.0
            if fl > 0:
                ratios.append(bd / fl)
            audible.append(10 * np.log10((bd or 1e-30) / (float(spec.max()) or 1e-30)))
            outside = np.concatenate([spec[max(0, lo - 3) : lo], spec[hi : hi + 3]])
            if outside.size and float(outside.max()) > 4.0 * max(bd, 1e-30):
                audible[-1] = -120.0
        tonality_db = float(10 * np.log10(np.median(ratios))) if ratios else 0.0
        audible_db = float(np.median(audible)) if audible else -120.0
        quiet_db = float(10 * np.log10(np.median(hum[quiet_idx]) + 1e-20))
        loud_db = float(10 * np.log10(np.percentile(hum, 90) + 1e-20))
        if (
            loud_db > -120.0
            and quiet_db > loud_db - 12.0
            and tonality_db >= 8.0
            and audible_db > -40.0
        ):
            if quiet_db > best["quiet_db"]:
                best = {"hz": hz, "quiet_db": quiet_db, "loud_db": loud_db}
    return best


def test_the_fast_hum_detector_matches_the_obvious_implementation():
    rng = np.random.default_rng(11)
    n = int(6 * SR)
    t = np.arange(n) / SR
    cases = {
        "hum+music": 0.2 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 220 * t),
        "music only": 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1310 * t),
        "noisy": rng.standard_normal(n) * 0.2,
        "silence": np.zeros(n),
    }
    for name, sig in cases.items():
        seg = np.stack([sig, sig], 1).astype(np.float32)
        want = _reference_hum(seg, SR)
        got = analyze.detect_hum(seg, SR)
        assert got["hz"] == want["hz"], (name, got, want)
        assert abs(got["quiet_db"] - want["quiet_db"]) < 0.01, (name, got, want)
        assert abs(got["loud_db"] - want["loud_db"]) < 0.01, (name, got, want)

def test_chinese_filenames_get_the_right_role():
    """The engine's users name their stems in their own language."""
    cases = {
        "主唱.wav": "lead_vocal",
        "和聲.wav": "backing_vocal",
        "底鼓.wav": "kick",
        "小鼓.wav": "snare",
        "踩鈸.wav": "hats",
        "貝斯.wav": "bass",
        "電鋼.wav": "keys",
        "鋼琴.wav": "piano",
        "吉他.wav": "guitar_electric",
        "弦樂.wav": "strings",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_simplified_chinese_names_work_too():
    for name, want in (("主唱.wav", "lead_vocal"), ("军鼓.wav", "snare"), ("电钢.wav", "keys")):
        assert analyze._keyword_role(name) == want, name


def test_a_mains_hum_does_not_turn_a_keyboard_into_a_kick():
    """The hum invented a low end that the music does not have."""
    n = int(8 * SR)
    t = np.arange(n) / SR
    tone = 0.25 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1318 * t)
    hum = 0.1 * np.sin(2 * np.pi * 50 * t)
    seg = np.stack([tone + hum, tone + hum], 1).astype(np.float32)
    assert analyze.detect_hum(seg, SR)["hz"] == 50.0
    assert analyze._spectral_role(seg, SR) != "kick"


def test_a_real_kick_is_still_a_kick():
    n = int(4 * SR)
    t = np.arange(n) / SR
    env = np.exp(-((t % 0.5) * 60))
    kick = (env * np.sin(2 * np.pi * 55 * t) * 0.8).astype(np.float32)
    seg = np.stack([kick, kick], 1)
    assert analyze._spectral_role(seg, SR) == "kick"

def _steady(name, hz, amp=0.3, secs=6.0):
    n = int(secs * SR)
    t = np.arange(n) / SR
    sig = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    st = analyze.Stem(name=name, path=name, audio=np.stack([sig, sig], 1), sr=SR)
    return st


def test_steady_tones_are_not_called_duplicates():
    """A bass and a kick both have a flat envelope; r=1.000 means nothing."""
    stems = [_steady("bass.wav", 82.4), _steady("drums_kick.wav", 55.0)]
    analyze.detect_roles(stems)
    for s in stems:
        analyze.measure(s)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert not any("same performance" in w for w in p["warnings"]), p["warnings"]


def test_a_real_duplicate_is_still_called_out():
    stem = _steady("keys.wav", 440.0)
    copy = _steady("keys_copy.wav", 440.0)
    stems = [stem, copy]
    analyze.detect_roles(stems)
    for s in stems:
        analyze.measure(s)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert any("same performance" in w for w in p["warnings"]), p["warnings"]

def test_the_gain_anchor_is_the_true_median():
    """With an even number of stems the two middle values are averaged.

    Indexing len//2 took the upper one, which pushed the anchor up to a few
    dB too loud and gave every quiet stem that much extra boost -- and more
    of them ran into the +/-18 dB clamp, which *does* change the balance.
    """
    preset = planner.load_preset("pop_2020s")
    levels = [-24.0, -20.0, -10.0, -9.0]

    class _S:
        def __init__(self, name, lufs):
            self.name = name
            self.role = "other"
            self.metrics = {"lufs": lufs}

    stems = [_S(f"s{i}.wav", lv) for i, lv in enumerate(levels)]
    warnings: list = []
    gains = planner._gain_stage(
        stems, preset, preset.get("guards", {}), warnings, balance_preserve=0.0
    )
    # With the style's balance alone, the implied anchor (gain + lufs -
    # style gain) must be one number for every stem: that is what "the anchor
    # is one value" means.  Indexing len//2 took the upper middle value, so
    # the anchor came out up to a few dB too loud and every quiet stem got
    # that much extra boost.
    style_gain = float(preset["role_gains_db"].get("other", -10.0))
    implied = {round(gains[s.name] + s.metrics["lufs"] - style_gain, 3) for s in stems}
    assert len(implied) == 1, implied
    # the style's gains are applied relative to their own centre, so the
    # anchor is the median minus that centre
    centre = style_gain  # every stem here has the same role
    assert implied.pop() == pytest.approx((-20.0 + -10.0) / 2 - centre, abs=0.01)

    # With the source's balance kept, the anchor still has to be the median:
    # the stems' own levels are added to it, not scattered around it.
    gains2 = planner._gain_stage(
        stems, preset, preset.get("guards", {}), warnings, balance_preserve=1.0
    )
    for s in stems:
        assert gains2[s.name] == pytest.approx(0.0, abs=0.01), gains2


def test_a_gain_that_would_break_the_clamp_is_clamped_and_said_so():
    """The +/-18 dB clamp changes the balance, so it is never silent."""
    preset = planner.load_preset("pop_2020s")

    class _S:
        def __init__(self, name, lufs):
            self.name = name
            self.role = "other"
            self.metrics = {"lufs": lufs}

    warnings: list = []
    stems = [_S(f"s{i}.wav", lv) for i, lv in enumerate([-40.0, -20.0, -10.0, -9.0])]
    gains = planner._gain_stage(
        stems, preset, preset.get("guards", {}), warnings, balance_preserve=0.0
    )
    assert gains["s0.wav"] <= 18.0
    assert any("clamped" in w for w in warnings), warnings

    # Keeping the source's own balance needs less staging, so the same input
    # no longer hits the clamp: a side effect worth having.
    warnings2: list = []
    gains2 = planner._gain_stage(
        stems, preset, preset.get("guards", {}), warnings2, balance_preserve=0.5
    )
    assert not any("clamped" in w for w in warnings2), warnings2
    assert gains2["s0.wav"] < gains["s0.wav"]

# ------------------------------------------------------------------ real-stem rules
# These mirror what was measured on 11 real MUSDB18 tracks (44 stems): the
# rules are about physical facts, not about that particular dataset, so
# synthetic stand-ins can lock them in.

def _spectral_stem(name, audio):
    """A stem whose name says nothing, so only the audio can decide."""
    st = analyze.Stem(name=name, path=name, audio=audio, sr=SR)
    analyze.detect_roles([st])
    return st


def test_a_sustained_band_limited_low_end_is_a_bass_not_a_kick():
    """A bass line has no content above 2 kHz; a kick has a beater click.

    Measured on real stems: 11 bass stems had presence+air <= 0.001 while
    every drum-kit stem had >= 0.005.  Before this rule every one of those
    basses was called a kick drum.
    """
    n = int(4 * SR)
    t = np.arange(n) / SR
    # a bass line: sustained, nothing above 2 kHz
    bass = 0.4 * np.sin(2 * np.pi * 55.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))
    # a kick: a low hit *with* a click on top
    click = 0.25 * np.exp(-t / 0.002) * np.sin(2 * np.pi * 4000.0 * t)
    body = 0.9 * np.exp(-t / 0.12) * np.sin(2 * np.pi * 60.0 * t)
    st_bass = _spectral_stem("low.wav", np.stack([bass, bass], 1).astype(np.float32))
    st_kick = _spectral_stem("low2.wav", np.stack([body + click, body + click], 1).astype(np.float32))
    assert st_bass.role == "bass", st_bass.role
    assert st_kick.role == "kick", st_kick.role


def test_a_centred_mid_band_voice_with_air_is_read_as_a_vocal():
    """A voice has sibilance well above its presence band and sits centred.

    Measured on real stems: the vocals had an air-to-presence ratio of at
    least 0.017 and an L/R correlation of 0.66-1.00, while the mixed
    accompaniment sat at a median ratio of 0.004.
    """
    n = int(4 * SR)
    t = np.arange(n) / SR
    # a voice-ish tone: 200-800 Hz fundamental, some 4-8 kHz sibilance, centred
    voice = (
        0.3 * np.sin(2 * np.pi * 220.0 * t)
        + 0.12 * np.sin(2 * np.pi * 700.0 * t)
        + 0.08 * np.sin(2 * np.pi * 3000.0 * t)
        + 0.05 * np.sin(2 * np.pi * 6500.0 * t)
    )
    centred = np.stack([voice, voice], 1).astype(np.float32)
    st = _spectral_stem("mystery.wav", centred)
    assert st.role == "lead_vocal", st.role

    # the same tone, hard panned, is an instrument, not a voice
    wide = np.stack([voice, np.zeros_like(voice)], 1).astype(np.float32)
    st_wide = _spectral_stem("mystery2.wav", wide)
    assert st_wide.role != "lead_vocal", st_wide.role

def test_a_hum_that_cannot_be_heard_is_not_a_hum():
    """A 60 Hz line 100 dB down is not worth notching.

    Measured on a real vocal stem (MUSDB18) that had no energy below 70 Hz
    worth speaking of: the 60 Hz bin sat 107 dB below the stem's peak and
    the detector still called it mains hum, and the plan notched it.
    """
    n = int(6 * SR)
    t = np.arange(n) / SR
    # a bright, centred voice-ish signal with a whisper of 60 Hz in it
    x = (
        0.3 * np.sin(2 * np.pi * 300.0 * t)
        + 0.1 * np.sin(2 * np.pi * 1200.0 * t)
        + 1e-5 * np.sin(2 * np.pi * 60.0 * t)
    )
    st = analyze.Stem(name="vocal.wav", path="vocal.wav", audio=np.stack([x, x], 1).astype(np.float32), sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    assert st.metrics.get("hum_hz", 0.0) == 0.0


def _bass_line(dur=6.0, freq=55.0, note_s=1.5, sr=SR):
    """A sparse bass line: a note every 1.5 s, nothing above 2 kHz.

    Sparse on purpose: a bass that plays once per bar has a huge crest
    factor, which is why the old "is the envelope peaky?" rule called every
    real bass stem a kick drum.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for start in np.arange(0, dur - note_s, note_s):
        i = int(start * sr)
        env[i : i + int(note_s * sr)] = 1.0
    env *= 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    x = 0.4 * env * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], 1).astype(np.float32)


def test_a_sparse_bass_line_is_a_bass_not_a_kick():
    """A bass that plays once per bar is still a bass.

    Measured on real stems: the bass stems' loud parts lasted 0.11-1.05 s
    and their loud centroid sat far below the drum kits', while the kits'
    loud parts lasted 0.05-0.13 s.  The old crest-factor rule called both
    of the two real basses it misread a kick drum.
    """
    st = _spectral_stem("line.wav", _bass_line())
    assert st.role == "bass", st.role


def test_a_single_held_tone_is_not_a_bass_line():
    """One held note is not a bass line -- and the context rule may say so.

    A single continuous tone has one loud part, not a series of notes, so
    the detector leaves it alone; the set-level rule can still call it the
    bass when nothing else is.
    """
    st = _spectral_stem("tone.wav", _bass_line(dur=6.0, note_s=6.0))
    assert st.role != "bass", st.role


def test_a_whole_kit_is_not_mistaken_for_a_single_kick():
    """A kit is short transients with a bright average; a kick is one hit.

    Measured on real stems: the kits' average centroid sat at 2560-7633 Hz
    (the basses at 187-757 Hz) and their loud parts lasted under 0.13 s.
    """
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for start in np.arange(0, 4.0, 0.5):  # eight hits
        i = int(start * SR)
        seg = np.zeros(n - i)
        st = np.arange(len(seg)) / SR
        seg += 0.6 * np.exp(-st / 0.04) * np.sin(2 * np.pi * 200.0 * st)
        seg += 0.25 * np.exp(-st / 0.01) * np.sin(2 * np.pi * 6000.0 * st)
        x[i:] += seg
    st = _spectral_stem("kit.wav", np.stack([x, x], 1).astype(np.float32))
    assert st.role in ("drums_full", "snare", "hats"), st.role


def test_a_track_id_survives_a_hostile_file_name():
    """A file name may legally hold a newline or a path separator.

    The id goes into the plan JSON, into file names on disk and into the
    UI's layout, so it has to be a single tidy token.  Letters of any
    script are kept: a Chinese file name is a normal thing to upload.
    """
    from automix.analyze import track_id

    assert track_id("take1.wav") == "take1"
    assert track_id("A Classic - take1.wav") == "A_Classic_-_take1"
    assert track_id("x\ny.wav") == "x_y"
    assert track_id("中文分軌.wav") == "中文分軌"
    assert "\n" not in track_id("bad\nname.wav")
    assert "/" not in track_id("dir/name.wav")
