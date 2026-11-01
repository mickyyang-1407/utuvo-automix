"""End-to-end and module tests for the automix MVP.

These are the tests that would have caught the bugs this MVP actually had:
a loudness meter that was 42 dB out, a limiter that flattened the mix, and a
reverb send that was 8 dB louder than the dry signal.  They use short synthetic
stems so the whole file runs in a couple of seconds.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 48000


# ------------------------------------------------------------------ fixtures

def _tone(freq: float, dur: float = 2.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1)


def _write(path: str, audio: np.ndarray, sr: int = SR) -> str:
    import wave

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture()
def session(tmp_path):
    """Four stems with recognisable names, so role detection is exercised."""
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    _write(str(d / "kick.wav"), _stereo(0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 8)))
    _write(str(d / "bass.wav"), _stereo(0.3 * np.sin(2 * np.pi * 55 * t)))
    _write(
        str(d / "vocal_lead.wav"),
        _stereo(0.25 * np.sin(2 * np.pi * 440 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))),
    )
    _write(str(d / "pad.wav"), _stereo(0.15 * np.sin(2 * np.pi * 220 * t)))
    stems = analyze.load_stems(str(d))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


# ------------------------------------------------------------------- loading

def test_load_and_detect_roles(session):
    roles = {s.name: s.role for s in session}
    assert roles["kick.wav"] == "kick"
    assert roles["bass.wav"] == "bass"
    assert roles["vocal_lead.wav"] == "lead_vocal"
    assert roles["pad.wav"] == "pad"


def test_mono_stem_becomes_stereo(tmp_path):
    import wave

    path = tmp_path / "mono.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.zeros(SR) ).astype("<i2").tobytes())
    audio, sr = analyze.load_stem_file(str(path))
    assert audio.shape == (SR, 2)
    assert sr == SR


def test_load_stems_rejects_empty_dir(tmp_path):
    with pytest.raises(analyze.StemError):
        analyze.load_stems(str(tmp_path))


def test_load_stems_resamples_mixed_rates(tmp_path):
    a = np.stack([_tone(440, 1.0, sr=44100)] * 2, axis=1)
    b = np.stack([_tone(220, 1.0, sr=48000)] * 2, axis=1)
    _write(str(tmp_path / "a.wav"), a, 44100)
    _write(str(tmp_path / "b.wav"), b, 48000)
    stems = analyze.load_stems(str(tmp_path))
    assert {s.sr for s in stems} == {48000}
    assert len({s.audio.shape[0] for s in stems}) == 1


# ------------------------------------------------------------------ metering

def test_true_peak_limit_respects_ceiling_and_keeps_dynamics():
    """The limiter must not turn music into a square wave."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SR, 2)) * 0.3
    x = x / np.max(np.abs(x)) * dsp.db_to_lin(6.0)  # 7 dB into the ceiling
    y = dsp.true_peak_limit(x, SR, -1.0)
    assert dsp.true_peak_db(y, SR) <= -1.0
    # hard clipping this material leaves ~4.6 dB of crest; a real limiter
    # should keep substantially more than that
    assert dsp.crest_db(y) > 7.0


def test_reverb_send_level_tracks_amount():
    x = _stereo(_tone(440.0, 2.0, 0.5))
    quiet = dsp.reverb_send(x, SR, 0.1, 1.0, 10.0, 8000.0)
    loud = dsp.reverb_send(x, SR, 0.4, 1.0, 10.0, 8000.0)
    r = lambda a: float(np.sqrt(np.mean(a ** 2)))
    q_ratio = r(quiet) / r(x)
    l_ratio = r(loud) / r(x)
    # a send is a fader: 4x the amount is 4x the wet level, exactly
    assert abs(l_ratio / q_ratio - 4.0) < 0.5
    # and the amount reads as a wet/dry ratio: a 0.4 send must not swamp the dry
    assert 0.2 < l_ratio < 0.8


# ---------------------------------------------------------------------- plan

def test_plan_is_valid_and_guarded(session):
    preset = planner.load_preset("pop_2020s")
    p = planner.build_plan(session, preset, preset_path="pop_2020s")
    planner.validate(p)
    assert len(p["tracks"]) == len(session)
    guards = preset["guards"]
    for t in p["tracks"]:
        assert -60.0 <= t["gain_db"] <= 18.0
        assert t["saturation"] <= guards["max_saturation"] + 1e-9
        assert t["width"] <= guards["max_width"] + 1e-9
        assert 20.0 <= t["hpf_hz"] <= 500.0
        for band in t["eq"]:
            if band.get("purpose") == "hum_notch":
                # a surgical repair, allowed past the tonal guard but not by much
                assert abs(band["gain_db"]) <= 12.0
                assert band["q"] >= 10.0
            else:
                assert abs(band["gain_db"]) <= guards["max_eq_db"] + 1e-9


def test_plan_records_evidence_for_every_track(session):
    preset = planner.load_preset("rock")
    p = planner.build_plan(session, preset, preset_path="rock")
    for t in p["tracks"]:
        assert t["evidence"], f"{t['id']} has no evidence"


def test_unknown_style_lists_alternatives():
    with pytest.raises(planner.PlanError) as exc:
        planner.load_preset("does_not_exist")
    assert "available" in str(exc.value)


def test_intensity_scales_compression(session):
    preset = planner.load_preset("pop_2020s")
    full = planner.build_plan(session, preset, intensity=1.0)
    none = planner.build_plan(session, preset, intensity=0.0)
    full_gr = sum(t.get("comp", {}).get("ratio", 1.0) for t in full["tracks"])
    none_gr = sum(t.get("comp", {}).get("ratio", 1.0) for t in none["tracks"])
    assert full_gr > none_gr


# -------------------------------------------------------------------- render

def test_render_hits_loudness_and_ceiling(session, tmp_path):
    preset = planner.load_preset("pop_2020s")
    p = planner.build_plan(session, preset, preset_path="pop_2020s")
    mix, report = render.render(session, p, {"decay_s": 0.4, "predelay_ms": 5.0})
    assert abs(report["lufs"] - p["buses"]["master"]["target_lufs"]) < 0.5
    assert report["true_peak_db"] <= p["buses"]["master"]["true_peak_db"] + 1e-6
    assert mix.shape == (max(s.audio.shape[0] for s in session), 2)
    # a mix must not be a square wave
    assert report["crest_db"] > 6.0

    out = tmp_path / "mix.wav"
    render.write_wav24(str(out), mix, SR)
    assert out.exists() and out.stat().st_size > 44
    back, sr = analyze.load_stem_file(str(out))
    assert sr == SR
    assert abs(float(np.max(np.abs(back))) - float(np.max(np.abs(mix)))) < 1e-3


def test_render_writes_plan_and_report(session, tmp_path):
    preset = planner.load_preset("lofi")
    p = planner.build_plan(session, preset, preset_path="lofi")
    mix, report = render.render(session, p, {"decay_s": 0.3, "predelay_ms": 0.0})
    plan_path = tmp_path / "mixplan.json"
    report_path = tmp_path / "report.md"
    planner.write_plan(p, str(plan_path))
    render.write_report(str(report_path), p, report, session)

    reloaded = json.loads(plan_path.read_text())
    assert reloaded["style"]["id"] == "lofi"
    assert len(reloaded["tracks"]) == len(session)
    text = report_path.read_text()
    assert "LUFS" in text and "## Tracks" in text


def test_cli_end_to_end(session, tmp_path):
    from automix import cli

    stems_dir = str(tmp_path / "stems")
    out = tmp_path / "out"
    rc = cli.main(
        ["--stems", stems_dir, "--style", "pop_2020s", "--out", str(out), "--quiet"]
    )
    assert rc == 0
    for name in ("mix.wav", "mixplan.json", "report.md"):
        assert (out / name).exists(), f"missing {name}"


def test_cli_reports_bad_style(session, tmp_path, capsys):
    from automix import cli

    rc = cli.main(
        [
            "--stems",
            str(tmp_path / "stems"),
            "--style",
            "nope",
            "--out",
            str(tmp_path / "o"),
            "--quiet",
        ]
    )
    assert rc == 2
    assert "unknown style" in capsys.readouterr().err

def test_vocal_forward_db_is_actually_used(session):
    """Every preset declared it; nothing read it.  Now lead gets it, doubles half."""
    preset = planner.load_preset("pop_2020s")
    assert preset["vocal_forward_db"] > 0

    with_fwd = planner.build_plan(session, preset, preset_path="pop_2020s", intensity=1.0)
    zeroed = dict(preset)
    zeroed["vocal_forward_db"] = 0.0
    without = planner.build_plan(session, zeroed, preset_path="pop_2020s", intensity=1.0)

    def gain(p, tid):
        return next(t["gain_db"] for t in p["tracks"] if t["id"] == tid)

    # The style's opinion on the balance is blended with the source's own
    # balance (balance_preserve), so at the default the vocal forward counts
    # (1 - balance_preserve) as much; at balance_preserve = 0 it counts fully.
    lead = gain(with_fwd, "vocal_lead") - gain(without, "vocal_lead")
    # at the default the source's own balance counts half, so the style's
    # vocal forward counts the other half
    k = float(with_fwd["style"].get("balance_preserve", 0.5))
    assert abs(lead - preset["vocal_forward_db"] * (1.0 - k)) < 0.01, (lead, k)

    style_only = planner.build_plan(
        session, preset, preset_path="pop_2020s", intensity=1.0, balance_preserve=0.0
    )
    lead_full = gain(style_only, "vocal_lead") - gain(
        planner.build_plan(
            session, zeroed, preset_path="pop_2020s", intensity=1.0, balance_preserve=0.0
        ),
        "vocal_lead",
    )
    assert abs(lead_full - preset["vocal_forward_db"]) < 0.01, lead_full

    # a backing vocal moves by half of it
    session_bv = session + []
    bv = analyze.Stem(
        name="vocal_bv.wav", path="v", audio=session[0].audio.copy(), sr=session[0].sr
    )
    bv.role, bv.role_source = "backing_vocal", "keyword"
    analyze.measure(bv)
    session_bv.append(bv)
    # at balance_preserve = 0 the style's balance is the only one that counts,
    # so the numbers are exact: the lead gets the full forward, the doubles
    # half of it
    a = planner.build_plan(session_bv, preset, intensity=1.0, balance_preserve=0.0)
    b = planner.build_plan(session_bv, zeroed, intensity=1.0, balance_preserve=0.0)
    bv_lead = next(t["gain_db"] for t in a["tracks"] if t["id"] == "vocal_bv") - next(
        t["gain_db"] for t in b["tracks"] if t["id"] == "vocal_bv"
    )
    assert abs(bv_lead - preset["vocal_forward_db"] * 0.5) < 0.01

    # and at balance_preserve = 1 the style has no say on the balance at all
    a1 = planner.build_plan(session_bv, preset, intensity=1.0, balance_preserve=1.0)
    b1 = planner.build_plan(session_bv, zeroed, intensity=1.0, balance_preserve=1.0)
    bv_1 = next(t["gain_db"] for t in a1["tracks"] if t["id"] == "vocal_bv") - next(
        t["gain_db"] for t in b1["tracks"] if t["id"] == "vocal_bv"
    )
    assert abs(bv_1) < 0.01, bv_1

    # and non-vocal tracks are untouched by it
    kick_delta = gain(with_fwd, "kick") - gain(without, "kick")
    assert abs(kick_delta) < 0.01

def test_an_unreachable_width_target_is_reported():
    """The guard is right; the receipt must not pretend it was reached."""
    from automix import pipeline

    warnings: list[str] = []
    plan = {"buses": {"master": {"width_target": 0.35}}}
    pipeline._note_width(plan, {"width": 0.20}, warnings)
    assert warnings and "could not be reached" in warnings[0]

    warnings.clear()
    pipeline._note_width(plan, {"width": 0.34}, warnings)
    assert not warnings

    warnings.clear()
    pipeline._note_width({"buses": {"master": {}}}, {"width": 0.1}, warnings)
    assert not warnings

def test_the_receipt_reports_how_wet_the_mix_is(session):
    """Reverb level is a decision; the receipt should show it."""
    import copy

    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    audio, report = render.render(session, plan, {"decay_s": 0.8, "predelay_ms": 0.0})
    assert report["wet_ratio_db"] is not None
    assert report["wet_ratio_db"] < 0.0  # always quieter than the dry sum

    dry = copy.deepcopy(plan)
    for t in dry["tracks"]:
        t["sends"] = {"plate": 0.0}
    _, dry_report = render.render(session, dry, {"decay_s": 0.8, "predelay_ms": 0.0})
    assert dry_report["wet_ratio_db"] is None


def test_more_send_means_more_wet(session):
    import copy

    quiet = planner.build_plan(session, planner.load_preset("pop_2020s"))
    loud = copy.deepcopy(quiet)
    for t in loud["tracks"]:
        sends = t.get("sends") or {}
        t["sends"] = {k: v * 3.0 for k, v in sends.items()} or {"plate": 0.3}

    _, a = render.render(session, quiet, {"decay_s": 0.8, "predelay_ms": 0.0})
    _, b = render.render(session, loud, {"decay_s": 0.8, "predelay_ms": 0.0})
    assert b["wet_ratio_db"] > a["wet_ratio_db"] + 3.0

def test_bus_compression_actually_compresses(session):
    """"Bus glue" that measures -0.00 dB of gain reduction is decoration.

    The thresholds used to come from a formula about levels rather than from
    the bus itself, so music and vocal buses never engaged.
    """
    from automix import render as rnd

    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    original = rnd._apply_comp
    reductions = []

    def spy(x, sr, spec):
        if spec:
            y = original(x, sr, spec)
            gr = 20 * np.log10(
                float(np.sqrt(np.mean(y.astype(float) ** 2)))
                / (float(np.sqrt(np.mean(x.astype(float) ** 2))) + 1e-12)
                + 1e-12
            )
            reductions.append(float(gr))
            return y
        return x

    rnd._apply_comp = spy
    try:
        rnd.render(session, plan, {})
    finally:
        rnd._apply_comp = original

    bus_reductions = reductions[-5:]  # five group buses are processed last
    assert min(bus_reductions) < -0.15, bus_reductions
    assert min(bus_reductions) > -6.0, bus_reductions  # glue, not squash

def test_rendering_the_same_plan_twice_is_bit_identical(session):
    """No unseeded randomness anywhere: the reverb IR must be deterministic.

    If it were not, every golden number would drift between runs and the
    tolerance would have to be loosened until it stopped catching anything.
    """
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    a, _ = render.render(session, plan, {})
    b, _ = render.render(session, plan, {})
    assert np.array_equal(a, b)


def test_two_plans_differ_only_in_their_timestamp(session):
    p1 = planner.build_plan(session, planner.load_preset("pop_2020s"))
    p2 = planner.build_plan(session, planner.load_preset("pop_2020s"))
    assert p1["created_at"] and p2["created_at"]
    p1 = dict(p1, created_at="x")
    p2 = dict(p2, created_at="x")
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)

def test_contradictory_ceilings_are_refused(session):
    """Two fields for one ceiling is a trap: editing the wrong one does nothing."""
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    plan["buses"]["master"]["true_peak_db"] = -3.0  # limiter still says -1.0
    with pytest.raises(planner.PlanError) as exc:
        planner.validate(plan)
    assert "disagree" in str(exc.value)

def test_the_receipt_leads_with_a_verdict_table(session):
    """A reader should see "did it land?" before 40 rows of numbers."""
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    _, report = render.render(session, plan, {"decay_s": 0.5, "predelay_ms": 0.0})
    verdicts = render._verdicts(plan, report)
    names = [v[0] for v in verdicts]
    assert "loudness" in names and "true peak" in names
    assert all(v[2] == "ok" for v in verdicts), verdicts

    import tempfile

    path = tempfile.mktemp(suffix=".md")
    render.write_report(
        path, plan,
        {**report, "tracks": [{"id": t["id"], "role": t["role"], "bus": "music",
                               "gain_db": t["gain_db"], "sends": 0.1} for t in plan["tracks"]]},
        session,
    )
    text = open(path, encoding="utf-8").read()
    assert "## Checks" in text
    assert "| loudness |" in text


def test_a_missed_target_shows_up_as_a_warning():
    plan = {"buses": {"master": {"target_lufs": -10.0, "true_peak_db": -1.0}},
            "style": {"crest_target_db": 9.0}}
    report = {"lufs": -13.0, "true_peak_db": -1.2, "wet_ratio_db": -20.0,
              "mono_loss_db": 0.5, "crest_db": 10.0}
    verdicts = {name: verdict for name, _, verdict in render._verdicts(plan, report)}
    assert verdicts["loudness"].startswith("warn")
    assert verdicts["true peak"] == "ok"

    report["mono_loss_db"] = 6.0
    verdicts = {name: verdict for name, _, verdict in render._verdicts(plan, report)}
    assert verdicts["mono fold"].startswith("warn")

    report["wet_ratio_db"] = -3.0
    verdicts = {name: verdict for name, _, verdict in render._verdicts(plan, report)}
    assert verdicts["reverb"].startswith("warn")

def test_stem_export_matches_the_reported_levels(session):
    """The files handed to a DAW must be the same signals the receipt describes."""
    collected: dict = {}
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    _, report = render.render(
        session, plan, {"decay_s": 0.5, "predelay_ms": 0.0}, collect=collected
    )
    levels = {t["id"]: t["contrib_db"] for t in report["tracks"]}
    stems = collected["stems"]
    assert set(stems) == set(levels)
    for tid, audio in stems.items():
        rms = 20 * np.log10(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-12)
        assert abs(rms - levels[tid]) < 0.05, (tid, rms, levels[tid])
    assert collected.get("reverb") is not None
    assert float(np.max(np.abs(collected["reverb"]))) > 0.0


def test_collect_is_optional_and_costs_nothing_when_off(session):
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    a, _ = render.render(session, plan, {"decay_s": 0.5, "predelay_ms": 0.0})
    b, _ = render.render(session, plan, {"decay_s": 0.5, "predelay_ms": 0.0}, collect={})
    assert np.array_equal(a, b)


def test_reference_second_pass_releases_first_mix(session, tmp_path, monkeypatch):
    """The measured baseline must not keep the first full render alive."""
    import weakref
    from automix import reference

    ref_path = _write(str(tmp_path / "reference.wav"), sum(s.audio for s in session) * .25)
    ref = reference.analyse(ref_path)
    original = render.render
    first_audio = None
    calls = 0

    def observed(*args, **kwargs):
        nonlocal first_audio, calls
        calls += 1
        if calls == 2:
            assert first_audio() is None
        output = original(*args, **kwargs)
        if calls == 1:
            first_audio = weakref.ref(output[0])
        return output

    monkeypatch.setattr(render, "render", observed)
    _, audio, report = pipeline.mix(session, planner.load_preset("pop_2020s"), reference=ref)
    assert calls == 2 and report["reference_passes"] == 2
    assert np.isfinite(audio).all()

def test_the_limiter_block_only_contains_things_that_do_something(session):
    """A field that looks like a control but is never read is worse than none.

    ``lookahead_ms`` and ``release_ms`` used to be written into every plan and
    read by nothing; the limiter's real lookahead is fixed in the DSP.
    """
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    limiter = plan["buses"]["master"]["limiter"]
    assert set(limiter) == {"ceiling_db"}, limiter
    assert limiter["ceiling_db"] == plan["buses"]["master"]["true_peak_db"]

def test_compressor_blocks_only_hold_fields_the_renderer_reads(session):
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    for t in plan["tracks"]:
        if "comp" in t:
            assert set(t["comp"]) <= {
                "threshold_db", "ratio", "ratio_law", "attack_ms", "release_ms", "makeup",
            }
    for bus in plan["buses"].values():
        if bus.get("comp"):
            assert set(bus["comp"]) <= {
                "threshold_db", "ratio", "ratio_law", "attack_ms", "release_ms", "threshold_offset_db",
                "makeup",
            }


def test_the_plan_says_how_the_make_up_is_computed(session):
    """The compressor adds make-up; the plan has to say how much.

    "makeup_db: 0" was untrue -- the DSP always recovered about half the
    average gain reduction -- so the plan now carries the *rule*, and the
    renderer reads it.  Without that, the plan's gain and the audio's level
    differ by an amount nobody could account for.
    """
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    with_comp = [t for t in plan["tracks"] if t.get("comp")]
    if not with_comp:
        pytest.skip("no compressor on this track")
    comp = with_comp[0]["comp"]
    assert comp["makeup"]["mode"] in ("half_average_reduction", "full_reduction", "none")
    assert comp["makeup"]["cap_db"] > 0

    # and the rule is honoured: with no make-up the stem ends up lower
    plan2 = json.loads(json.dumps(plan))
    for t in plan2["tracks"]:
        if t.get("comp"):
            t["comp"]["makeup"] = {"mode": "none", "cap_db": 12.0}
    _, report2 = render.render(session, plan2)
    _, report = render.render(session, plan)
    tid = with_comp[0]["id"]
    a = next(t["level_change_db"] for t in report["tracks"] if t["id"] == tid)
    b = next(t["level_change_db"] for t in report2["tracks"] if t["id"] == tid)
    assert b <= a + 0.01, (a, b)


def test_role_confidence_is_in_the_receipt(session):
    """The user has to be able to tell a guess from a certainty."""
    plan = planner.build_plan(session, planner.load_preset("pop_2020s"))
    for t in plan["tracks"]:
        assert any("confidence" in e for e in t["evidence"])

def test_the_plan_carries_the_reverb_shape(session):
    """The renderer can only build what the plan describes."""
    preset = planner.load_preset("lofi")
    p = planner.build_plan(session, preset)
    block = p["reverb"]
    assert block["high_cut_hz"] == preset["reverb"]["high_cut_hz"]
    assert block["decay_s"] == preset["reverb"]["decay_s"]
    assert block["pre_delay_ms"] == preset["reverb"]["pre_delay_ms"]


def test_a_styles_reverb_high_cut_reaches_the_audio(session):
    """Two styles with different cuts must not sound the same."""
    import numpy as np

    from automix import dsp

    def reverb_spectrum(style):
        preset = planner.load_preset(style)
        collect = {}
        pipeline.mix(session, preset, collect=collect)
        rev = collect.get("reverb")
        assert rev is not None and float(np.max(np.abs(rev))) > 1e-6
        mono = rev.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1 / 48000)
        total = spec.sum()
        return 10 * np.log10(spec[freqs > 5000].sum() / total)

    lofi = reverb_spectrum("lofi")      # high cut 5000 Hz
    bright = reverb_spectrum("cinematic")  # high cut 9000 Hz
    assert lofi < bright - 3.0, (lofi, bright)


def test_the_source_balance_can_be_kept_or_replaced(session):
    """balance_preserve decides whose balance wins.

    Measured on 11 real multitracks (MUSDB18): the stems there are post-fader,
    so they already carry the engineer's balance, and the style's absolute
    balance moved the accompaniment 14.4 dB away from it.  At
    balance_preserve = 1 the source's own levels are kept (every stem keeps
    its level, so the gain is 0); at 0 the style's balance is the only one
    that counts.
    """
    preset = planner.load_preset("pop_2020s")
    kept = planner.build_plan(session, preset, balance_preserve=1.0)
    style = planner.build_plan(session, preset, balance_preserve=0.0)

    # the plan says which way it went
    assert kept["style"]["balance_preserve"] == 1.0
    assert style["style"]["balance_preserve"] == 0.0

    def gains(p):
        return {t["id"]: t["gain_db"] for t in p["tracks"]}

    kept_g, style_g = gains(kept), gains(style)
    # keeping the source's balance means the stems are not moved at all
    for tid, g in kept_g.items():
        assert abs(g) < 0.01, (tid, g)
    # the style's balance does move them
    assert any(abs(g) > 0.5 for g in style_g.values()), style_g

    # and the two are different balances, not a rescaling of one another
    assert kept_g != style_g


def test_balance_preserve_is_clamped_and_reported(session):
    preset = planner.load_preset("pop_2020s")
    high = planner.build_plan(session, preset, balance_preserve=5.0)
    low = planner.build_plan(session, preset, balance_preserve=-3.0)
    assert high["style"]["balance_preserve"] == 1.0
    assert low["style"]["balance_preserve"] == 0.0


def test_the_level_chain_explains_the_level_change(session):
    """The plan promises a gain; the audio gets a different one.

    The compressor's make-up and the corrective EQ move the level too, so the
    plan's number and the audio's number do not add up on their own.  The
    renderer measures every step, so the receipt can show what happened --
    and the steps have to add up to the measured total, or the explanation is
    worse than none.
    """
    plan, audio, report = pipeline.mix(session, planner.load_preset("pop_2020s"))
    assert report["tracks"]
    for t in report["tracks"]:
        chain = t.get("chain")
        assert chain, t["id"]
        total = sum(c["delta_db"] for c in chain)
        assert abs(total - float(t["level_change_db"])) < 0.15, (t["id"], total, t["level_change_db"])
        # the first step is the gain the plan promised
        assert chain[0]["step"] == "gain"
        assert abs(chain[0]["delta_db"] - float(t["gain_db"])) < 0.01


def test_the_renderer_measures_what_it_does(session):
    """The audit needs the renderer to report its own work.

    A claim like "high-pass at 30 Hz" cannot be checked by comparing the
    input and the output: the compressor's make-up and the saturation move
    every band afterwards, and a codec's noise floor can even make the
    comparison look inverted.  So the renderer measures each step where it
    is applied, and the numbers go into the report.
    """
    plan, audio, report = pipeline.mix(session, planner.load_preset("pop_2020s"))
    for t in report["tracks"]:
        if t.get("hpf_atten_db") is not None:
            assert t["hpf_atten_db"] <= 0.0, (t["id"], t["hpf_atten_db"])
        for m in t.get("eq_atten_db") or []:
            assert abs(m["measured_db"] - m["gain_db"]) < 1.0, (t["id"], m)
        if t.get("comp_crest"):
            before, after = t["comp_crest"]
            assert after <= before + 0.5, (t["id"], before, after)


def test_a_hostile_file_name_still_reaches_the_mix(tmp_path):
    """The renderer and the planner must agree on what a track id is.

    The renderer used to compute the id its own way (basename, spaces to
    underscores) while the planner used analyze.track_id().  For a file
    name with a newline the two ids differed, so every stem was skipped and
    the mix came out silent -- with no message anywhere.
    """
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    _write(str(d / "kick\nodd.wav"), _stereo(0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 8)))
    _write(str(d / "bass.wav"), _stereo(0.3 * np.sin(2 * np.pi * 55 * t)))
    stems = analyze.load_stems(str(d))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)

    preset = planner.load_preset("pop_2020s")
    plan = planner.build_plan(stems, preset)
    audio, report = render.render(stems, plan)

    assert not report.get("skipped_stems"), report.get("skipped_stems")
    assert len(report["tracks"]) == len(stems)
    # the mix is not silence
    assert float(np.max(np.abs(audio))) > 0.01
    assert report["lufs"] > -40.0


def test_a_stem_the_plan_does_not_mention_is_reported(tmp_path):
    """Leaving a stem out is allowed -- doing it silently is not."""
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    _write(str(d / "kick.wav"), _stereo(0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 8)))
    _write(str(d / "bass.wav"), _stereo(0.3 * np.sin(2 * np.pi * 55 * t)))
    stems = analyze.load_stems(str(d))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)

    preset = planner.load_preset("pop_2020s")
    plan = planner.build_plan(stems, preset)
    # drop one track from the plan, as a hand-edited plan might
    dropped = plan["tracks"].pop()["id"]
    audio, report = render.render(stems, plan)

    # the report names the file, so the user can find it again
    assert any(name.startswith(dropped) for name in report["skipped_stems"])
    names = [name for name, _, _ in render._verdicts(plan, report)]
    assert "stem coverage" in names


# --------------------------------------------------- the user's pan wins

def _pan_stem(name: str, freq: float, dur: float = 2.0, amp: float = 0.3) -> analyze.Stem:
    """One in-memory stem, measured the way the pipeline expects."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t) * np.exp(-((t % 0.5)) * 12.0)
    st = analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], axis=1).astype(np.float32), sr=SR
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def _pan_session(*names: str) -> list:
    stems = [_pan_stem(n, 200.0 + 30.0 * i) for i, n in enumerate(names)]
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


def _plan_pans(plan: dict) -> dict:
    return {t["id"]: float(t["pan"]) for t in plan["tracks"]}


def _plan_evidence(plan: dict) -> list:
    return [e for t in plan["tracks"] for e in t.get("evidence", [])]


def _lr_balance(audio) -> float:
    """Left energy over right energy; above 1.0 means the mix leans left."""
    left = float(np.sqrt(np.mean(audio[:, 0].astype(np.float64) ** 2)))
    right = float(np.sqrt(np.mean(audio[:, 1].astype(np.float64) ** 2)))
    return left / (right + 1e-12)


def test_a_manual_pan_beats_the_duplicate_role_spread():
    """Two tracks share a role and the user panned both to the same place.

    The automatic fan used to run *after* the manual pan and moved them
    apart again -- two tracks the user put at +0.60 came out at +0.50 and
    +0.70.  A pan the user set by hand is the last word, not a suggestion.
    """
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_1.wav", "tom_2.wav")
    plan = planner.build_plan(stems, preset, pans={"tom_1": 0.6, "tom_2": 0.6})
    assert _plan_pans(plan) == {"tom_1": 0.6, "tom_2": 0.6}
    assert any("manual pan +0.60" in e for e in _plan_evidence(plan))


def test_a_manual_centre_pan_survives_the_duplicate_role_spread():
    """Centre is a choice too: two tracks panned to 0 stayed at -0.10/+0.10."""
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_1.wav", "tom_2.wav")
    plan = planner.build_plan(stems, preset, pans={"tom_1": 0.0, "tom_2": 0.0})
    assert _plan_pans(plan) == {"tom_1": 0.0, "tom_2": 0.0}
    assert any("manual pan +0.00" in e for e in _plan_evidence(plan))


def test_a_manual_pan_beats_the_left_right_file_name_hints():
    """The file name may say left, but the user may say otherwise."""
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_left.wav", "tom_right.wav")
    # with no manual pan the names still decide
    automatic = planner.build_plan(stems, preset)
    assert _plan_pans(automatic) == {"tom_left": -0.4, "tom_right": 0.4}
    # with one, the user's pan wins over the name
    plan = planner.build_plan(stems, preset, pans={"tom_left": 0.7, "tom_right": 0.7})
    assert _plan_pans(plan) == {"tom_left": 0.7, "tom_right": 0.7}


def test_an_unmapped_track_keeps_its_normal_automatic_position():
    """Panning one track must not drag its neighbours around with it.

    The unmapped tracks have to end up exactly where the automatic pass
    put them -- a pan map is an override, not a change to the layout.
    """
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_1.wav", "tom_2.wav", "bass.wav", "guitar_elec.wav")
    automatic = _plan_pans(planner.build_plan(stems, preset))
    partial = _plan_pans(planner.build_plan(stems, preset, pans={"tom_2": 0.33}))
    assert partial["tom_2"] == 0.33
    for tid, pan in automatic.items():
        if tid == "tom_2":
            continue
        assert partial[tid] == pan, tid


def test_an_absent_empty_or_unknown_pan_map_changes_nothing():
    """No manual pan keeps automatic positions and adds no manual-pan evidence."""
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_1.wav", "tom_2.wav", "bass.wav")
    base = _plan_pans(planner.build_plan(stems, preset))
    for pans in (None, {}, {"no_such_track": 0.5}):
        plan = planner.build_plan(stems, preset, pans=pans)
        assert _plan_pans(plan) == base, pans
        assert not any("manual pan" in e for e in _plan_evidence(plan)), pans


def test_a_manual_pan_reaches_the_audio_with_the_right_direction():
    """A pan that never reaches the audio is not a pan.

    Rendered through the real pipeline -- no mocked renderer, no
    hand-edited plan.  This stem's automatic role pan is -0.30, so the
    left-leaning case alone would prove nothing: the right-leaning case
    is the one that shows the manual value arrived, and the left one
    shows it was not mirrored.
    """
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("guitar_elec.wav")
    _, left, _ = pipeline.mix(stems, preset, pans={"guitar_elec": -0.9})
    _, right, _ = pipeline.mix(stems, preset, pans={"guitar_elec": 0.9})
    assert _lr_balance(left) > 1.5, _lr_balance(left)
    assert _lr_balance(right) < 0.67, _lr_balance(right)


def test_a_manual_pan_on_a_shared_role_reaches_the_audio():
    """The same, for the case that was broken: two tracks, one role."""
    preset = planner.load_preset("pop_2020s")
    stems = _pan_session("tom_1.wav", "tom_2.wav")
    automatic_plan, _, _ = pipeline.mix(stems, preset)
    plan, manual, _ = pipeline.mix(stems, preset, pans={"tom_1": 0.9, "tom_2": 0.9})
    # the plan the renderer actually used still holds the user's numbers
    assert _plan_pans(plan) == {"tom_1": 0.9, "tom_2": 0.9}
    # the automatic placement of one role's duplicates is a deliberate pair
    # around the centre; the manual pan puts both hard right, and the audio
    # has to follow
    assert _plan_pans(automatic_plan) == {"tom_1": -0.1, "tom_2": 0.1}
    assert _lr_balance(manual) < 0.67, _lr_balance(manual)
