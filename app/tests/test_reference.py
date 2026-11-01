"""Tests for reference-track mode.

The property that matters: a reference contributes *measurements*, and the
planner blends them with the preset instead of copying them.  So the tests
check the blending arithmetic, the "aim at it and land" end-to-end behaviour,
and the guards that stop a loud reference from overriding the delivery ceiling.
"""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from automix import analyze, cli, dsp, pipeline, plan as planner, reference as refmod, render

SR = 48000


def _write(path, audio, sr=SR):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture()
def session(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    _write(d / "kick.wav", np.stack([np.sin(2 * np.pi * 60 * t) * np.exp(-t * 6)] * 2, axis=1) * 0.6)
    _write(d / "bass.wav", np.stack([0.4 * np.sin(2 * np.pi * 55 * t)] * 2, axis=1))
    _write(d / "vocal_lead.wav", np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1))
    _write(d / "pad.wav", np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1))
    stems = analyze.load_stems(str(d))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


@pytest.fixture()
def reference_file(tmp_path):
    """A deliberately dark, quiet, narrow 'reference'."""
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    left = 0.12 * np.sin(2 * np.pi * 110 * t) + 0.05 * np.sin(2 * np.pi * 220 * t)
    right = left * 0.98
    path = tmp_path / "ref.wav"
    _write(path, np.stack([left, right], axis=1))
    return str(path)


# ---------------------------------------------------------------- measurement

def test_spectrum_is_31_zero_mean_bands(tmp_path, reference_file):
    audio, sr = analyze.load_stem_file(reference_file)
    tone = refmod.spectrum_db(audio, sr)
    assert len(tone) == 31
    # rounded to 2 dp for the JSON, so the mean is zero to within that grid
    assert abs(sum(tone) / len(tone)) < 0.01
    assert all(np.isfinite(tone))


def test_region_tilt_uses_the_preset_vocabulary(reference_file):
    ref = refmod.analyse(reference_file)
    assert set(ref["tone_tilt"]) == {"low", "low_mid", "mid", "presence", "air"}


def test_analyse_reproduces_its_own_measurements(reference_file):
    ref = refmod.analyse(reference_file)
    audio, sr = analyze.load_stem_file(reference_file)
    assert abs(ref["lufs"] - dsp.loudness_lufs(audio, sr)) < 0.05
    assert abs(ref["true_peak_db"] - dsp.true_peak_db(audio, sr)) < 0.05
    assert ref["width"] >= 0.0


def test_corrective_tilt_against_itself_is_flat(reference_file):
    ref = refmod.analyse(reference_file)
    corr = refmod.corrective_tilt(ref["tone"], ref["tone"])
    assert all(abs(v) < 1e-6 for v in corr.values())


def test_missing_reference_raises():
    with pytest.raises(refmod.ReferenceError):
        refmod.analyse("/nonexistent/ref.wav")


# -------------------------------------------------------------------- blending

def test_blend_zero_matches_no_reference(session, reference_file):
    preset = planner.load_preset("rock")
    ref = refmod.analyse(reference_file)
    plain = planner.build_plan(session, preset, preset_path="rock")
    withz = planner.build_plan(session, preset, preset_path="rock", reference=ref, blend=0.0)
    assert plain["buses"]["master"]["target_lufs"] == withz["buses"]["master"]["target_lufs"]
    assert "reference" not in withz["style"]


def test_blend_one_takes_the_reference_loudness(session, reference_file):
    preset = planner.load_preset("rock")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, preset_path="rock", reference=ref, blend=1.0)
    assert abs(p["buses"]["master"]["target_lufs"] - ref["lufs"]) < 0.01


def test_blend_is_linear_in_between(session, reference_file):
    preset = planner.load_preset("rock")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, preset_path="rock", reference=ref, blend=0.5)
    want = 0.5 * preset["target_lufs"] + 0.5 * ref["lufs"]
    assert abs(p["buses"]["master"]["target_lufs"] - want) < 0.02


def test_reference_cannot_push_the_ceiling_up(session, tmp_path):
    """A hot reference master must not raise the style's true-peak ceiling."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    hot = np.stack([0.99 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    path = tmp_path / "hot.wav"
    _write(path, hot)
    ref = refmod.analyse(str(path))
    preset = planner.load_preset("acoustic")  # ceiling -1.0
    p = planner.build_plan(session, preset, reference=ref, blend=1.0)
    assert p["buses"]["master"]["true_peak_db"] <= preset["true_peak_db"]


def test_plan_records_the_reference_block(session, reference_file):
    preset = planner.load_preset("pop_2020s")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, reference=ref, blend=0.7)
    block = p["style"]["reference"]
    assert block["mode"] == "global"
    assert len(block["tone"]) == 31
    assert block["lufs"] == ref["lufs"]
    assert "blended 0.70" in block["notes"]


def test_self_reference_asks_for_almost_no_correction(session, tmp_path):
    """A reference made from our own output must say "nothing to fix".

    This is the guard against double counting: reference mode compares against
    what the *renderer* produces, not against the raw stem sum.  An earlier
    version compared the two and concluded the lows needed -6 dB, which was
    simply the mono-below and HPF stage being counted twice.
    """
    preset = planner.load_preset("pop_2020s")
    p0 = planner.build_plan(session, preset, preset_path="pop_2020s")
    mix0, rep0 = render.render(session, p0, {"decay_s": 0.3, "predelay_ms": 0.0})

    ref_path = tmp_path / "self.wav"
    render.write_wav24(str(ref_path), mix0, SR)
    ref = refmod.analyse(str(ref_path))

    p1, mix1, rep1 = pipeline.mix(
        session,
        preset,
        preset_path="pop_2020s",
        reference=ref,
        blend=1.0,
        reverb_kwargs={"decay_s": 0.3, "predelay_ms": 0.0},
    )
    assert rep1["reference_passes"] == 2

    curve = {b["freq"]: b["gain_db"] for b in p1["buses"]["master"]["eq"]}
    # Baseline already contains style EQ: a self-reference preserves it.
    baseline_curve = {b["freq"]: b["gain_db"] for b in p0["buses"]["master"]["eq"]}
    assert curve == baseline_curve
    assert rep1["reference_tone"]["after_db"] < 0.1
    assert rep1["reference_tone"]["before_db"] < 0.1
    assert abs(rep1["lufs"] - rep0["lufs"]) < 0.5


# ------------------------------------------------------------------ end to end

def test_render_lands_on_the_blended_target(session, reference_file):
    preset = planner.load_preset("rock")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, reference=ref, blend=0.6)
    mix, report = render.render(session, p, {"decay_s": 0.3, "predelay_ms": 0.0})
    assert abs(report["lufs"] - p["buses"]["master"]["target_lufs"]) < 0.5
    assert report["true_peak_db"] <= p["buses"]["master"]["true_peak_db"] + 1e-6


def test_render_matches_the_width_target(session, reference_file):
    preset = planner.load_preset("rock")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, reference=ref, blend=0.8)
    target = p["buses"]["master"]["width_target"]
    mix, report = render.render(session, p, {"decay_s": 0.3, "predelay_ms": 0.0})
    assert abs(report["width"] - target) < 0.02, f"{report['width']} vs {target}"


def test_cli_accepts_a_reference(session, reference_file, tmp_path, capsys):
    out = tmp_path / "o"
    rc = cli.main(
        [
            "--stems",
            str(tmp_path / "stems"),
            "--style",
            "lofi",
            "--reference",
            reference_file,
            "--blend",
            "0.5",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0
    plan = json.loads((out / "mixplan.json").read_text())
    assert plan["style"]["reference"]["mode"] == "global"
    assert plan["style"]["blend"] == 0.5
    assert (out / "mix.wav").exists()


def test_cli_reports_a_bad_reference(session, tmp_path, capsys):
    rc = cli.main(
        [
            "--stems",
            str(tmp_path / "stems"),
            "--style",
            "lofi",
            "--reference",
            "/nope.wav",
            "--out",
            str(tmp_path / "o2"),
            "--quiet",
        ]
    )
    assert rc == 2
    assert "reference not found" in capsys.readouterr().err

# ------------------------------------------------------- reverb matching (opt-in)

def test_reverb_match_is_off_by_default(session, reference_file):
    preset = planner.load_preset("pop_2020s")
    ref = refmod.analyse(reference_file)
    p = planner.build_plan(session, preset, reference=ref, blend=1.0)
    assert p["reverb"]["scale"] == 1.0
    assert not any("reverb match" in w for w in p["warnings"])


def test_reverb_match_is_bounded_and_needs_evidence(session, reference_file):
    preset = planner.load_preset("pop_2020s")
    ref = refmod.analyse(reference_file)
    # no baseline at all: cannot compare, so nothing happens
    p = planner.build_plan(
        session, preset, reference=ref, blend=1.0, reverb_match=True, baseline=None
    )
    assert p["reverb"]["scale"] == 1.0

    # with a baseline whose tail is far from the reference's, the move is small
    baseline = {"tone": ref["tone"], "width": 0.1, "crest": 10.0, "tail": 0.05}
    p2 = planner.build_plan(
        session, preset, reference=ref, blend=1.0, reverb_match=True, baseline=baseline
    )
    assert 0.85 <= p2["reverb"]["scale"] <= 1.2
    assert any("reverb match" in w for w in p2["warnings"])
