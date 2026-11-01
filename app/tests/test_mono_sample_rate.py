"""The mono fold-down check has to measure at the rate the mix was rendered at.

The renderer measures stereo loudness at the session's native rate, but its
report used to carry no rate, and the mono check filled the gap with 48 kHz.
Handed 96 or 192 kHz audio as if it were 48 kHz, the meter skips its resample
and reads the fold-down low -- 3.27 dB on an exact dual-mono 4 kHz tone at
192 kHz, enough for a phase warning about a mix that has no phase problem.
These tests use synthetic audio only and keep every signal short.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from automix import analyze, dsp, pipeline, plan as planner, reference as refmod

RATES = (44100, 48000, 96000, 192000)


def _tone(freq, sr, dur=1.5, amp=0.3):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(dur * sr)) / sr)


def _stereo(left, right):
    return np.stack([left, right], axis=1)


def _report(audio, sr):
    """What the renderer reports: stereo loudness measured at the native rate."""
    return {"lufs": round(dsp.loudness_lufs(audio, sr), 2), "sample_rate": sr}


def _session(left, right, sr):
    stems = [analyze.Stem(name="pad.wav", path="pad.wav",
                          audio=_stereo(left, right).astype(np.float32), sr=sr)]
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
@pytest.mark.parametrize("channels", [1, 2])
def test_resample_keeps_the_analytic_signal_on_the_late_output_clock(
    source_rate, target_rate, channels
):
    n = int(1.503 * source_rate)
    duration = n / source_rate

    def signal(times):
        envelope = np.sin(np.pi * times / duration) ** 2
        left = 0.3 * envelope * np.sin(2 * np.pi * 997.0 * times)
        if channels == 1:
            return left
        right = 0.17 * envelope * np.sin(2 * np.pi * 1291.0 * times + 0.37)
        return _stereo(left, right)

    audio = signal(np.arange(n) / source_rate)
    output = dsp.resample_fft(audio, source_rate, target_rate)
    m = int(round(n * target_rate / source_rate))
    assert output.shape == ((m,) if channels == 1 else (m, 2))
    times = np.arange(m) / target_rate
    expected = signal(times)
    late = (times >= 1.0) & (times < 1.4)
    error = np.sqrt(np.mean((output[late] - expected[late]) ** 2))
    reference = np.sqrt(np.mean(expected[late] ** 2))
    assert error / reference < 1e-5


# ------------------------------------------------------------- the check itself

@pytest.mark.parametrize("sr", RATES)
def test_exact_dual_mono_loses_nothing_at_any_native_rate(sr):
    tone = _tone(4000.0, sr)
    audio = _stereo(tone, tone)
    report, warnings = _report(audio, sr), []
    pipeline._note_mono(report, audio, warnings)
    assert abs(report["mono_loss_db"]) <= 0.02
    assert warnings == []
    assert report["mono_diagnostic"]["available"] is True
    assert report["mono_diagnostic"]["sample_rate"] == sr


def test_the_192k_fixture_reproduces_the_old_false_alarm():
    """Guards the fixture: measured as 48 kHz, the same audio did read > 3 dB."""
    sr = 192000
    tone = _tone(4000.0, sr)
    audio = _stereo(tone, tone)
    stereo = dsp.loudness_lufs(audio, sr)
    wrong = stereo - (dsp.loudness_lufs(audio.mean(axis=1, keepdims=True), 48000) + 3.0103)
    assert wrong > 3.0


@pytest.mark.parametrize("sr", RATES)
def test_real_cancellation_is_still_called_out(sr):
    tone = _tone(220.0, sr)
    noise = np.random.default_rng(7).standard_normal(tone.shape[0]) * 0.01
    audio = _stereo(tone + noise, -tone + noise)
    report, warnings = _report(audio, sr), []
    pipeline._note_mono(report, audio, warnings)
    assert report["mono_loss_db"] > 10.0
    assert any("mono fold-down" in w for w in warnings)


def test_partly_wide_material_is_measured_not_alarmed():
    """A quarter of the energy in anti-phase is a real but small loss."""
    sr = 96000
    mid, side = _tone(220.0, sr), _tone(330.0, sr, amp=0.15)
    audio = _stereo(mid + side, mid - side)
    report, warnings = _report(audio, sr), []
    pipeline._note_mono(report, audio, warnings)
    assert 0.3 < report["mono_loss_db"] < 3.0
    assert warnings == []


@pytest.mark.parametrize("bad", [None, 0, -48000, "fast", True, 44100.5, float("nan"), float("inf"), "inf", 10**400])
def test_an_unknown_rate_is_not_assumed_to_be_48k(bad):
    sr = 192000
    tone = _tone(4000.0, sr)
    audio = _stereo(tone, tone)
    report, warnings = _report(audio, sr), []
    if bad is None:
        del report["sample_rate"]
    else:
        report["sample_rate"] = bad
    report["mono_loss_db"] = 3.27  # a stale number must not survive either
    pipeline._note_mono(report, audio, warnings)
    assert "mono_loss_db" not in report
    assert report["mono_diagnostic"]["available"] is False
    assert warnings == []


@pytest.mark.parametrize("bad", [None, "loud", float("inf"), float("nan"), 10**400])
def test_an_unknown_stereo_loudness_is_not_assumed_either(bad):
    sr = 48000
    tone = _tone(220.0, sr)
    audio = _stereo(tone, -tone)
    report, warnings = _report(audio, sr), []
    if bad is None:
        del report["lufs"]
    else:
        report["lufs"] = bad
    report["mono_loss_db"] = 3.27  # a stale number must not survive either
    pipeline._note_mono(report, audio, warnings)
    assert "mono_loss_db" not in report
    assert report["mono_diagnostic"]["available"] is False
    assert warnings == []


@pytest.mark.parametrize("kind", ["silent", "empty", "shorter_than_a_gating_block"])
def test_silent_or_too_short_audio_is_unavailable_not_alarming(kind):
    sr = 96000
    if kind == "silent":
        audio = np.zeros((int(1.5 * sr), 2))
    elif kind == "empty":
        audio = np.zeros((0, 2))
    else:  # anti-phase, but too short to measure: no verdict either way
        tone = _tone(220.0, sr, dur=0.1)
        audio = _stereo(tone, -tone)
    # the meter's own floor for anything it cannot gate, as the renderer reports it
    report = {"lufs": -70.0 if kind == "empty" else round(dsp.loudness_lufs(audio, sr), 2),
              "sample_rate": sr}
    warnings = []
    pipeline._note_mono(report, audio, warnings)
    assert "mono_loss_db" not in report
    assert report["mono_diagnostic"]["available"] is False
    assert warnings == []


def test_an_unexpected_failure_leaves_no_stale_number(monkeypatch):
    """The diagnostic still never fails a mix, and never keeps an old verdict."""
    sr = 48000
    tone = _tone(220.0, sr)
    audio = _stereo(tone, -tone)
    report, warnings = _report(audio, sr), []
    report["mono_loss_db"] = 42.0
    report["mono_diagnostic"] = {"available": True, "loss_db": 42.0}

    def broken(*args, **kwargs):
        raise RuntimeError("meter exploded")

    monkeypatch.setattr(pipeline.dsp, "loudness_lufs", broken)
    pipeline._note_mono(report, audio, warnings)
    assert "mono_loss_db" not in report and "mono_diagnostic" not in report
    assert warnings == []


def test_a_mono_mix_is_left_alone():
    report, warnings = {"lufs": -14.0, "sample_rate": 48000}, []
    pipeline._note_mono(report, np.zeros((48000, 1)), warnings)
    assert "mono_loss_db" not in report and "mono_diagnostic" not in report
    assert warnings == []


# ------------------------------------------------------- the production routes

def _watch_mono(monkeypatch):
    seen = []
    original = pipeline._note_mono

    def watched(report, audio, warnings):
        seen.append(report.get("sample_rate"))
        return original(report, audio, warnings)

    monkeypatch.setattr(pipeline, "_note_mono", watched)
    return seen


def test_the_renderer_hands_its_native_rate_to_the_mono_check(monkeypatch):
    sr = 192000
    tone = _tone(4000.0, sr)
    seen = _watch_mono(monkeypatch)
    plan, _, report = pipeline.mix(_session(tone, tone, sr), planner.load_preset("pop_2020s"))
    assert abs(report["mono_loss_db"]) < 1.0
    assert not any("mono fold-down" in w for w in plan["warnings"])
    assert seen == [sr]
    assert report["sample_rate"] == int(plan["source"]["sample_rate"]) == sr
    assert report["mono_diagnostic"]["sample_rate"] == sr


def test_the_reference_route_measures_both_passes_at_the_native_rate(monkeypatch, tmp_path):
    sr = 96000
    tone = _tone(4000.0, sr)
    path = str(tmp_path / "reference.wav")
    sf.write(path, _stereo(tone, tone) * 0.5, sr, subtype="PCM_24")
    ref = refmod.analyse(path)
    seen = _watch_mono(monkeypatch)
    plan, _, report = pipeline.mix(_session(tone, tone, sr), planner.load_preset("pop_2020s"),
                                   reference=ref)
    assert report["reference_passes"] == 2
    assert abs(report["mono_loss_db"]) < 1.0
    assert not any("mono fold-down" in w for w in plan["warnings"])
    assert seen == [sr, sr]
    assert report["mono_diagnostic"]["sample_rate"] == sr
