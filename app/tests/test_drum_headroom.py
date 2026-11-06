"""The measured correction is a replayable fader; failed trials leave no residue."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from automix import analyze, pipeline, plan, render

HERE = Path(__file__).resolve().parent


def saved_plan():
    return json.loads((HERE / "r25_reference" / "R121-pop-plan.json").read_text())


def test_frozen_headroom_fader_moves_drum_returns_and_matches_one_db_manual_trim():
    stems = analyze.load_stems(str(HERE.parent / "demo_stems"))
    analyze.align(stems)
    auto = saved_plan()
    auto["headroom_trim_db"] = {"drum": -1}
    plan.validate(auto)
    manual = saved_plan()
    manual["buses"]["drum"]["user_trim_db"] = -1
    x, report = render.render(stems, auto)
    y, expected = render.render(stems, manual)
    np.testing.assert_array_equal(x, y)
    assert report["send_follow"]["drum"] == expected["send_follow"]["drum"]
    assert report["headroom_trim_db"] == {"drum": -1}
    assert "user_trims" not in report


@pytest.mark.parametrize("loss, lufs, corr, peak, mono, accepts", [
    (-2.9, -10.7, 0.8, -1.38, 0.4, True),
    (-3.0, -11.0, 0.3, -1.0, 3.0, True),
    (-3.01, -10.7, 0.8, -1.38, 0.4, False),
    (-2.9, -11.004, 0.8, -1.38, 0.4, False),
    (-2.9, -8.99, 0.8, -1.38, 0.4, False),
    (-2.9, -10.7, 0.29, -1.38, 0.4, False),
    (-2.9, -10.7, 0.8, -0.99, 0.4, False),
    (-2.9, -10.7, 0.8, -1.38, 3.01, False),
])
def test_trial_only_replaces_plan_audio_and_exports_on_acceptance(monkeypatch, loss, lufs, corr, peak, mono, accepts):
    p = saved_plan()
    before = copy.deepcopy(p)
    audio = object()
    report = {"buses": {"drum": True, "bass": True}, "lufs": -11.05,
              "target_lufs": -10.0, "ceiling_db": -1.0, "limiter": {"gain_reduction_p95_db": -3.41}}
    trial_audio = object()
    trial = {"limiter": {"gain_reduction_p95_db": loss}, "lufs": lufs,
             "correlation": corr, "true_peak_db": peak}
    collect = {"stems": "original"}

    def mix_trial(stems, candidate, reverb_kwargs, collect):
        assert candidate["headroom_trim_db"] == {"drum": -1}
        collect["stems"] = "trial"
        return trial_audio, trial

    monkeypatch.setattr(render, "render", mix_trial)
    monkeypatch.setattr(pipeline, "_note_mono", lambda r, a, w: r.update(mono_loss_db=mono))
    chosen, got_audio, got_report = pipeline._try_headroom([], p, audio, report, {}, collect)
    assert p == before
    if accepts:
        assert chosen["headroom_trim_db"] == {"drum": -1}
        assert got_audio is trial_audio and got_report is trial
        assert collect["stems"] == "trial"
    else:
        assert chosen is p and got_audio is audio and got_report is report
        assert collect["stems"] == "original"
