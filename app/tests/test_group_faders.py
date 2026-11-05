"""R118: a few group faders after the mix ("vocal a little louder").

The owner (2026-11-05): Automix should not need changing, but when it is a
little off, changing it must be very simple -- list the important groups and
let the user move their balance.  Each group fader is a plan field
(buses.<group>.user_trim_db) applied after every automatic balance; the
group's reverb and delay move with it; fx is the returns bus itself.  Plans
without it render as before (test_replay_r115.py / test_replay_r116.py).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 48000


def _session(seconds=10):
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(4)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    sings = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    stems = []
    for name, a in (("Kick.wav", .5 * kick), ("Bass.wav", .1 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t) + .02 * rng.standard_normal(t.size)),
                    ("Lead Vocal.wav", .1 * sings * np.sin(2 * np.pi * 440 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def _plan(stems, **trims):
    return planner.build_plan(stems, planner.load_preset("pop_2020s"), group_trims=trims or None)


def test_the_planner_writes_the_faders_it_was_given(stems):
    p = _plan(stems, vocal=2.0, music=-1.5, fx=3.0, drum=0.0)
    planner.validate(p)
    assert p["buses"]["vocal"]["user_trim_db"] == 2.0
    assert p["buses"]["music"]["user_trim_db"] == -1.5
    assert p["buses"]["fx"]["user_trim_db"] == 3.0
    assert "user_trim_db" not in p["buses"]["drum"]  # 0 dB is no fader
    plain = _plan(stems)
    assert all("user_trim_db" not in b for b in plain["buses"].values())


@pytest.mark.parametrize("bad", [{"master": 1.0}, {"vocal": 12.5}, {"vocal": float("nan")}, {"guitar": 1.0}])
def test_a_wrong_fader_is_refused(stems, bad):
    with pytest.raises(planner.PlanError):
        planner.build_plan(stems, planner.load_preset("pop_2020s"), group_trims=bad)


def _spy_after_faders(stems, p):
    """The buses and the followed returns where they join the fx bus."""
    got = {}
    real = render._join_followed

    def spy(buses, followed, fader_db, n):
        got["buses"] = {k: v.copy() for k, v in buses.items()}
        got["followed"] = {k: v.copy() for k, v in followed.items()}
        got["fader"] = dict(fader_db)
        return real(buses, followed, fader_db, n)
    render._join_followed = spy
    try:
        _, rep = render.render(stems, p)
    finally:
        render._join_followed = real
    return got, rep


def test_the_vocal_fader_moves_the_vocal_and_its_reverb_after_the_balance(stems):
    base, rep0 = _spy_after_faders(stems, _plan(stems))
    up, rep1 = _spy_after_faders(stems, _plan(stems, vocal=3.0))
    ratio = up["buses"]["vocal"] / np.where(base["buses"]["vocal"] == 0, 1, base["buses"]["vocal"])
    i = int(np.argmax(np.abs(base["buses"]["vocal"][:, 0])))
    assert 20 * np.log10(ratio[i, 0]) == pytest.approx(3.0, abs=1e-6)
    # the automatic balance itself did not change: the fader comes after it
    assert rep1["vocal_balance"]["applied_db"] == rep0["vocal_balance"]["applied_db"]
    # the vocal's reverb/delay join at the balance move plus the fader
    assert up["fader"]["vocal"] == pytest.approx(base["fader"]["vocal"] + 3.0, abs=1e-9)
    assert rep1["user_trims"] == {"vocal": 3.0}


def test_the_music_fader_carries_its_reverb(stems):
    got, rep = _spy_after_faders(stems, _plan(stems, music=-4.0))
    assert "music" in got["followed"]  # its sends were summed apart
    assert got["fader"]["music"] == pytest.approx(-4.0)
    assert rep["send_follow"]["music"]["gain_db"] == pytest.approx(-4.0)


def test_the_space_fader_moves_the_returns(stems):
    _, dry = render.render(stems, _plan(stems, fx=-12.0))
    _, wet = render.render(stems, _plan(stems, fx=6.0))
    _, mid = render.render(stems, _plan(stems))
    # less return -> a more mono (more correlated) mix; more -> wider
    assert dry["correlation"] > mid["correlation"] > wet["correlation"]
    assert wet["user_trims"] == {"fx": 6.0}


def test_the_mix_pipeline_passes_the_faders_on(stems):
    p, _, rep = pipeline.mix(stems, planner.load_preset("pop_2020s"), group_trims={"bass": -2.0})
    assert p["buses"]["bass"]["user_trim_db"] == -2.0
    assert rep["user_trims"] == {"bass": -2.0}


def test_the_report_names_the_faders(stems, tmp_path):
    p = _plan(stems, vocal=1.5, fx=-2.0)
    _, rep = render.render(stems, p)
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    text = path.read_text()
    assert "your faders (after the automatic balance) | vocal +1.5 dB, FX & space -2.0 dB" in text


def test_a_hand_edited_fader_out_of_range_is_refused(stems):
    p = _plan(stems)
    p["buses"]["vocal"]["user_trim_db"] = 20.0
    with pytest.raises(render.RenderError):
        render.render(stems, p)
