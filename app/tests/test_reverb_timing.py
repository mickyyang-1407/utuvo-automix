"""The reverb locked to the song's tempo (R116, tempo_space_v1).

The published rule: a reverb tail
that has died by the next backbeat, a pre-delay that is a note value.  The
planner takes the style's own decay and pre-delay as the aim and moves each
to the beat times the power of two nearest to it, only when the tempo is clear.
A plan without the timing block keeps the numbers it was written with.
"""
import copy
import json
import math

import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 48000


def _clicks(bpm, seconds=24.0):
    n = int(SR * seconds)
    x = np.zeros(n)
    k = np.arange(int(.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k / .04)
    beat = 60.0 / bpm
    i = 0
    while (i + 1) * beat < seconds - .3:
        at = int(i * beat * SR)
        x[at:at + kick.size] += kick
        i += 1
    return np.stack([x, x], 1)


def _session(bpm=110, seconds=24.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    stems = []

    def add(name, audio):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=audio.astype(np.float32)))

    add("Kick.wav", _clicks(bpm, seconds))
    add("Bass.wav", .2 * np.sin(2 * np.pi * 55 * t)[:, None] * np.ones((1, 2)))
    env = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    add("Lead Vocal.wav", (.15 * env * np.sin(2 * np.pi * 330 * t))[:, None] * np.ones((1, 2)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def _plan(stems, preset):
    return planner.build_plan(stems, preset, intensity=1.0)


@pytest.mark.parametrize("bpm", [62, 85, 97, 110, 128, 140, 174, 199])
@pytest.mark.parametrize("aim", [0.04, 0.9, 1.6, 2.8, 3.5])
def test_the_snap_is_a_power_of_two_of_the_beat_and_close_to_the_aim(bpm, aim):
    beat = 60.0 / bpm
    got = planner._snap_to_beat(aim, beat)
    beats = got / beat
    assert abs(math.log2(beats) - round(math.log2(beats))) < 1e-9
    assert 1 / math.sqrt(2) - 1e-9 <= got / aim <= math.sqrt(2) + 1e-9


def test_a_clear_tempo_locks_the_decay_and_pre_delay():
    preset = planner.load_preset("pop_2020s")
    p = _plan(_session(110), preset)
    planner.validate(p)
    tm = p["reverb"]["timing"]
    assert tm["law"] == "tempo_space_v1" and tm["bpm"] == pytest.approx(110, rel=.01)
    assert p["reverb"]["decay_s"] == tm["decay_s"] and p["reverb"]["pre_delay_ms"] == tm["pre_delay_ms"]
    assert tm["style_decay_s"] == preset["reverb"]["decay_s"]
    assert tm["style_pre_delay_ms"] == preset["reverb"]["pre_delay_ms"]
    beat = 60.0 / tm["bpm"]
    for value, unit in ((tm["decay_s"], beat), (tm["pre_delay_ms"] / 1000.0, beat)):
        k = math.log2(value / unit)
        assert abs(k - round(k)) < 1e-3
    # the delay and the reverb hear the same tempo
    assert p["delay"]["tempo"]["bpm"] == tm["bpm"]


def test_every_style_locks_to_the_tempo_and_stays_within_its_own_range():
    import glob, os
    stems = _session(97)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = sorted(glob.glob(os.path.join(here, "presets", "*.json")))
    assert len(files) == 14
    for f in files:
        preset = planner.load_preset(f)
        p = _plan(stems, preset)
        planner.validate(p)
        tm = p["reverb"]["timing"]
        r = preset["reverb"]
        assert 0.7 <= tm["decay_s"] / r["decay_s"] <= 1.42, f
        assert 0.7 <= tm["pre_delay_ms"] / r["pre_delay_ms"] <= 1.42, f


def test_without_a_clear_tempo_the_style_numbers_stay(monkeypatch):
    monkeypatch.setattr(dsp, "estimate_tempo", lambda x, sr: {"bpm": 120.0, "strength": .2, "halves_agree": 1})
    preset = planner.load_preset("pop_2020s")
    p = _plan(_session(), preset)
    assert "timing" not in p["reverb"]
    assert p["reverb"]["decay_s"] == preset["reverb"]["decay_s"]
    assert p["reverb"]["pre_delay_ms"] == preset["reverb"]["pre_delay_ms"]


def test_a_style_that_does_not_ask_keeps_its_numbers():
    preset = copy.deepcopy(planner.load_preset("pop_2020s"))
    preset["techniques"].pop("reverb_tempo_lock")
    p = _plan(_session(), preset)
    assert "timing" not in p["reverb"]
    assert p["reverb"]["decay_s"] == preset["reverb"]["decay_s"]


def test_the_tempo_is_looked_up_once_for_the_delay_and_the_reverb(monkeypatch):
    calls = []
    real = planner._tempo_for_delay
    monkeypatch.setattr(planner, "_tempo_for_delay", lambda stems, tracks: calls.append(1) or real(stems, tracks))
    p = _plan(_session(), planner.load_preset("pop_2020s"))
    assert "delay" in p and "timing" in p["reverb"]      # both used it
    assert len(calls) == 1


def test_the_renderer_uses_the_locked_numbers_and_the_report_says_so(tmp_path):
    stems = _session(110)
    p = _plan(stems, planner.load_preset("pop_2020s"))
    mix, report = render.render(stems, p)
    sp = report["space"]["timing"]
    assert sp["decay_s"] == p["reverb"]["decay_s"] and sp["pre_delay_ms"] == p["reverb"]["pre_delay_ms"]
    path = tmp_path / "report.md"
    render.write_report(str(path), p, report, stems)
    text = path.read_text()
    assert f"locked to {p['reverb']['timing']['bpm']:g} BPM (tempo_space_v1)" in text, text
    assert f"decay {p['reverb']['decay_s']:.2f} s" in text
    # unlocking the same plan (style numbers back) changes the reverb, so the
    # lock is really heard
    q = copy.deepcopy(p)
    q["reverb"]["decay_s"] = q["reverb"]["timing"]["style_decay_s"]
    q["reverb"]["pre_delay_ms"] = q["reverb"]["timing"]["style_pre_delay_ms"]
    del q["reverb"]["timing"]
    other, rep2 = render.render(stems, q)
    assert "timing" not in rep2["space"]
    if (p["reverb"]["decay_s"], p["reverb"]["pre_delay_ms"]) != (q["reverb"]["decay_s"], q["reverb"]["pre_delay_ms"]):
        assert not np.allclose(other, mix)


def test_the_first_render_matches_the_replay_of_the_saved_plan():
    stems = _session(110)
    preset = planner.load_preset("pop_2020s")
    # what pipeline.mix passes when nobody overrides the reverb: the preset's
    kwargs = {"decay_s": float(preset["reverb"]["decay_s"]), "predelay_ms": float(preset["reverb"]["pre_delay_ms"])}
    p = _plan(stems, preset)
    first, _ = render.render(stems, p, pipeline._timed_reverb(kwargs, set(), p))
    replay, _ = render.render(stems, p)
    assert np.array_equal(first, replay)
    # an explicit caller value still wins
    kept = pipeline._timed_reverb({"decay_s": 0.5, "predelay_ms": 7.0}, {"decay_s"}, p)
    assert kept["decay_s"] == 0.5 and kept["predelay_ms"] == p["reverb"]["pre_delay_ms"]


def test_a_plan_without_timing_passes_the_kwargs_through_untouched():
    p = {"reverb": {"decay_s": 1.6, "pre_delay_ms": 28.0}}
    kw = {"decay_s": 1.6, "predelay_ms": 28.0}
    assert pipeline._timed_reverb(kw, set(), p) is kw


def test_the_schema_refuses_a_timing_receipt_out_of_range():
    p = _plan(_session(110), planner.load_preset("pop_2020s"))
    bad = copy.deepcopy(p)
    bad["reverb"]["timing"]["pre_delay_ms"] = 400.0
    with pytest.raises(Exception):
        planner.validate(bad)
    bad = copy.deepcopy(p)
    bad["reverb"]["timing"]["law"] = "tempo_space_v9"
    with pytest.raises(Exception):
        planner.validate(bad)


def test_a_value_outside_the_safety_range_is_clamped_and_says_so(tmp_path):
    preset = copy.deepcopy(planner.load_preset("ambient_space"))
    preset["reverb"]["decay_s"] = 12.0          # snaps to 8.73 s at 110 BPM (16 beats)
    preset["reverb"]["pre_delay_ms"] = 3.0      # snaps to 2.13 ms (1/256 beat), below the 5 ms floor
    stems = _session(110)
    p = _plan(stems, preset)
    planner.validate(p)
    tm = p["reverb"]["timing"]
    assert tm["decay_s"] == planner.REVERB_TIMING_DECAY_S[1] and tm["pre_delay_ms"] == planner.REVERB_TIMING_PRE_MS[0]
    assert tm["clamped"] == ["decay_s", "pre_delay_ms"]
    mix, report = render.render(stems, p)
    path = tmp_path / "report.md"
    render.write_report(str(path), p, report, stems)
    assert "decay_s, pre_delay_ms kept in range" in path.read_text()
    # on the grid: no clamp, no flag
    assert "clamped" not in _plan(stems, planner.load_preset("ambient_space"))["reverb"]["timing"]


def test_the_report_counts_beats_of_what_was_rendered():
    stems = _session(110)
    p = _plan(stems, planner.load_preset("pop_2020s"))
    _, report = render.render(stems, p, {"decay_s": 0.5})     # an explicit override
    tm = report["space"]["timing"]
    assert tm["decay_s"] == 0.5 and tm["overridden"] is True
    assert tm["decay_beats"] == pytest.approx(0.5 / (60.0 / tm["bpm"]), abs=.01)


def test_the_first_render_matches_the_replay_when_the_top_level_numbers_are_missing():
    stems = _session(110)
    p = _plan(stems, planner.load_preset("pop_2020s"))
    p["reverb"].pop("decay_s")
    kw = pipeline._timed_reverb({"decay_s": 1.6, "predelay_ms": 28.0}, set(), p)
    first, _ = render.render(stems, p, kw)
    replay, _ = render.render(stems, p)
    assert np.array_equal(first, replay)


def test_pipeline_mix_renders_exactly_what_its_saved_plan_replays():
    """The wiring, not only the helper: pipeline.mix's audio and a replay of
    the plan it returns are the same samples (the tempo-locked reverb is
    what both use)."""
    stems = _session(110)
    plan, audio, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert plan["reverb"]["decay_s"] != planner.load_preset("pop_2020s")["reverb"]["decay_s"]   # locked
    replay, _ = render.render(stems, json.loads(json.dumps(plan)))
    assert np.array_equal(audio, replay)
