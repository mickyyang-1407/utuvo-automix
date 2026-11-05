"""R121: the lead vocal's artificial double (ADT).

The owner wants the lead big, thick and with tension.  Without a real double,
the studio trick is two short copies whose delay drifts slowly -- a drifting
delay is a pitch change, so each copy wanders a few cents sharp and flat --
hard left and right, low under the voice (SOS: 5-20 ms, 2-10 cents, kept low,
or the copies phase-cancel with the dry voice).  Never on a session that
already stacks the voice (a second take, a second lead, a harmony sung with
the lead).  A plan field (tracks[].double); plans without it
are not doubled (test_replay_r11*.py).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _cents(y, f0):
    zc = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    z = zc + (-y[zc]) / (y[zc + 1] - y[zc])
    return 1200 * np.log2(SR / np.diff(z) / f0)


@pytest.mark.parametrize("cents, rate", [(7.0, 0.23), (7.0, 0.31), (12.0, 0.2)])
def test_the_drifting_delay_moves_the_pitch_by_exactly_the_cents(cents, rate):
    seconds = 2.0 / rate                   # two whole swings
    t = np.arange(int(SR * seconds)) / SR
    y = dsp.drifting_delay(np.sin(2 * np.pi * 440 * t), SR, 9.0, cents, rate)
    c = _cents(y, 440.0)[50:]
    assert c.max() == pytest.approx(cents, abs=0.1) and c.min() == pytest.approx(-cents, abs=0.1)
    assert abs(np.mean(c[c.size // 4:])) < 1.0   # it wanders, it does not drift off


def test_the_double_is_two_low_wide_copies():
    r = np.random.default_rng(1)
    t = np.arange(SR * 6) / SR
    # (in the double's band: it is high-passed at 800 Hz)
    voice = 0.2 * np.sin(2 * np.pi * 1500 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)) + 0.02 * r.standard_normal(t.size)
    x = np.stack([voice, voice], 1)
    spec = {"law": "adt_v1", "level_db": -15.0, **planner.VOCAL_DOUBLE}
    y = dsp.vocal_double(x, SR, spec)
    lvl = lambda v: 10 * np.log10(np.mean(v ** 2))
    assert lvl(y[:, 0]) - lvl(voice) == pytest.approx(-15.0, abs=0.6)
    assert lvl(y[:, 1]) - lvl(voice) == pytest.approx(-15.0, abs=0.6)
    assert np.corrcoef(y[SR:, 0], y[SR:, 1])[0, 1] < 0.9             # left and right differ
    # the voice summed with it, folded to mono, moves less than 1.5 dB
    mono_dry = lvl(voice[SR:])
    mono_wet = lvl((x + y)[SR:].mean(axis=1))
    assert abs(mono_wet - mono_dry) < 1.5
    assert np.array_equal(dsp.vocal_double(x, SR, spec), y)           # deterministic
    with pytest.raises(ValueError):
        dsp.vocal_double(x, SR, {**spec, "law": "adt_v9"})


def _session(names=("Lead Vocal.wav",), backing=None, seconds=12):
    t = np.arange(SR * seconds) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    sings = ((t % 2.0) < 1.2).astype(float)
    harm = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 16))
    lead = .1 * sings * harm * (1 + .2 * np.sin(2 * np.pi * 2.5 * t))
    tracks = {"Kick.wav": .5 * kick, "Bass.wav": .1 * np.sin(2 * np.pi * 55 * t),
              "Keys.wav": .08 * np.sin(2 * np.pi * 330 * t)}
    for name in names:
        tracks[name] = lead
    if backing == "harmony":       # a third above, sung with the lead
        tracks["Backing Vocal.wav"] = .1 * sings * np.sin(2 * np.pi * 277 * t) * (1 + .2 * np.sin(2 * np.pi * 2.5 * t + .1))
    elif backing == "lead2":       # the double printed as a second lead track
        tracks["Lead Vocal 2.wav"] = .1 * sings * np.sin(2 * np.pi * 221 * t) * (1 + .2 * np.sin(2 * np.pi * 2.5 * t + .1))
    elif backing == "copy":        # a second take of the same line
        tracks["Backing Vocal.wav"] = .1 * sings * np.sin(2 * np.pi * 442 * t) * (1 + .2 * np.sin(2 * np.pi * 2.5 * t + .1))
    elif backing == "answer":      # an answer line in the lead's gaps
        tracks["Backing Vocal.wav"] = .1 * (1 - sings) * np.sin(2 * np.pi * 550 * t)
    stems = []
    for name, a in tracks.items():
        a = a + 1e-4 * np.random.default_rng(len(name)).standard_normal(t.size)
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def _double_of(p):
    return {t["id"]: t["double"] for t in p["tracks"] if t.get("double")}


def test_new_plans_double_the_lead_by_the_style(stems):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    d = _double_of(p)
    assert list(d) == [analyze.track_id("Lead Vocal.wav")]
    one = next(iter(d.values()))
    assert one["law"] == "adt_v1" and one["level_db"] == -15.0 and one["cents"] == 7.0
    assert one["delays_ms"] == [9.0, 14.0] and one["hpf_hz"] == 800.0
    assert next(iter(_double_of(planner.build_plan(stems, planner.load_preset("lofi"))).values()))["level_db"] == -18.0
    half = planner.build_plan(stems, planner.load_preset("pop_2020s"), intensity=0.5)
    assert next(iter(_double_of(half).values()))["level_db"] == pytest.approx(-15.0 - 6.02, abs=0.01)
    assert not _double_of(planner.build_plan(stems, planner.load_preset("pop_2020s"), intensity=0.2))
    for style in ("jazz_natural", "acoustic"):
        assert not _double_of(planner.build_plan(stems, planner.load_preset(style)))


def test_a_session_with_a_real_double_is_left_alone():
    named = _session(names=("Lead Vocal.wav", "Lead Vocal Dbl.wav"))
    p = planner.build_plan(named, planner.load_preset("pop_2020s"))
    assert not _double_of(p) and any("already stacks the voice" in w for w in p["warnings"])
    take = _session(backing="copy")
    p = planner.build_plan(take, planner.load_preset("pop_2020s"))
    assert not _double_of(p) and any("already stacks the voice" in w for w in p["warnings"])
    answer = _session(backing="answer")          # a different line: still doubled
    assert _double_of(planner.build_plan(answer, planner.load_preset("pop_2020s")))


@pytest.mark.parametrize("backing", ["harmony", "lead2"])
def test_a_stacked_voice_is_already_thick(backing):
    """A harmony sung with the lead, or the double printed as a second lead
    track: the voice is stacked already, no artificial double on top."""
    stems = _session(backing=backing)
    if backing == "lead2":
        assert sum(1 for s in stems if s.role == "lead_vocal") == 2
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    assert not _double_of(p) and any("already stacks the voice" in w for w in p["warnings"])


def test_the_render_doubles_before_the_sends_and_says_so(stems, tmp_path):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    mix, rep = render.render(stems, p)
    lead = next(t for t in rep["tracks"] if t["role"] == "lead_vocal")
    assert any(c["step"] == "double" for c in lead["chain"])
    assert rep["vocal_double"][0]["level_db"] == -15.0
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    assert "| lead vocal doubled (artificial double tracking) | " in path.read_text()
    off = copy.deepcopy(p)
    for t in off["tracks"]:
        t.pop("double", None)
    mix0, rep0 = render.render(stems, off)
    assert "vocal_double" not in rep0 and not np.allclose(mix, mix0)
    # wider: the centred lead gets a side channel of its own, well under its mid
    lid = analyze.track_id("Lead Vocal.wav")
    got, got0 = {}, {}
    render.render(stems, p, collect=got)
    render.render(stems, off, collect=got0)
    def side_mid(x):
        x = np.asarray(x, dtype=np.float64)
        return 10 * np.log10(np.mean(((x[:, 0] - x[:, 1]) / 2) ** 2) / np.mean(((x[:, 0] + x[:, 1]) / 2) ** 2) + 1e-30)
    assert side_mid(got0["stems"][lid]) < -60.0
    assert -35.0 < side_mid(got["stems"][lid]) < -12.0


@pytest.mark.parametrize("bad", [{"law": "adt_v9"}, {"level_db": -3.0}, {"level_db": float("nan")},
                                 {"cents": 40.0}, {"delays_ms": [9.0]}, {"delays_ms": [9.0, 90.0]},
                                 {"rates_hz": [0.2, 5.0]}, {"lpf_hz": True}])
def test_a_wrong_double_is_refused(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    t = next(t for t in p["tracks"] if t.get("double"))
    t["double"] = {**t["double"], **bad}
    with pytest.raises(render.RenderError):
        render.render(stems, p)


def test_a_lead_that_is_already_stereo_is_not_doubled():
    """A printed double or a stereo effect on the lead: it is wide enough already
    (the pro vocal stems sit around -16 dB side/mid)."""
    stems = _session()
    lead = next(s for s in stems if s.role == "lead_vocal")
    a = lead.audio.astype(np.float64)
    wide = np.stack([a[:, 0], np.roll(a[:, 1], int(0.012 * SR)) * 0.9], 1)
    lead.audio = wide.astype(np.float32)
    assert planner._side_mid_db(lead.audio) > planner.DOUBLE_MAX_SOURCE_SIDE_DB
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    t = next(t for t in p["tracks"] if t["role"] == "lead_vocal")
    assert "double" not in t and any(e.startswith("not doubled: the lead is already stereo") for e in t["evidence"])
    assert planner._side_mid_db(np.zeros((10, 2))) == float("-inf")
