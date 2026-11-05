"""R122: the lead vocal's narrow resonances smoothed, frame by frame.

A soothe-style move (not oeksound's algorithm, which is unpublished): per
STFT frame, what sticks out of the voice's own neighbourhood is pulled down,
in 2-8 kHz mostly.  It must not tilt the broadband brightness (that decision
belongs to the owner) and must leave a clean voice alone; a resonance must
come down clearly more than the voice around it.  A plan field
(tracks[].resonance); plans without it are untouched (test_replay_r11*.py).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000
SPEC = {"law": "res_suppress_v1", "ratio": 1.0, **planner.RESONANCE}


def _voice(f0, seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t))) / SR
    y = np.zeros_like(t)
    for h in range(1, int(15000 / f0)):
        f = h * f0
        env = sum(a / np.sqrt(1 + ((f - fc) / bw) ** 2)
                  for fc, bw, a in ((700, 120, 1.0), (1200, 150, 0.6), (2600, 250, 0.35))) + 0.05
        y += env / h ** 0.5 * np.sin(h * ph)
    return 0.1 * y / np.max(np.abs(y)) * ((t % 1.0) < 0.8)


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return 10 * np.log10(np.sum(np.abs(X[(f >= lo) & (f < hi)]) ** 2))


def _st(v):
    return np.stack([v, v], 1)


def test_nothing_to_cut_gives_back_the_input():
    x = np.random.default_rng(0).standard_normal((SR * 2, 2)) * 0.1
    y, rec = dsp.resonance_suppress(x, SR, {**SPEC, "threshold_db": 20.0})
    assert np.max(np.abs(y - x)) < 1e-12
    assert rec["mean_cut_2k_8k_db"] == 0.0


@pytest.mark.parametrize("f0", [150.0, 250.0, 400.0])
def test_a_clean_voice_keeps_its_brightness(f0):
    v = _voice(f0)
    y, _ = dsp.resonance_suppress(_st(v), SR, SPEC)
    assert abs(_band(y[:, 0], 20, 20000) - _band(v, 20, 20000)) < 0.3
    assert abs(_band(y[:, 0], 2000, 8000) - _band(v, 2000, 8000)) < 0.6


@pytest.mark.parametrize("f0", [150.0, 250.0, 400.0])
def test_a_resonance_comes_down_much_more_than_the_voice_around_it(f0):
    v = _voice(f0)
    narrow = (3200 * 2 ** (-1 / 24), 3200 * 2 ** (1 / 24))
    clean, _ = dsp.resonance_suppress(_st(v), SR, SPEC)
    clean_cut = _band(clean[:, 0], *narrow) - _band(v, *narrow)
    r = dsp.apply_eq(_st(v), SR, [("peaking", 3200, 8.0, 6.0)])
    y, rec = dsp.resonance_suppress(r, SR, SPEC)
    cut = _band(y[:, 0], *narrow) - _band(r[:, 0], *narrow)
    assert cut < -1.5
    assert cut < 4.0 * min(clean_cut, -0.05)
    assert abs(_band(y[:, 0], 5000, 8000) - _band(r[:, 0], 5000, 8000)) < 0.3   # elsewhere: alone
    assert rec["deepest_mean_cut_hz"] == pytest.approx(3200, rel=0.15)


def test_noise_and_quiet_passages_are_left_alone():
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(SR * 3) * 0.05
    y, _ = dsp.resonance_suppress(_st(noise), SR, SPEC)
    assert abs(_band(y[:, 0], 2000, 8000) - _band(noise, 2000, 8000)) < 0.2
    # a resonant passage 50 dB under the loudest one is not touched
    v = dsp.apply_eq(_st(_voice(250.0)), SR, [("peaking", 3200, 8.0, 6.0)])[:, 0]
    quiet = np.concatenate([v, v * 10 ** (-50 / 20)])
    y, _ = dsp.resonance_suppress(_st(quiet), SR, SPEC)
    # judged a full second after the loud passage, uncapped at full ratio: the
    # 80 ms release of the cut before it has died away (exp(-1000/80) ~ 4e-6)
    y, _ = dsp.resonance_suppress(_st(quiet), SR, {**SPEC, "max_mean_cut_db": None} | {"ratio": 1.0})
    tail = slice(v.size + SR, quiet.size - SR // 2)
    assert np.max(np.abs(y[tail, 0] - quiet[tail])) < 1e-4 * np.max(np.abs(quiet[tail]))


def test_a_wrong_law_is_refused():
    with pytest.raises(ValueError):
        dsp.resonance_suppress(np.zeros((SR, 2)), SR, {**SPEC, "law": "res_suppress_v9"})


def _session(seconds=8):
    t = np.arange(SR * seconds) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    lead = dsp.apply_eq(_st(np.tile(_voice(220.0, 2.0), seconds // 2)), SR, [("peaking", 3200, 8.0, 6.0)])[:, 0]
    tracks = {"Kick.wav": .5 * kick, "Bass.wav": .1 * np.sin(2 * np.pi * 55 * t),
              "Keys.wav": .08 * np.sin(2 * np.pi * 330 * t), "Lead Vocal.wav": 1.5 * lead}
    stems = []
    for name, a in tracks.items():
        a = a + 1e-4 * np.random.default_rng(len(name)).standard_normal(t.size)
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=_st(a).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def _res(p):
    return {t["id"]: t["resonance"] for t in p["tracks"] if t.get("resonance")}


def test_new_plans_smooth_the_lead_by_the_style(stems):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    r = _res(p)
    assert list(r) == [analyze.track_id("Lead Vocal.wav")]
    assert next(iter(r.values()))["ratio"] == 0.6 and next(iter(r.values()))["max_mean_cut_db"] == 0.6
    assert next(iter(_res(planner.build_plan(stems, planner.load_preset("rock"))).values()))["ratio"] == 0.5
    assert next(iter(_res(planner.build_plan(stems, planner.load_preset("jazz_natural"))).values()))["ratio"] == 0.4
    half = planner.build_plan(stems, planner.load_preset("pop_2020s"), intensity=0.5)
    assert next(iter(_res(half).values()))["ratio"] == 0.3


def test_the_render_smooths_before_the_compressor_and_says_so(stems, tmp_path):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    mix, rep = render.render(stems, p)
    lead = next(t for t in rep["tracks"] if t["role"] == "lead_vocal")
    steps = [c["step"] for c in lead["chain"]]
    assert steps.index("resonance suppression") < steps.index("compressor")
    rr = rep["resonance"][0]
    assert rr["deepest_mean_cut_hz"] == pytest.approx(3200, rel=0.15) and rr["deepest_mean_cut_db"] > 1.0
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    assert "| vocal resonances smoothed (dynamic, per band) | " in path.read_text()
    off = copy.deepcopy(p)
    for t in off["tracks"]:
        t.pop("resonance", None)
    mix0, rep0 = render.render(stems, off)
    assert "resonance" not in rep0 and not np.allclose(mix, mix0)


@pytest.mark.parametrize("bad", [{"law": "res_suppress_v9"}, {"ratio": 3.0}, {"ratio": float("nan")},
                                 {"max_db": 40.0}, {"threshold_db": True}, {"detect_oct": 0.0}])
def test_a_wrong_resonance_block_is_refused(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    t = next(t for t in p["tracks"] if t.get("resonance"))
    t["resonance"] = {**t["resonance"], **bad}
    with pytest.raises(render.RenderError):
        render.render(stems, p)


def test_a_whole_song_is_held_to_the_cap():
    """Many strong resonances would take more than the cap off 2-8 kHz on
    average; the ratio is scaled down once so the song keeps its brightness."""
    v = _voice(250.0)
    harsh = dsp.apply_eq(_st(v), SR, [("peaking", f, 12.0, 6.0) for f in (2500, 3600, 5000, 6800)])
    assert SPEC["max_mean_cut_db"] == 0.6
    free, rec_free = dsp.resonance_suppress(harsh, SR, {k: x for k, x in SPEC.items() if k != "max_mean_cut_db"})
    cap = 0.2     # (well under what these four resonances take off)
    assert rec_free["mean_cut_2k_8k_db"] > 1.5 * cap
    held, rec = dsp.resonance_suppress(harsh, SR, {**SPEC, "max_mean_cut_db": cap})
    assert rec["mean_cut_2k_8k_db"] <= cap + 0.02
    assert rec["capped_from_db"] == pytest.approx(rec_free["mean_cut_2k_8k_db"], abs=0.01)
    assert 0.0 < rec["ratio_used"] < SPEC["ratio"]
    # a song under the cap is not touched by it
    _, rec_clean = dsp.resonance_suppress(_st(v), SR, SPEC)
    assert "capped_from_db" not in rec_clean
