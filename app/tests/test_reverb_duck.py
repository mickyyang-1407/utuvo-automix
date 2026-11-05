"""R119: the vocal reverb steps back while the voice sings and blooms in the gaps.

The owner wants a vocal that is never dry; expert listeners mark a mix down
harder for too much reverb than for too little (De Man, McNally & Reiss,
JAES 2017: about -14 LU of reverb under the voice preferred, -9 LU and up
judged too wet).  The owner also said the vocal was a bit dry and the image
must be big.  So the vocal's own return steps back while the voice sings
(full depth at or above its singing level, none 10 dB below) and the whole
return is lifted by the same amount: as wet as before under the words, the
tail blooms by the depth in the gaps.  A plan field (reverb.duck); plans
without it are not ducked (test_replay_r11*.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _phrases(seconds=8):
    """A voice singing 1 s phrases with 1 s gaps, and a steady 'reverb' to duck."""
    t = np.arange(SR * seconds) / SR
    sings = ((t % 2.0) < 1.0).astype(float)
    voice = 0.2 * sings * np.sin(2 * np.pi * 330 * t)
    wet = 0.05 * np.random.default_rng(3).standard_normal(t.size)
    return np.stack([voice, voice], 1), np.stack([wet, wet], 1), sings


def _gain_db(out, wet, mask):
    g = 20 * np.log10(np.abs(out[:, 0]) + 1e-12) - 20 * np.log10(np.abs(wet[:, 0]) + 1e-12)
    return float(np.median(g[mask]))


def test_the_return_steps_back_while_singing_and_comes_back_in_the_gaps():
    voice, wet, sings = _phrases()
    out, rec = dsp.voice_duck(wet, voice, SR, 4.0, attack_ms=20.0, release_ms=250.0)
    t = np.arange(len(sings)) / SR
    mid_phrase = (sings > 0) & ((t % 2.0) > 0.2) & ((t % 2.0) < 0.9)
    deep_gap = (sings == 0) & ((t % 2.0) > 1.5)
    assert _gain_db(out, wet, mid_phrase) == pytest.approx(-4.0, abs=0.2)
    assert _gain_db(out, wet, deep_gap) == pytest.approx(0.0, abs=0.1)
    assert rec["gain_while_singing_db"] == pytest.approx(-4.0, abs=0.2)
    assert rec["gain_in_gaps_db"] == pytest.approx(0.0, abs=0.1)


def test_the_duck_only_reacts_to_what_was_sung():
    voice, wet, sings = _phrases()
    out, _ = dsp.voice_duck(wet, voice, SR, 4.0, attack_ms=20.0, release_ms=250.0)
    # everything up to the sample before the second phrase starts is untouched:
    # neither the 10 ms level window nor the gain's 20 ms reach ahead
    a, b = int(1.9 * SR), int(2.0 * SR)
    assert np.array_equal(out[a:b], wet[a:b])
    assert not np.array_equal(out[b:b + SR // 100], wet[b:b + SR // 100])


def test_the_kick_duck_keeps_its_centred_level_and_the_voice_reads_only_the_past():
    x = np.zeros((1000, 2))
    x[500:] = 1.0
    old = dsp.key_level_db(x, 48000, 1.0)             # 48-sample centred window
    new = dsp.key_level_db(x, 48000, 1.0, trailing=True)
    assert old[480] > -100.0 and new[499] < -100.0 and new[500] > -100.0


def test_nothing_to_duck_leaves_the_return_alone():
    _, wet, _ = _phrases(2)
    out, rec = dsp.voice_duck(wet, np.zeros_like(wet), SR, 4.0)
    assert out is wet and "skipped" in rec
    out, _ = dsp.voice_duck(wet, wet, SR, 0.0)
    assert out is wet


def _session(seconds=10):
    t = np.arange(SR * seconds) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    sings = ((t % 2.0) < 1.2).astype(float)
    stems = []
    for name, a in (("Kick.wav", .5 * kick), ("Bass.wav", .1 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t)),
                    ("Lead Vocal.wav", .15 * sings * np.sin(2 * np.pi * 440 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def test_new_plans_duck_the_vocal_reverb_by_the_style(stems):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    duck = p["reverb"]["duck"]
    assert duck["law"] == "voice_duck_v1" and duck["depth_db"] == 4.0
    assert duck["makeup_db"] == duck["depth_db"]
    assert 120.0 <= duck["release_ms"] <= 400.0
    metal = planner.build_plan(stems, planner.load_preset("heavy_rock_metal"))
    assert metal["reverb"]["duck"]["depth_db"] == 2.0
    half = planner.build_plan(stems, planner.load_preset("pop_2020s"), intensity=0.5)
    assert half["reverb"]["duck"]["depth_db"] == 2.0


def test_the_render_keeps_the_words_as_wet_and_lets_the_gaps_bloom(stems):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    rd = rep["reverb_duck"]
    assert rd["law"] == "voice_duck_v1" and rd["depth_db"] == 4.0 and rd["makeup_db"] == 4.0
    assert rd["gain_while_singing_db"] == pytest.approx(0.0, abs=0.3)   # as wet as before under the words
    assert rd["gain_in_gaps_db"] == pytest.approx(4.0, abs=0.3)         # the tail blooms in the gaps
    assert rd["return_change_db"] > 0.5
    off = dict(p)
    off["reverb"] = {k: v for k, v in p["reverb"].items() if k != "duck"}
    _, rep0 = render.render(stems, off)
    assert "reverb_duck" not in rep0
    # the vocal's return is what moved: its receipt where it joins the fx bus
    assert rep["send_follow"]["vocal"]["return_db"] > rep0["send_follow"]["vocal"]["return_db"] + 0.5
    # without makeup it is a plain duck: 4 dB down under the words
    plain = dict(p)
    plain["reverb"] = {**p["reverb"], "duck": {**p["reverb"]["duck"], "makeup_db": 0.0}}
    _, rep1 = render.render(stems, plain)
    assert rep1["reverb_duck"]["gain_while_singing_db"] == pytest.approx(-4.0, abs=0.3)


def test_the_report_names_the_duck(stems, tmp_path):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    assert ("| vocal reverb steps back while the voice sings | ducked 4.0 dB while singing, whole return +4.0 dB: "
            in path.read_text())


@pytest.mark.parametrize("bad", [{"law": "voice_duck_v9", "depth_db": 3.0},
                                 {"law": "voice_duck_v1", "depth_db": 20.0},
                                 {"law": "voice_duck_v1", "depth_db": 3.0, "makeup_db": -2.0},
                                 {"law": "voice_duck_v1", "depth_db": float("nan")}])
def test_a_wrong_duck_is_refused(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p["reverb"]["duck"] = bad
    with pytest.raises(render.RenderError):
        render.render(stems, p)


def test_a_plan_without_a_vocal_balance_still_ducks_the_vocal_return(stems):
    """The duck needs the vocal's own return; a hand-edited plan that has no
    vocal balance (so no post-fader sends) gets it summed apart anyway."""
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p["buses"]["vocal"].pop("balance", None)
    _, rep = render.render(stems, p)
    assert rep["reverb_duck"]["gain_in_gaps_db"] == pytest.approx(4.0, abs=0.3)


def test_the_exported_reverb_is_the_ducked_return(stems):
    """The DAW export holds the return the mix had: ducked and lifted."""
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    got, _ = {}, None
    render.render(stems, p, collect=got)
    off = dict(p)
    off["reverb"] = {k: v for k, v in p["reverb"].items() if k != "duck"}
    plain = {}
    render.render(stems, off, collect=plain)
    assert not np.allclose(got["reverb"], plain["reverb"])
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))
    assert rms(got["reverb"]) > rms(plain["reverb"])   # the tail blooms in the gaps
