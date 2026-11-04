"""Vocal weight and edge on the vocal bus (vocal_character_v1).

Requested 2026-11-04: "Vocal要能更有張力與厚實，那個分類都是".  The chain a
vocal mixer reaches for after the leveling compressor: a parallel compressed
copy (density), a little saturation (edge), body / presence / air EQ.  The
vocal balance that follows resets the level, so the change is in the tone.
"""
import copy

import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000


def _singer(seconds=12.0, seed=2):
    """A voice-like signal: phrases with loud and soft syllables and gaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 196.0 * (1 + .02 * np.sin(2 * np.pi * 5 * t))
    tone = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / SR) / k for k in range(1, 12))
    syll = np.repeat(rng.uniform(.15, 1.0, int(seconds * 4) + 1), SR // 4)[: t.size]
    phrase = (np.sin(2 * np.pi * t / 3.0) > -.3).astype(float)
    x = .12 * tone * syll * phrase
    return np.stack([x, x], 1)


def _session(with_vocal=True):
    t = np.arange(int(SR * 12)) / SR
    stems = []
    def add(name, audio):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=audio.astype(np.float32)))
    add("Bass.wav", .2 * np.sin(2 * np.pi * 55 * t)[:, None] * np.ones((1, 2)))
    add("Keys.wav", .05 * np.sin(2 * np.pi * 440 * t)[:, None] * np.ones((1, 2)))
    if with_vocal:
        add("Lead Vocal.wav", _singer())
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_every_style_gives_the_vocal_weight_and_edge():
    import os
    styles = sorted(f[:-5] for f in os.listdir(os.path.join(os.path.dirname(planner.__file__), "..", "..", "presets"))
                    if f.endswith(".json"))
    assert len(styles) == 14
    for style in styles:
        c = planner.build_plan(_session(), planner.load_preset(style))["buses"]["vocal"]["character"]
        assert c["law"] == "vocal_character_v1" and c["parallel_blend"] > 0, style
        assert c["body_db"] > 0, style


def test_no_vocal_no_character_and_intensity_scales_it():
    assert "character" not in planner.build_plan(_session(False), planner.load_preset("pop_2020s"))["buses"]["vocal"]
    half = planner.build_plan(_session(), planner.load_preset("pop_2020s"), intensity=.5)["buses"]["vocal"]["character"]
    full = planner.build_plan(_session(), planner.load_preset("pop_2020s"))["buses"]["vocal"]["character"]
    assert half["presence_db"] == pytest.approx(full["presence_db"] / 2)
    assert "character" not in planner.build_plan(_session(), planner.load_preset("pop_2020s"), intensity=0.0)["buses"]["vocal"]


def test_the_vocal_gets_denser_and_the_tone_moves_as_written():
    sig = _singer()
    spec = {"law": "vocal_character_v1", "parallel_blend": .5, "saturation": .1,
            "body_db": 2.0, "presence_db": 2.0, "air_db": 0.0}
    out, rec = render._vocal_character(sig, SR, spec)
    assert rec["crest_after_db"] < rec["crest_before_db"] - .5        # denser: soft syllables up
    assert rec["body_db"] > .5 and rec["presence_db"] > .5             # and the tone moved
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(out) == pytest.approx(rms(sig), rel=.3)                 # not just louder


def test_the_level_still_lands_on_the_vocal_target():
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    vb = rep["vocal_balance"]
    assert vb["measured_db"] + vb["applied_db"] == pytest.approx(vb["target_db"], abs=.05)
    assert "vocal_character" in rep["mix_moves"]


def test_removing_the_block_renders_as_before_it_existed():
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    with_c, _ = render.render(stems, p)
    q = copy.deepcopy(p); del q["buses"]["vocal"]["character"]
    without, rep = render.render(stems, q)
    assert not np.allclose(with_c, without)
    assert "vocal_character" not in (rep.get("mix_moves") or {})


def test_character_only_on_the_vocal_bus():
    p = planner.build_plan(_session(), planner.load_preset("pop_2020s"))
    q = copy.deepcopy(p)
    q["buses"]["music"]["character"] = q["buses"]["vocal"]["character"]
    with pytest.raises(planner.PlanError, match="only the vocal bus"):
        planner.validate(q)


def test_long_gaps_do_not_make_the_vocal_compressed_harder():
    """The same phrases with short or long silences between them get the
    same treatment (the threshold follows where the voice sings)."""
    phrase = _singer(4.0)
    gap_short = np.zeros((SR // 2, 2)); gap_long = np.zeros((SR * 8, 2))
    tight = np.concatenate([phrase, gap_short, phrase])
    sparse = np.concatenate([phrase, gap_long, phrase])
    spec = {"law": "vocal_character_v1", "parallel_blend": .5, "saturation": 0.0,
            "body_db": 0.0, "presence_db": 0.0, "air_db": 0.0}
    out_a, _ = render._vocal_character(tight, SR, spec)
    out_b, _ = render._vocal_character(sparse, SR, spec)
    n = phrase.shape[0]
    first_a, first_b = out_a[:n], out_b[:n]          # the same first phrase, both ways
    diff = np.sqrt(np.mean((first_a - first_b) ** 2)) / np.sqrt(np.mean(first_a ** 2))
    # measured: following the whole bus differs by about -64 dB here, the
    # active level makes the two identical to rounding
    assert 20 * np.log10(diff + 1e-30) < -100.0
