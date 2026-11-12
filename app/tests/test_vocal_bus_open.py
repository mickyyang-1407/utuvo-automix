"""R124: the vocal bus catches the loudest syllables instead of every one.

The owner on Velvet Curtain: "Vocal完全Overcompressed" -- but "厚度以及飽滿是對的".
The vocal bus threshold sat 2.3-3.7 dB under the bus's average level; on the
46 MUSDB vocals that stage alone took a median 3.8-5.4 dB of sung-window level
spread.  New plans put the threshold above the average with a quick attack
(the drum bus got the same fix in R117).  The thickness -- the vocal
character's parallel copy, saturation and tone -- is untouched.  These are
plain compressor numbers in the plan: older plans render as they were
(test_replay_r121.py replays a plan from before this change byte for byte).
"""
from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000


def _band(seconds=12):
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(11)
    # a voice that moves: syllables at different levels, with quiet tails
    env = np.zeros_like(t)
    for i, s in enumerate(range(0, t.size - SR // 4, SR // 4)):
        env[s:s + SR // 5] = (0.35 if i % 5 == 0 else 0.12 if i % 2 else 0.06)
    voice = env * np.sin(2 * np.pi * 220 * t) * (1 + 0.2 * rng.standard_normal(t.size))
    kick = np.zeros_like(t)
    for s in range(0, t.size - 9600, SR // 2):
        tt = t[:9600]
        kick[s:s + 9600] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt / .08)
    stems = []
    for name, a in (("Kick.wav", .6 * kick), ("Bass.wav", .1 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .06 * np.sin(2 * np.pi * 330 * t)), ("Lead Vocal.wav", voice)):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module", params=["pop_2020s", "rock", "trap", "heavy_rock_metal"])
def band_plan(request):
    stems = _band()
    return stems, planner.build_plan(stems, planner.load_preset(request.param)), request.param


def test_the_vocal_bus_only_catches_the_loudest_syllables(band_plan):
    _, p, style = band_plan
    planner.validate(p)
    c = p["buses"]["vocal"]["comp"]
    assert c["threshold_offset_db"] == planner.VOCAL_BUS_THRESHOLD_OFFSET_DB > 0
    assert c["attack_ms"] <= planner.VOCAL_BUS_MAX_ATTACK_MS
    dial = p["buses"]["vocal"]["comp_style_dial"]
    assert dial["law"] == planner.VOCAL_OPEN_LAW
    # the receipt holds what the style's dial alone would have written
    assert dial["threshold_offset_db"] < 0
    assert dial["attack_ms"] > c["attack_ms"]
    # the ratio and release are the style's, unchanged
    preset = planner.load_preset(style)
    amt = float(preset["compression"]["vocal"]) * float(p["style"]["intensity"]) * 0.75
    full = planner._comp_params("bus", amt)
    assert c["ratio"] == full["ratio"] and c["release_ms"] == full["release_ms"]


def test_only_the_vocal_bus_moves(band_plan):
    _, p, style = band_plan
    preset = planner.load_preset(style)
    intensity = float(p["style"]["intensity"])
    for name in ("bass", "music"):
        comp = p["buses"][name]["comp"]  # these styles compress both buses
        amt = float(preset["compression"][name]) * intensity * 0.75
        full = planner._comp_params("bus", amt)
        assert comp["threshold_offset_db"] == round(-1.0 - 6.0 * amt, 1) < 0, name
        for key in ("ratio", "attack_ms", "release_ms"):
            assert comp[key] == full[key], (name, key)
        assert "comp_style_dial" not in p["buses"][name], name
    assert p["buses"]["drum"]["comp"]["threshold_offset_db"] == planner.DRUM_BUS_THRESHOLD_OFFSET_DB


def test_the_thickness_is_kept(band_plan):
    # The owner: "Vocal的厚度以及飽滿是對的" -- the character block (parallel copy,
    # saturation, tone) is still written, with real amounts, for every style here
    _, p, _ = band_plan
    ch = p["buses"]["vocal"]["character"]
    assert ch["law"] == "vocal_character_v1"
    assert ch["parallel_blend"] > 0.2
    assert ch["saturation"] > 0.0
    assert ch["body_db"] > 0.0 and ch["presence_db"] > 0.0


def _spread_db(x, sung):
    m = np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=1)
    w = SR // 10
    n = len(m) // w
    e = 10 * np.log10(m[: n * w].reshape(n, w).mean(1) + 1e-20)
    v = e[sung[:n]]
    return float(np.percentile(v, 95) - np.percentile(v, 10))


def test_the_new_vocal_bus_keeps_more_of_the_voice_moving(band_plan):
    stems, p, _ = band_plan
    voice = next(s for s in stems if s.role in planner.VOCAL_ROLES).audio.astype(np.float32)
    m = np.mean(voice.astype(np.float64) ** 2, axis=1)
    w = SR // 10
    e = 10 * np.log10(m[: (len(m) // w) * w].reshape(-1, w).mean(1) + 1e-20)
    sung = e > e.max() - 30.0
    new = p["buses"]["vocal"]["comp"]
    dial = p["buses"]["vocal"]["comp_style_dial"]
    old = dict(new, attack_ms=dial["attack_ms"], threshold_offset_db=dial["threshold_offset_db"])
    law = p.get("compressor_law", "centred_box_v1")
    before = _spread_db(voice, sung)
    with_old = _spread_db(render._apply_comp(voice, SR, old, law), sung)
    with_new = _spread_db(render._apply_comp(voice, SR, new, law), sung)
    assert with_old < before - 1.0          # the old bus flattened the voice
    assert with_new > with_old + 1.0         # the new one leaves it moving
    assert with_new <= before + 0.1          # and still never expands it
