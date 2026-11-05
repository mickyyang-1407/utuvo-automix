"""R117: drums that still move.

The drum bus came out with 7-9 dB of level spread against 16.6 dB in the
professional stems (46 MedleyDB songs); the bus compressor's threshold sat
3 dB under the bus's average level, so every hit lost 6-9 dB, and the close
mics were compressed at full ratio with a fast attack.  New plans glue the
bus (threshold above the average, at most 2:1, slower attack) and compress
close drum mics at half the ratio with an attack that lets the hit through.
These are plain compressor numbers in the plan: older plans render as they
were (test_replay_r116.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000


def _kit(seconds=10):
    t = np.arange(SR * seconds) / SR
    kick = np.zeros_like(t)
    snare = np.zeros_like(t)
    rng = np.random.default_rng(5)
    for i, s in enumerate(range(0, t.size - 9600, SR // 4)):
        tt = t[:9600]
        if i % 4 == 0:
            kick[s:s + 9600] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt / .08)
        if i % 4 == 2:
            snare[s:s + 9600] += rng.standard_normal(9600) * np.exp(-tt / .05)
    ghost = 0.03 * rng.standard_normal(t.size)  # room / ghost notes between hits
    stems = []
    for name, a in (("Kick.wav", .7 * kick), ("Snare.wav", .4 * snare + ghost),
                    ("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t)),
                    ("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t)),
                    ("Lead Vocal.wav", .1 * np.sin(2 * np.pi * 440 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def kit_plan():
    stems = _kit()
    return stems, planner.build_plan(stems, planner.load_preset("pop_2020s"))


def test_the_drum_bus_glues_instead_of_holding_every_hit_down(kit_plan):
    _, p = kit_plan
    planner.validate(p)
    c = p["buses"]["drum"]["comp"]
    assert c["threshold_offset_db"] == planner.DRUM_BUS_THRESHOLD_OFFSET_DB > 0
    assert c["ratio"] <= planner.DRUM_BUS_MAX_RATIO
    assert c["attack_ms"] >= planner.DRUM_BUS_MIN_ATTACK_MS
    # the other buses keep their glue as before
    assert p["buses"]["music"]["comp"]["threshold_offset_db"] < 0


def test_close_drum_mics_let_the_hit_through(kit_plan):
    _, p = kit_plan
    drums = [t for t in p["tracks"] if planner.ROLE_GROUP.get(t["role"]) == "drum" and t.get("comp")]
    assert drums
    preset = planner.load_preset("pop_2020s")
    full = planner._comp_params("drum", float(preset["compression"]["drum"]) * float(p["style"]["intensity"]))
    # the style's dial alone would attack faster than the hit lasts
    assert full["attack_ms"] < 20.0
    for t in drums:
        assert t["comp"]["attack_ms"] >= 25.0 > full["attack_ms"]
        assert t["comp_style_dial"] == {"law": "drum_open_v1", "ratio": full["ratio"], "attack_ms": full["attack_ms"]}
        assert t["comp"]["ratio"] == pytest.approx(1.0 + (full["ratio"] - 1.0) * planner.DRUM_TRACK_RATIO_SCALE, abs=0.01)
    # only drums: the bass bus keeps its glue below the average as before
    assert p["buses"]["bass"]["comp"]["threshold_offset_db"] < 0


def _spread_db(x):
    mono = x.mean(axis=1)
    w = int(.05 * SR)
    n = (mono.size // w) * w
    p = 10 * np.log10(np.mean(mono[:n].reshape(-1, w) ** 2, axis=1) + 1e-20)
    act = p > p.max() - 30
    return float(np.percentile(p[act], 90) - np.percentile(p[act], 10))


def test_the_new_bus_settings_keep_more_of_the_kits_movement(kit_plan):
    stems, p = kit_plan
    bus = sum(np.asarray(s.audio, dtype=np.float64) for s in stems if planner.ROLE_GROUP.get(s.role) == "drum")
    new = p["buses"]["drum"]["comp"]
    old = dict(new, threshold_offset_db=-3.0, ratio=2.69, attack_ms=28.2)
    raw = _spread_db(bus)
    got_new = _spread_db(render._apply_comp(bus, SR, new))
    got_old = _spread_db(render._apply_comp(bus, SR, old))
    assert got_new > got_old + 2.0
    assert raw - got_new < 4.0  # a few dB of glue, not a lid
