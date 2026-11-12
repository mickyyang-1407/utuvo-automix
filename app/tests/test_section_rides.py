"""R124 (stage B): section rides after the adaptive limiter.

When the adaptive limiter alone cannot meet both mastering gates, the drum
and music buses are ridden down -- never the vocal -- in the 15 s blocks where
the limiter still works hardest, and the drive is searched again.  Only with
--adaptive-release (pipeline.mix(adaptive_release=True)); without it nothing
changes.  The rides are plain drum_automation / music_automation moves in the
plan, so the written plan replays to the same audio.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 44100


def _dense_song(seconds=60):
    """A quiet verse, then a wall: drums and guitars slammed for 15 s of 60."""
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(3)
    loud = (t > 30) & (t < 45)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4410, SR // 4):
        tt = t[:4410]
        kick[s:s + 4410] += np.sin(2 * np.pi * 50 * tt) * np.exp(-tt / .06)
    drums = kick * np.where(loud, 1.0, 0.25) + rng.standard_normal(t.size) * np.where(loud, 0.25, 0.02)
    guitar = np.sign(np.sin(2 * np.pi * 110 * t)) * np.where(loud, 0.35, 0.04)
    voice = 0.12 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    bass = 0.15 * np.sin(2 * np.pi * 55 * t)
    stems = []
    for name, a in (("Kick.wav", drums), ("Guitar.wav", guitar), ("Bass.wav", bass), ("Lead Vocal.wav", voice)):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_over_blocks_counts_only_hard_limited_windows():
    x = np.zeros((SR * 60, 2))
    y = np.zeros_like(x)
    x[:, :] = 0.1
    y[:, :] = 0.1
    # 16-20 s: 6 dB of gain reduction -> block 1 (15-30 s)
    y[16 * SR:20 * SR] *= dsp.db_to_lin(-6.0)
    # 46-47 s: 2 dB only -> not counted
    y[46 * SR:47 * SR] *= dsp.db_to_lin(-2.0)
    counts = pipeline._over_blocks(x, y, 0.0, SR)
    assert set(counts) == {1}
    assert counts[1] == pytest.approx(4 / 0.05, abs=2)


def test_no_rides_when_the_gates_already_pass():
    p = {"tracks": [{"role": "kick"}, {"role": "guitar_electric"}]}
    report = {"limiter": {"gain_reduction_p95_db": -1.0}, "target_lufs": -10.0, "lufs": -10.2}
    out = pipeline._try_section_rides([], p, "audio", report, {}, None, {"audio": np.zeros((10, 2))})
    assert out == (p, "audio", report)


def test_drive_search_respects_the_gate_and_the_band():
    rng = np.random.default_rng(1)
    x = (0.05 * rng.standard_normal((SR * 20, 2))).astype(np.float32)
    x[5 * SR:5 * SR + SR // 2] *= 6.0  # half a hot second: 2.5% of the song
    found, lo = pipeline._search_drive(x, SR, -1.0, -12.0)
    assert lo[1][2] == pytest.approx(-12.0 - pipeline.SECTION_RIDE_LOUDNESS_MARGIN_LU, abs=0.05)
    # 2.5% of the song cannot hold the loudest 5% down: a drive passes
    assert found is not None
    assert found[1][3] >= pipeline.SECTION_RIDE_GATE_DB
    assert -13.0 <= found[1][2] <= -11.0
    # and it is the loudest one that does: a little more drive breaks the gate or the band
    louder = pipeline._limit_trial(x, SR, -1.0, found[0] + 0.3)
    assert louder[3] < pipeline.SECTION_RIDE_GATE_DB or louder[2] > -11.0 or found[1][2] >= -12.01


def test_a_band_limited_drum_ride_is_left_alone():
    p = {"tracks": [{"role": "kick"}, {"role": "guitar_electric"}],
         "drum_automation": {"law": "drum_high_cosine_v1", "moves": []}}
    report = {"limiter": {"gain_reduction_p95_db": -5.0}, "target_lufs": -10.0, "lufs": -10.2,
              "sample_rate": SR, "ceiling_db": -1.0}
    capture = {"audio": np.zeros((SR * 120, 2))}
    assert pipeline._try_section_rides([], p, "audio", report, {}, None, capture) == (p, "audio", report)


@pytest.fixture(scope="module")
def dense_mix():
    stems = _dense_song()
    preset = planner.load_preset("heavy_rock_metal")
    return stems, pipeline.mix(stems, preset, adaptive_release=True)


def test_the_vocal_is_never_ridden_and_the_plan_replays(dense_mix):
    stems, (p, audio, report) = dense_mix
    planner.validate(p)
    for key in ("drum_automation", "music_automation"):
        for m in (p.get(key) or {}).get("moves", []):
            assert m["trim_db"] <= 0.0
            assert m["trim_db"] >= pipeline.SECTION_RIDE_FLOOR_DB
    # the renderer's only vocal automation field is vocal_dry_automation; rides never write it
    assert "vocal_dry_automation" not in p
    again, _ = render.render(stems, copy.deepcopy(p))
    assert np.array_equal(np.asarray(again), np.asarray(audio))


def test_forced_rides_are_written_validated_and_replay(monkeypatch):
    # the synthetic song passes on the adaptive limiter alone; a stricter gate
    # forces the ride path so it is exercised end to end
    monkeypatch.setattr(pipeline, "SECTION_RIDE_GATE_DB", -0.5)
    stems = _dense_song()
    p, audio, report = pipeline.mix(stems, planner.load_preset("heavy_rock_metal"), adaptive_release=True)
    moves = (p.get("music_automation") or {}).get("moves", [])
    assert moves, "the forced gate must produce section rides"
    assert (p.get("drum_automation") or {}).get("moves")
    for m in moves:
        assert pipeline.SECTION_RIDE_FLOOR_DB <= m["trim_db"] < 0
        assert m["end_frame"] - m["start_frame"] > 2 * m["fade_frames"]
    assert report["limiter"]["gain_reduction_p95_db"] >= -0.5
    assert abs(report["lufs"] - report["target_lufs"]) <= 1.0
    planner.validate(p)
    again, _ = render.render(stems, copy.deepcopy(p))
    assert np.array_equal(np.asarray(again), np.asarray(audio))


def test_without_adaptive_release_nothing_changes(dense_mix):
    stems, _ = dense_mix
    p, _, _ = pipeline.mix(stems, planner.load_preset("heavy_rock_metal"))
    assert "music_automation" not in p
    assert p["buses"]["master"]["limiter"].get("law") != dsp.ADAPTIVE_LIMITER_LAW


def _never_passes(monkeypatch):
    """No drive ever meets the gate: every search reports only its quiet-edge attempt."""
    real = pipeline._search_drive

    def search(*args):
        _, lo = real(*args)
        return None, lo
    monkeypatch.setattr(pipeline, "_search_drive", search)


def test_best_effort_takes_the_least_squashed_round(dense_mix, monkeypatch):
    # what the mix is without the ride step
    stems, _ = dense_mix
    preset = planner.load_preset("heavy_rock_metal")
    with monkeypatch.context() as m:
        m.setattr(pipeline, "_try_section_rides", lambda s, p, a, r, *rest: (p, a, r))
        _, _, base = pipeline.mix(stems, preset, adaptive_release=True)
    base_gr = base["limiter"]["gain_reduction_p95_db"]
    monkeypatch.setattr(pipeline, "SECTION_RIDE_GATE_DB", -0.5)
    assert base_gr < -0.5, "the ride step must have something to do"
    _never_passes(monkeypatch)
    p, audio, report = pipeline.mix(stems, preset, adaptive_release=True)
    assert (p.get("music_automation") or {}).get("moves"), "the rides of the best round are kept"
    assert p["buses"]["master"]["limiter"]["law"] == dsp.ADAPTIVE_LIMITER_LAW
    assert report["limiter"]["gain_reduction_p95_db"] >= base_gr + pipeline.SECTION_RIDE_BEST_EFFORT_GAIN_DB
    assert abs(report["lufs"] - report["target_lufs"]) <= 1.0
    assert "vocal_dry_automation" not in p
    planner.validate(p)
    again, _ = render.render(stems, copy.deepcopy(p))
    assert np.array_equal(np.asarray(again), np.asarray(audio))


def test_best_effort_keeps_the_mix_when_the_gain_is_small(dense_mix, monkeypatch):
    # R126: "the mix" is whatever the steps before the rides arrived at.  Since the
    # drum hits are sharper, the adaptive limiter alone already helps this song, so
    # the mix that is kept carries its law; what must not happen is a ride.
    stems, _ = dense_mix
    preset = planner.load_preset("heavy_rock_metal")
    with monkeypatch.context() as m:
        m.setattr(pipeline, "_try_section_rides", lambda s, p, a, r, *rest: (p, a, r))
        kept, kept_audio, _ = pipeline.mix(stems, preset, adaptive_release=True)
    monkeypatch.setattr(pipeline, "SECTION_RIDE_GATE_DB", -0.5)
    monkeypatch.setattr(pipeline, "SECTION_RIDE_BEST_EFFORT_GAIN_DB", 50.0)
    _never_passes(monkeypatch)
    p, audio, report = pipeline.mix(stems, preset, adaptive_release=True)
    assert "music_automation" not in p and "drum_automation" not in p
    assert p["buses"]["master"]["limiter"] == kept["buses"]["master"]["limiter"]
    assert np.array_equal(np.asarray(audio), np.asarray(kept_audio))
