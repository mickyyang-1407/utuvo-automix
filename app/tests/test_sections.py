"""Tests for section detection and the per-section character it drives.

The rule these protect: sections may change the *character* of the mix (width,
reverb send) by a small bounded amount, and must do nothing when the song has
no clear structure or is too short to have one.
"""

from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render, sections

SR = 48000


@pytest.fixture(autouse=True)
def tiny_sections(monkeypatch):
    """Shrink the section thresholds so tests can use 12 s songs, not 36 s.

    The logic is duration-independent; only the constants are.  Patching them
    keeps the suite honest and roughly three times faster, because every one of
    these tests renders audio.
    """
    monkeypatch.setattr(sections, "MIN_TOTAL_S", 6.0)
    monkeypatch.setattr(sections, "MIN_SECTION_S", 2.0)
    monkeypatch.setattr(sections, "HALF_WINDOW_S", 1.0)


def _song(parts, sr=SR):
    """Concatenate (amplitude, seconds) blocks into a stereo signal."""
    out = []
    for amp, dur in parts:
        n = int(dur * sr)
        t = np.arange(n) / sr
        out.append(amp * np.sin(2 * np.pi * 220.0 * t) + 0.1 * amp * np.sin(2 * np.pi * 6600.0 * t))
    x = np.concatenate(out)
    return np.stack([x, np.roll(x, 5)], axis=1).astype(np.float32)


def _stem(audio, name="pad.wav", role="pad"):
    st = analyze.Stem(name=name, path=name, audio=audio, sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def test_find_sections_follows_the_level_steps():
    mix = _song([(0.06, 4), (0.20, 4), (0.06, 4)])
    secs = sections.detect(mix, SR)
    assert len(secs) == 3
    assert [s["label"] for s in secs] == ["verse", "chorus", "verse"]
    # boundaries land within a couple of seconds of the real ones
    assert abs(secs[0]["end_s"] - 4.0) < 1.5
    assert abs(secs[1]["end_s"] - 8.0) < 1.5


def test_flat_song_has_one_section():
    secs = sections.detect(_song([(0.1, 10)]), SR)
    assert len(secs) == 1
    assert secs[0]["label"] == "other"


def test_short_song_has_no_sections():
    assert sections.detect(_song([(0.06, 2), (0.2, 2)]), SR) == []


def test_plan_records_bounded_section_scales():
    st = _stem(_song([(0.06, 4), (0.20, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    secs = p["sections"]
    assert len(secs) == 3
    for s in secs:
        assert 0.95 <= s["width_scale"] <= 1.05
        assert 0.85 <= s["send_scale"] <= 1.15
        assert s["start_s"] < s["end_s"]


def test_flat_song_gets_no_sections_in_the_plan():
    st = _stem(_song([(0.1, 10)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert p["sections"] == []


def test_intensity_zero_switches_sections_off():
    st = _stem(_song([(0.06, 4), (0.20, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"), intensity=0.0)
    assert p["sections"] == []


def test_chorus_renders_wider_than_verse():
    st = _stem(_song([(0.06, 4), (0.20, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    mix, _ = render.render([st], p, {"decay_s": 0.5, "predelay_ms": 0.0})

    verse = dsp.width_ratio(mix[int(1 * SR) : int(3 * SR)])
    chorus = dsp.width_ratio(mix[int(5 * SR) : int(7 * SR)])
    assert chorus > verse * 1.03, (verse, chorus)


def test_section_envelope_is_smooth_at_boundaries():
    """A hard step in width is an audible click; the envelope must ramp."""
    env = render._section_env(
        [
            {"start_s": 0.0, "end_s": 2.0, "width_scale": 0.9},
            {"start_s": 2.0, "end_s": 4.0, "width_scale": 1.1},
        ],
        SR * 4,
        SR,
        "width_scale",
    )
    assert env is not None
    # no single-sample jump larger than a small fraction of the whole change
    assert float(np.max(np.abs(np.diff(env)))) < 0.02

# ------------------------------------------------- reference-driven contrast

def _ref_file(tmp_path, name, parts):
    import wave

    out = []
    for amp, dur in parts:
        n = int(dur * SR)
        t = np.arange(n) / SR
        out.append(amp * np.sin(2 * np.pi * 220.0 * t))
    x = np.concatenate(out)
    sig = np.stack([x, np.roll(x, 11)], axis=1)
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    path = tmp_path / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return str(path)


def _sectioned_stem():
    return _stem(_song([(0.06, 4), (0.22, 4), (0.06, 4)]))


def test_strong_reference_increases_the_section_move(tmp_path):
    from automix import reference as refmod

    ref = refmod.analyse(_ref_file(tmp_path, "strong.wav", [(0.05, 4), (0.30, 4), (0.05, 4)]))
    assert ref["section_contrast_db"] > 6.0

    st = _sectioned_stem()
    plain = planner.build_plan([st], planner.load_preset("pop_2020s"))
    tuned = planner.build_plan(
        [st], planner.load_preset("pop_2020s"), reference=ref, blend=1.0
    )

    plain_chorus = next(s for s in plain["sections"] if s["label"] == "chorus")
    tuned_chorus = next(s for s in tuned["sections"] if s["label"] == "chorus")
    assert tuned["section_contrast"]["factor"] > 1.0
    assert tuned_chorus["width_scale"] > plain_chorus["width_scale"]
    # and the factor is bounded
    assert tuned["section_contrast"]["factor"] <= 1.8


def test_flat_reference_switches_section_variation_off(tmp_path):
    from automix import reference as refmod

    ref = refmod.analyse(_ref_file(tmp_path, "flat.wav", [(0.15, 12)]))
    assert ref["section_contrast_db"] == 0.0

    p = planner.build_plan(
        [_sectioned_stem()], planner.load_preset("pop_2020s"), reference=ref, blend=1.0
    )
    assert p["section_contrast"]["factor"] == 0.0
    for s in p["sections"]:
        assert s["width_scale"] == 1.0
        assert s["send_scale"] == 1.0
    assert any("no section contrast" in w for w in p["warnings"])


def test_blend_zero_ignores_the_reference_contrast(tmp_path):
    from automix import reference as refmod

    ref = refmod.analyse(_ref_file(tmp_path, "strong.wav", [(0.05, 4), (0.30, 4), (0.05, 4)]))
    p = planner.build_plan(
        [_sectioned_stem()], planner.load_preset("pop_2020s"), reference=ref, blend=0.0
    )
    # blend 0 means "ignore the reference", so the default move stays
    assert p["section_contrast"]["factor"] == 1.0

def _air_frac(seg):
    m = seg.mean(axis=1)
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / SR)
    band = (f >= 8000) & (f <= 12000)
    return float(spec[band].sum()) / float(spec.sum())


def test_chorus_air_is_lifted_relative_to_no_sections():
    """Per-section tone must actually shift the high band, and upward.

    Measured as a *relative* change: the louder chorus compresses more, so its
    absolute high-band fraction is lower than the verse even after the +0.5 dB
    shelf.  Comparing with and without sections isolates the shelf itself.
    """
    import copy

    st = _stem(_song([(0.06, 4), (0.22, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    with_sec, _ = render.render([st], p, {"decay_s": 0.3, "predelay_ms": 0.0})

    flat = copy.deepcopy(p)
    flat["sections"] = []
    without, _ = render.render([st], flat, {"decay_s": 0.3, "predelay_ms": 0.0})

    def ratio(mix):
        return _air_frac(mix[int(5 * SR) : int(7 * SR)]) / _air_frac(mix[int(1 * SR) : int(3 * SR)])

    assert ratio(with_sec) > ratio(without) * 1.03, (ratio(without), ratio(with_sec))


def test_section_air_defaults_to_zero_for_old_plans():
    """A plan written before `air_db` existed must not gain +1 dB of air."""
    env = render._section_env(
        [{"start_s": 0.0, "end_s": 1.0, "width_scale": 1.0}], SR, SR, "air_db", default=0.0
    )
    assert env is not None and float(np.max(np.abs(env))) == 0.0

# ------------------------------------------- reference section-contour matching

def test_reference_lifts_the_chorus_to_match_its_contour(tmp_path):
    from automix import reference as refmod

    # a reference whose chorus is far louder than its verse
    ref = refmod.analyse(_ref_file(tmp_path, "deep.wav", [(0.05, 4), (0.30, 4), (0.05, 4)]))
    assert ref["section_contrast_db"] > 10

    # ours lifts only slightly
    st = _stem(_song([(0.06, 4), (0.095, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"), reference=ref, blend=1.0)

    by_label = {s["label"]: s["gain_db"] for s in p["sections"]}
    assert by_label["chorus"] > 0
    assert by_label["verse"] < 0
    assert all(abs(v) <= 2.0 for v in by_label.values())


def test_no_reference_means_no_section_gain():
    st = _stem(_song([(0.06, 4), (0.095, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert all(s["gain_db"] == 0.0 for s in p["sections"])


def test_section_gain_moves_the_rendered_contour():
    import copy

    st = _stem(_song([(0.06, 4), (0.095, 4), (0.06, 4)]))
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))

    def ratio(mix):
        def rms(a, b):
            return float(np.sqrt(np.mean(mix[int(a * SR) : int(b * SR)].astype(np.float64) ** 2)))

        return rms(5, 7) / rms(1, 3)

    base, _ = render.render([st], p, {"decay_s": 0.3, "predelay_ms": 0.0})
    pushed = copy.deepcopy(p)
    for s in pushed["sections"]:
        s["gain_db"] = 2.0 if s["label"] == "chorus" else -2.0
    louder, _ = render.render([st], pushed, {"decay_s": 0.3, "predelay_ms": 0.0})

    assert 20 * np.log10(ratio(louder) / ratio(base)) > 3.0
