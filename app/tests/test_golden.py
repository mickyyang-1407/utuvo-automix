"""Golden-draft test: the demo mix must keep sounding like itself.

Every other test asks "does this rule work?".  This one asks "did any of that
work make the *output* different in a way nobody intended?"  It is deliberately
coarse -- tolerances are wide enough for honest refactors and tight enough to
catch a mix that suddenly gets duller, narrower or squashed.

If a change is meant to alter the mix, update the numbers *and say why* in the
commit message.  A silent update here defeats the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from automix import analyze, plan as planner, pipeline

DEMO = str(Path(__file__).resolve().parents[1] / "demo_stems")
SR = 48000

#: measured on the current engine; see the module docstring before editing.
#:
#: 2026-10-29: updated after fixing the saturation curve (``1 + 6*drive`` ->
#: ``5*drive``).  The old mapping put 13% THD on a vocal at drive 0.1, which is
#: fuzz rather than warmth; with the fix the same presets produce ~2%.  That
#: removes harmonic energy that used to land in the low-mid band, so the balance
#: shifts (low 0.356 -> 0.405, low_mid 0.446 -> 0.360) and the mix narrows
#: slightly (0.124 -> 0.107).  This is the intended change, not drift.
#: 2026-10-29 (fourth update): the preset's reverb pre-delay was never wired,
#: so every style used 20 ms; pop now uses its own 28 ms (IR onset measured at
#: exactly 28 ms).  Width 0.109 -> 0.110, presence/air shift by <0.003.
#:
#: 2026-10-29 (third update): the music bus was applying the style's width a
#: second time on top of the per-track width.  Track width now owns it; side/mid
#: moves 0.113 -> 0.109.  No other metric changes.
#:
#: 2026-10-29 (second update): bus compressors were measured doing -0.00 dB of
#: gain reduction because their thresholds were computed from a formula rather
#: than from the bus's own level.  Now that they engage (bass ~-1.8 dB, drum
#: ~-1.7 dB, music ~-0.7 dB), the balance shifts again slightly.  Intended.
#:
#: 2026-10-29 (fifth update): the gain staging no longer replaces the source's
#: own balance -- it blends with it (balance_preserve, default 0.5).  Measured
#: on 11 real multitracks (MUSDB18), the professional's balance put the lead
#: vocal 5.6 dB above the drums and the accompaniment within a few dB of both,
#: while the style's absolute balance moved the accompaniment 14.4 dB away
#: from what the engineer chose.  The demo stems carry their own balance, so
#: keeping half of it moves the mix off the bass (low 0.366 -> 0.309) and into
#: the low mid (0.391 -> 0.465).  Intended.
#: 2026-10-29 (sixth update): post-track matching now restores the planned
#: active-window LUFS after role-dependent EQ/compression/saturation. k is
#: applied only to the target balance. This restores vocal/mid energy rather
#: than letting the DSP move it away from the plan: low_mid .465 -> .440,
#: mid .142 -> .196, width .122 -> .105. Tolerances remain unchanged.
#: 2026-11-01 (R104): compressor blocks now carry ratio_law and new plans
#: compress at the written ratio (literal_v1).  Before, a written 3.25:1 acted
#: as 2.01:1 because the ratio was mapped twice.  New plans compress as
#: the numbers say.  The demo's kick/bass/master now compress harder, so low
#: energy drops .301 -> .265 and low_mid .440 -> .456, mid .196 -> .211;
#: LUFS -10.63 -> -10.57, crest 11.48 -> 11.54, width .105 -> .106,
#: correlation .981 -> .980.  Intended.  Saved plans without ratio_law still
#: render exactly as before.
#: 2026-11-02 (R109): the vocal bus is moved to where a professional mix puts
#: the vocal against the rest (median of 11 MUSDB18 mixes measured with the
#: renderer's own meter, -2.7 dB, plus the style's vocal_forward_db: -1.2 dB
#: for pop_2020s), measured after bus compression on the dry buses.  The
#: synthetic demo vocal sat at +1.81 dB, so it comes down 3.01 dB: low
#: .2645 -> .3412, mid .211 -> .1486, low_mid .4557 -> .4373, LUFS -10.57 ->
#: -10.66, crest 11.54 -> 11.41, width .106 -> .122, correlation .980 ->
#: .974.  Intended.  Saved plans without the vocal balance block render
#: exactly as before.
EXPECTED = {
    "lufs": -10.66,
    "crest_db": 11.41,
    "width": 0.122,
    "correlation": 0.974,
    "band_energy": {
        "low": 0.3412,
        "low_mid": 0.4373,
        "mid": 0.1486,
        "presence": 0.0219,
        "air": 0.0294,
    },
}


@pytest.fixture(scope="module")
def demo_mix():
    stems = analyze.load_stems(DEMO)
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan, audio, report = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    return plan, audio, report


def test_loudness_and_headroom_hold(demo_mix):
    plan, audio, report = demo_mix
    assert abs(report["lufs"] - EXPECTED["lufs"]) < 0.3
    assert report["true_peak_db"] <= plan["buses"]["master"]["true_peak_db"] + 0.2


def test_dynamic_character_holds(demo_mix):
    _, _, report = demo_mix
    assert abs(report["crest_db"] - EXPECTED["crest_db"]) < 1.0


def test_stereo_image_holds(demo_mix):
    _, _, report = demo_mix
    assert abs(report["width"] - EXPECTED["width"]) < 0.03
    assert abs(report["correlation"] - EXPECTED["correlation"]) < 0.03


def test_spectral_balance_holds(demo_mix):
    _, audio, _ = demo_mix
    got = analyze._band_fractions(audio, SR)
    for band, want in EXPECTED["band_energy"].items():
        # 0.02 absolute: enough for honest refactors, tight enough that the
        # air band dropping from 0.038 to 0.011 (the lofi preset) fails
        assert abs(got[band] - want) < 0.02, (band, got[band], want)


def test_every_track_explains_itself(demo_mix):
    plan, _, report = demo_mix
    assert len(plan["tracks"]) == len(report["tracks"])
    for track in plan["tracks"]:
        assert track.get("evidence"), track["id"]
        assert -18.0 <= track["gain_db"] <= 18.0
