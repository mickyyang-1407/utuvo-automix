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
#: 2026-11-02 (R111): pop_2020s now asks for three more mixing moves: the
#: bass dips under the kick (up to 2 dB; measured -1.65 dB on the hits), the
#: drum bus gets parallel compression (envelope crest 14.4 -> 12.7 dB) and
#: the vocal a 1/4-note delay locked to the demo's 100 BPM (600 ms).  Less
#: bass under the kick: low .3412 -> .3129; the delay adds vocal energy:
#: mid .1486 -> .1579, presence .0219 -> .0249, air .0294 -> .0374.  LUFS
#: -10.66 -> -10.61, crest 11.41 -> 11.54, width .122 -> .120.  Intended.
#: Plans without the three blocks render exactly as before.
#: 2026-11-04 (R114, listening notes on Pop/Metal: vocal a little small, drums and bass
#: too big, a bit clipped, too dry, image too small): drums and bass are now
#: balanced against the band (46 MedleyDB mixes: drums -5.2, bass -3.9, pop
#: -0.5 each) -- the synthetic demo's drums sat at +0.4 dB and come down
#: 5.0 dB, its sine bass sat at -8.7 and comes up 4.6; the vocal target is
#: +0.1 (median of 46 mixes -1.9, pop +1.5, lift +0.5); the vocal bus gets
#: parallel density, saturation and body/presence/air; plate_v2 reverb with
#: more on the vocal; a release limiter; side air +1.5 dB.  Less drum bus
#: (where the demo's hats live): air .0374 -> .0127, low .3129 -> .2687;
#: more bass: low_mid .4399 -> .5128; mid .1579 -> .1739; width .120 ->
#: .159 (side air, wider reverb); correlation .974 -> .956; LUFS -10.61 ->
#: -10.56; crest 11.54 -> 11.40.  Intended.  Saved plans render as before
#: (R25 comparison, replay of R94/R104).
#: 2026-11-04 (R116): the balance steps measure through the master EQ (what
#: the listener hears; pop's +1.5 dB low shelf had left kick and bass above
#: their targets), the drum/bass cut may go to 12 dB (the demo's drums need
#: 6.19, the old limit stopped at 6), the reverb is locked to the demo's
#: 100 BPM (decay 1.6 -> 1.2 s = 2 beats, pre-delay 28 -> 37.5 ms = 1/16
#: beat, a 1/64 note) and the drum parallel blend is halved (.35 -> .175).  Less kick and
#: bass as heard: low .2687 -> .2470, low_mid .5128 -> .5353, mid .1739 ->
#: .1757, presence .0228 -> .0249, air .0127 -> .0099; width .159 -> .178;
#: correlation .956 -> .940; LUFS -10.56 -> -10.54; crest 11.40.  Intended.
#: Saved plans render as before (R25 comparison, replay of R94/R104/R115).
#: 2026-11-05 (R117): the drum, bass and vocal reverb/delay sends follow
#: their balance moves (post_fader_v1: the demo's drums come down 8.0 dB
#: and their wide plate with them, the vocal 3.6 dB), saturation follows
#: each passage's level at 4x (env_tanh_os4_v1; the old law also lifted the
#: saturated tracks' level), and the drums are compressed less (bus
#: threshold 6 dB above its average, at most 2:1, 30 ms attack; close mics
#: half the ratio, 25 ms attack).  Less wide plate: width .178 -> .145,
#: correlation .940 -> .961; low .2470 -> .2587, mid .1757 -> .1676,
#: presence .0249 -> .0233, air .0099 -> .0080; LUFS -10.54 -> -10.57,
#: crest 11.40 -> 11.36.  Intended.  Saved plans render as before (replay of
#: R94/R104/R115/R116).
#: 2026-11-05 (R119): every compressor's detector reads only the past
#: (compressor_law trailing_box_v1; the centred one looked half a window
#: ahead and shaved each hit's start, -10.6 dB on a synthetic kick's first
#: 5 ms against -7.0 on its body).  The demo's synthetic drums and pulses now
#: keep their attack: crest 11.36 -> 12.49, and the release limiter backs
#: off 1.4 dB rather than squash them (LUFS -10.57 -> -11.58; on 15 real
#: MedleyDB songs the law moves LUFS by -0.05 dB, at most -0.31).  The vocal reverb steps
#: back 4 dB while the voice sings and the return is lifted 4 dB (reverb.duck):
#: as wet as before under the words, more tail in the gaps.  Low .2587 ->
#: .2191, low_mid .5365 -> .5597, mid .1676 -> .1808, presence .0233 ->
#: .0254, air .0080 -> .0086; width .145 -> .149; correlation .961 -> .958.
#: Intended.  Saved plans render as before (replay of R94/R104/R115-R118).
EXPECTED = {
    "lufs": -11.58,
    "crest_db": 12.49,
    "width": 0.149,
    "correlation": 0.958,
    "band_energy": {
        "low": 0.2191,
        "low_mid": 0.5597,
        "mid": 0.1808,
        "presence": 0.0254,
        "air": 0.0086,
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


def test_new_plans_only_fall_short_by_the_squash_guard(demo_mix):
    """R119: compressors that read only the past keep the demo's attacks, so
    the release limiter's squash guard backs the master off (1.4 dB here)
    rather than flatten them.  The miss is that back-off and nothing more,
    the back-off stays within the plan's cap, the loudest 5% stay within the
    guard, and the report says so."""
    plan, _, report = demo_mix
    lim = plan["buses"]["master"]["limiter"]
    target = plan["buses"]["master"]["target_lufs"]
    backed = float(report["limiter"].get("backed_off_db", 0.0))
    assert backed <= float(lim["max_backoff_db"]) + 1e-9
    assert target - report["lufs"] <= backed + 0.3
    assert report["limiter"]["gain_reduction_p95_db"] >= -float(lim["max_gr_p95_db"]) - 0.2
    if backed > 0:
        assert any(w.startswith("loudness:") for w in report["warnings"])
