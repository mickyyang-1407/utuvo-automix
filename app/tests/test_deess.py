"""De-esser tests.

The promise: harsh S sounds come down, the body of the voice does not move, and
nothing happens to material that is not a sibilant vocal.
"""

from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def vocal(sib_amp=0.8, dur=4.0, seed=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = 0.25 * np.sin(2 * np.pi * 220.0 * t) + 0.12 * np.sin(2 * np.pi * 440.0 * t)
    rng = np.random.default_rng(seed)
    sib = np.zeros(n)
    for start in np.arange(0.4, dur - 0.4, 0.8):
        i = int(start * SR)
        d = int(0.12 * SR)
        sib[i : i + d] += rng.standard_normal(d) * np.hanning(d) * sib_amp
    return np.stack([body + sib, body + sib], axis=1).astype(np.float32)


def band(x, lo, hi):
    m = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / SR)
    return float(spec[(f >= lo) & (f <= hi)].sum())


def test_deess_only_touches_the_high_band():
    sig = vocal(0.8)
    out = dsp.deess(sig, SR, amount=0.8)
    assert band(out, 6000, 9000) / band(sig, 6000, 9000) < 0.75
    # the body of the voice must be essentially untouched
    assert abs(band(out, 200, 1000) / band(sig, 200, 1000) - 1.0) < 0.02


def test_deess_barely_moves_clean_material():
    """A clean voice has almost nothing in the sibilance zone to begin with.

    Compared as an energy *fraction*: the absolute 6-9 kHz energy of a clean
    220/440 Hz tone is numerical dust, so a plain ratio of two dust piles says
    nothing (it came out as 7161x on the first attempt).
    """
    sig = vocal(sib_amp=0.0)
    out = dsp.deess(sig, SR, amount=0.8)

    def frac(x):
        total = band(x, 20, 20000)
        return band(x, 6000, 9000) / total

    assert frac(sig) < 1e-4 and frac(out) < 1e-4
    rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    assert abs(rms(out) / rms(sig) - 1.0) < 0.02


def test_sibilance_index_separates_harsh_from_clean():
    def measure(amp):
        st = analyze.Stem(name="vocal_lead.wav", path="v", audio=vocal(amp), sr=SR)
        analyze.detect_roles([st])
        analyze.measure(st)
        return st.metrics["sibilance"]

    assert measure(0.0) < 0.01
    assert measure(0.8) > 0.05


def test_plan_adds_a_deesser_for_a_sibilant_vocal_only():
    harsh = analyze.Stem(name="vocal_lead.wav", path="v", audio=vocal(0.8), sr=SR)
    clean = analyze.Stem(name="vocal_lead2.wav", path="v2", audio=vocal(0.0), sr=SR)
    analyze.detect_roles([harsh])
    analyze.measure(harsh)
    analyze.detect_roles([clean])
    analyze.measure(clean)

    p_harsh = planner.build_plan([harsh], planner.load_preset("pop_2020s"))
    assert "deess" in p_harsh["tracks"][0]
    assert any("de-esser" in e for e in p_harsh["tracks"][0]["evidence"])

    p_clean = planner.build_plan([clean], planner.load_preset("pop_2020s"))
    assert "deess" not in p_clean["tracks"][0]


def test_non_vocal_tracks_never_get_a_deesser():
    st = analyze.Stem(name="guitar_elec.wav", path="g", audio=vocal(0.8), sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    assert st.role != "lead_vocal"
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    assert "deess" not in p["tracks"][0]


def test_render_reduces_sibilance_without_moving_the_body():
    st = analyze.Stem(name="vocal_lead.wav", path="v", audio=vocal(0.8), sr=SR)
    analyze.detect_roles([st])
    analyze.measure(st)
    p = planner.build_plan([st], planner.load_preset("pop_2020s"))
    on, _ = render.render([st], p, {"decay_s": 0.3, "predelay_ms": 0.0})

    import copy

    off_plan = copy.deepcopy(p)
    off_plan["tracks"][0].pop("deess", None)
    off, _ = render.render([st], off_plan, {"decay_s": 0.3, "predelay_ms": 0.0})

    assert band(on, 6000, 9000) / band(off, 6000, 9000) < 0.85
    # the body moves a little because this compares two *full renders*: pulling
    # the sibilance down changes what the limiter does to the rest.  The band
    # itsself is untouched, which `test_deess_only_touches_the_high_band` checks.
    assert abs(20 * np.log10(band(on, 200, 1000) / band(off, 200, 1000))) < 1.5
