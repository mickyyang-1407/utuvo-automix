"""R121: under the kick only the bass's low band dips (kick_duck_band_v1).

The bass bus is split just above the kick's fundamental (zero-phase LR4
magnitudes, the two bands add back to the bus exactly) and only the band
below the crossover is ducked, keyed on the kick's own low end.  The bass's
line and attack above the crossover keep their level under every kick.
Plans written before R121 (kick_duck_v1) render as before.
"""
from __future__ import annotations

import copy

import jsonschema
import numpy as np
import pytest

from automix import dsp, plan as planner, render

SR = 48000


def _kick_train(f0=55.0, seconds=8.0, every=0.5, n=None):
    n = n or int(SR * seconds)
    x = np.zeros(n)
    k = np.arange(int(.15 * SR)) / SR
    hit = np.sin(2 * np.pi * f0 * k) * np.exp(-k / .04)
    for s in range(int(.25 * SR), n - hit.size, int(every * SR)):
        x[s: s + hit.size] += hit
    return x


def _hats(seconds=8.0, every=0.25):
    n = int(SR * seconds)
    x = np.zeros(n)
    rng = np.random.default_rng(3)
    h = np.arange(int(.03 * SR)) / SR
    for s in range(int(.25 * SR), n - h.size, int(every * SR)):
        x[s: s + h.size] += .5 * rng.standard_normal(h.size) * np.exp(-h / .008)
    return dsp.apply_eq(x, SR, [("highpass", 4000.0, 0.0, 0.707)])


def _bass(seconds=8.0, lo=50.0, hi=800.0):
    t = np.arange(int(SR * seconds)) / SR
    a = .3 * np.sin(2 * np.pi * lo * t)
    b = .1 * np.sin(2 * np.pi * hi * t)
    return a, b


def _stereo(x):
    return np.stack([x, x], axis=1)


def _spec(**kw):
    s = {"law": "kick_duck_band_v1", "key_ids": ["Kick"], "key_lowpass_hz": 120.0, "depth_db": 4.0,
         "attack_ms": 5.0, "release_ms": 120.0, "crossover_hz": 120.0}
    s.update(kw)
    return s


def _db_in_hits(y, ref, key):
    """Level change inside the key's hits (dB).  Both ducks are linear in the
    bus for a given key, so each component is ducked on its own."""
    hits = np.abs(key) > .3 * np.abs(key).max()
    hits = np.convolve(hits, np.ones(int(.03 * SR)), "same") > 0
    return 10 * np.log10(np.mean(y[hits, 0] ** 2) / np.mean(ref[hits, 0] ** 2))


# ------------------------------------------------------------- the split


@pytest.mark.parametrize("fc", [90.0, 120.0, 160.0])
def test_the_two_bands_add_back_to_the_bus(fc):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR * 2, 2))
    low = dsp.crossover_lowpass(x, SR, fc)
    assert np.allclose((x - low) + low, x, atol=1e-12)
    # LR4 magnitudes: -6 dB at the crossover, the low end kept, the line gone
    t = np.arange(SR * 3) / SR
    mid = slice(SR, 2 * SR)

    def at(hz):
        s = np.sin(2 * np.pi * hz * t)
        return 20 * np.log10(np.std(dsp.crossover_lowpass(s, SR, fc)[mid]) / np.std(s[mid]))
    assert at(fc) == pytest.approx(-6.02, abs=.02)
    assert at(fc / 4) > -.05 and at(fc * 4) < -45.0


def test_the_split_does_not_wrap_around():
    """The low-pass rings both ways; a hit on the song's last sample must not
    show up at its first (the FFT is padded with silence either side)."""
    x = np.zeros(SR * 2)
    x[-1] = 1.0
    low = dsp.crossover_lowpass(x, SR, 90.0)
    assert np.max(np.abs(low[: SR // 2])) < 1e-9


def test_the_crossover_decides_what_dips():
    """A 110 Hz bass note: under a 90 Hz split it is mostly above the ducked
    band, under a 160 Hz split it is in it."""
    t = np.arange(SR * 8) / SR
    note = _stereo(.3 * np.sin(2 * np.pi * 110 * t))
    key = _kick_train()
    lo90, _ = render._kick_duck(note, _stereo(key), SR, _spec(crossover_hz=90.0))
    lo160, _ = render._kick_duck(note, _stereo(key), SR, _spec(crossover_hz=160.0))
    d90, d160 = _db_in_hits(lo90, note, key), _db_in_hits(lo160, note, key)
    assert d160 < -1.0 and d90 > .6 * d160


def test_the_split_keeps_the_callers_dtype_and_length():
    x = np.random.default_rng(2).standard_normal((12345, 2)).astype(np.float32)
    low = dsp.crossover_lowpass(x, SR, 100.0)
    assert low.dtype == np.float32 and low.shape == x.shape


@pytest.mark.parametrize("f0,xo", [(50.0, 90.0), (60.0, 96.0), (80.0, 128.0)])
def test_the_crossover_follows_the_kick(f0, xo):
    found = dsp.kick_fundamental_hz(_kick_train(f0), SR)
    assert abs(found - f0) <= 2.0
    assert dsp.kick_crossover_hz(f0) == pytest.approx(xo)


def test_the_fundamental_is_looked_for_where_a_kick_has_it():
    """Snare body bleeding into the kick mic, louder than the kick's own
    thump: the fundamental is still the kick's (looked for in 40-90 Hz)."""
    t = np.arange(SR * 8) / SR
    key = _kick_train(60.0) + .8 * np.sin(2 * np.pi * 200 * t)
    assert abs(dsp.kick_fundamental_hz(key, SR) - 60.0) <= 2.0


def test_a_key_without_a_low_end_gets_the_default_crossover():
    assert dsp.kick_fundamental_hz(np.zeros(SR * 2), SR) is None
    assert dsp.kick_crossover_hz(None) == 120.0
    assert dsp.kick_crossover_hz(200.0) == 130.0 and dsp.kick_crossover_hz(20.0) == 90.0
    assert dsp.kick_crossover_hz(86.0) == 130.0          # a kit whose low peak sits at 86 Hz


# ------------------------------------------------------------- the duck


V1 = {"law": "kick_duck_v1", "key_ids": ["Kick"], "depth_db": 4.0}


def test_only_the_low_band_dips_under_the_kick():
    a, b = _bass()
    key = _kick_train()
    lo, rec = render._kick_duck(_stereo(a), _stereo(key), SR, _spec())
    hi, _ = render._kick_duck(_stereo(b), _stereo(key), SR, _spec())
    assert _db_in_hits(lo, _stereo(a), key) < -1.5          # the 50 Hz fundamental makes room for the kick
    assert abs(_db_in_hits(hi, _stereo(b), key)) < .01      # the 800 Hz line keeps its level
    assert rec["law"] == "kick_duck_band_v1" and rec["crossover_hz"] == 120.0
    assert -4.0 <= rec["under_hits_db"] <= -2.0 and 0.0 < rec["time_ducked"] < 1.0
    # the whole-bus duck (R111-R120) takes the line down with it
    old, _ = render._kick_duck(_stereo(b), _stereo(key), SR, V1)
    assert _db_in_hits(old, _stereo(b), key) < -1.5


def test_where_nothing_ducks_the_bus_is_untouched_to_the_bit():
    a, b = _bass()
    bus = _stereo(a + b).astype(np.float32)
    out, rec = render._kick_duck(bus, np.zeros_like(bus), SR, _spec())
    assert np.array_equal(out, bus) and rec["time_ducked"] == 0.0
    # between the hits the gain is back to 1 and the bus is the input, exactly
    key = _kick_train(every=4.0)                      # one hit at 0.25 s, one at 4.25 s
    out, _ = render._kick_duck(bus, _stereo(key), SR, _spec())
    quiet = slice(int(2.0 * SR), int(4.0 * SR))
    assert np.array_equal(out[quiet], bus[quiet])


def test_a_hihat_in_the_kick_mic_does_not_open_it():
    """Kick on every second, a hi-hat as loud as the kick half a second
    later (bleed into the kick mic).  A steady bus shows the gain itself."""
    kick = _kick_train(every=1.0)
    hats = _hats(every=1.0)
    hats = np.roll(hats, int(.5 * SR)) * (np.abs(kick).max() / np.abs(hats).max())
    key = _stereo(kick + hats)
    bus = np.full((kick.size, 2), .5)
    band, _ = render._kick_duck(bus, key, SR, _spec())
    old, _ = render._kick_duck(bus, key, SR, V1)
    at_hats = np.zeros(kick.size, bool)
    for s in range(int(.75 * SR), kick.size - SR, SR):           # 0.75 s, 1.75 s, ...
        at_hats[s: s + int(.1 * SR)] = True
    gain_db = lambda y: 20 * np.log10(np.min(y[at_hats, 0]) / .5)
    assert gain_db(band) > -.2            # the kick's low end is the key: a hat cannot open it
    assert gain_db(old) < -.5             # the full-band key (R111-R120) opens on every hat


def test_kick_duck_v1_is_unchanged():
    a, b = _bass()
    bus = _stereo(a + b)
    key = _stereo(_kick_train())
    out, rec = render._kick_duck(bus, key, SR, {**V1, "depth_db": 3.0})
    ref, _ = dsp.duck(bus, key, SR, 3.0, attack_ms=5.0, release_ms=120.0)
    assert np.array_equal(out, ref) and "crossover_hz" not in rec


@pytest.mark.parametrize("bad", [
    {"crossover_hz": None}, {"crossover_hz": 30.0}, {"crossover_hz": float("nan")},
    {"crossover_hz": True}, {"key_lowpass_hz": None}, {"depth_db": 20.0}, {"release_ms": 5.0},
    {"law": "kick_duck_band_v9"},
])
def test_a_wrong_band_duck_is_refused(bad):
    a, b = _bass(seconds=1.0)
    bus = _stereo(a + b)
    with pytest.raises(render.RenderError):
        render._kick_duck(bus, _stereo(_kick_train(seconds=1.0)), SR, _spec(**bad))


# ------------------------------------------------------------- the plan


def _session():
    from test_mix_moves import _session as mm
    return mm()


def test_a_band_duck_without_its_crossover_is_not_a_plan():
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(p)
    q = copy.deepcopy(p)
    del q["buses"]["bass"]["duck"]["crossover_hz"]
    with pytest.raises((planner.PlanError, jsonschema.ValidationError)):
        planner.validate(q)
    q = copy.deepcopy(p)
    q["buses"]["bass"]["duck"]["crossover_hz"] = 400.0
    with pytest.raises((planner.PlanError, jsonschema.ValidationError)):
        planner.validate(q)


def test_the_render_ducks_the_low_band_and_says_so():
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    mix, report = render.render(stems, p)
    m = report["mix_moves"]["kick_duck"]
    assert m["law"] == "kick_duck_band_v1" and m["crossover_hz"] == p["buses"]["bass"]["duck"]["crossover_hz"]
    assert m["under_hits_db"] < -1.0 and np.all(np.isfinite(mix))
    rows = render._move_rows(report["mix_moves"])
    assert any("below 90 Hz only, the bass above it untouched" in r for r in rows)
