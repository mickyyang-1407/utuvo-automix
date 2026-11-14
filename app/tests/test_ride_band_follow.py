"""R132: slow rides follow the band's sections instead of flattening them.

A ride used to hold every sustained stem at its own median: the guitar that
gets louder with everybody else in the chorus was ridden down there and up in
the verse, which took the arrangement's build out of the song.  New plans
re-plan every ride once all gains are final: the stem is held at its usual
distance from the rest of the band (the median of stem - rest where both
play), weighted ``follow`` against the old own-median target.  A stem that
moves with the band is left alone; a stem that stays put while the band rises
is brought along; where the band drops out the stem is left as it is; a stem
that never plays with a band keeps the old ride.  Receipt ``ride_follow``
(law band_follow_v1); the renderer reads only the ride points, so older plans
render as they were.

Numbers are written out on purpose (follow 1.0, the 40 dB live window, the
-120 dB silence floor, the 2.5 dB cap): a test that reads the planner's
constant passes with it broken.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000
SECONDS = 48.0
CHORUS_DB = 8.0


def _section_env(n: int, rise_db: float, at: float = SECONDS / 2, sr: int = SR) -> np.ndarray:
    """Quiet before ``at`` seconds, ``rise_db`` louder after, 1 s crossfade."""
    t = np.arange(n) / sr
    x = np.clip((t - at + 0.5) / 1.0, 0.0, 1.0)
    return 10 ** (rise_db * x / 20)


def _tone(freq: float, n: int, seed: int, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)) * (1 + 0.05 * rng.standard_normal(n))


def _stems(parts, sr: int = SR):
    stems = [analyze.Stem(name=name, path=name, sr=sr, audio=np.stack([a, a], 1).astype(np.float32))
             for name, a in parts]
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    kick = np.zeros(n)
    for s in range(0, n - 9600, SR // 2):
        tt = t[:9600]
        kick[s:s + 9600] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt / .08)
    return kick


def _session(keys_rise: float = 0.0, with_band: bool = True, band_env=None):
    n = int(SECONDS * SR)
    band = _section_env(n, CHORUS_DB) if band_env is None else band_env
    parts = [("Electric Guitar.wav", .08 * band * _tone(196, n, 1)),
             ("Keys.wav", .05 * _section_env(n, keys_rise) * _tone(330, n, 2))]
    if with_band:
        parts = [("Kick.wav", .5 * _kick(n) * band), ("Bass.wav", .1 * band * _tone(55, n, 3))] + parts
    return _stems(parts)


def _track(p, role):
    (t,) = [t for t in p["tracks"] if t["role"] == role]
    return t


def _ride_at(t, when: float) -> float:
    pts = np.asarray(t.get("ride") or [[0.0, 0.0]], dtype=float)
    return float(np.interp(when, pts[:, 0], pts[:, 1]))


def _plan(stems, follow=None, **kw):
    if follow is None:
        return planner.build_plan(stems, planner.load_preset("pop_2020s"), **kw)
    mp = pytest.MonkeyPatch()
    mp.setattr(planner, "RIDE_FOLLOW_K", follow)
    try:
        return planner.build_plan(stems, planner.load_preset("pop_2020s"), **kw)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def session():
    stems = _session()
    return stems, _plan(stems)


@pytest.fixture(scope="module")
def old_plan(session):
    return _plan(session[0], follow=0.0)


# ------------------------------------------------------------ what it does

def test_a_stem_that_rises_with_the_band_is_left_alone(session, old_plan):
    _, p = session
    new, old = _track(p, "guitar_electric"), _track(old_plan, "guitar_electric")
    # the old law rode the build out: up in the verse, down in the chorus
    assert _ride_at(old, 8.0) > 1.5 and _ride_at(old, 40.0) < -1.5
    for when in (4.0, 8.0, 12.0, 32.0, 36.0, 40.0):
        assert abs(_ride_at(new, when)) < 0.5, when


def test_a_stem_that_stays_put_is_brought_along(session):
    _, p = session
    keys = _track(p, "keys")
    assert _ride_at(keys, 40.0) > 1.5
    assert _ride_at(keys, 8.0) < -1.5
    assert max(g for _, g in keys["ride"]) <= 2.5 + 1e-9
    assert min(g for _, g in keys["ride"]) >= -2.5 - 1e-9


def test_the_section_contrast_survives_the_render(session, old_plan):
    stems, p = session
    a, _ = render.render(stems, old_plan)
    b, _ = render.render(stems, p)

    def contrast(x):
        m = np.asarray(x, dtype=np.float64).mean(axis=1)
        verse = m[int(4 * SR):int(20 * SR)]
        chorus = m[int(28 * SR):int(44 * SR)]
        return 10 * np.log10(np.mean(chorus ** 2) / np.mean(verse ** 2))

    assert contrast(b) > contrast(a) + 0.5


def test_where_the_band_drops_out_the_stem_is_left_alone():
    # verse quiet with a quiet band, chorus loud with a loud band, then a
    # breakdown: the guitar alone, quiet again, the band silent
    n = int(SECONDS * SR)
    t = np.arange(n) / SR
    guitar_env = np.where((t >= 12) & (t < 40), 3.5, 1.0)
    band_env = np.where(t < 12, 1.0, np.where(t < 40, 3.5, 0.0))
    stems = _stems([("Electric Guitar.wav", .1 * guitar_env * _tone(196, n, 1)),
                    ("Bass.wav", .1 * band_env * _tone(55, n, 3))])
    p, old = _plan(stems), _plan(stems, follow=0.0)
    g, g_old = _track(p, "guitar_electric"), _track(old, "guitar_electric")
    # the old ride pushed the lone quiet guitar up toward its (chorus) median
    assert _ride_at(g_old, 45.0) > 1.5
    for when in (6.0, 25.0, 45.0):
        assert abs(_ride_at(g, when)) < 0.5, when


# ------------------------------------------------------------ receipts

def test_receipt_and_evidence(session, old_plan):
    _, p = session
    for role in ("guitar_electric", "keys", "bass"):
        t = _track(p, role)
        assert t["ride_follow"] == {"law": "band_follow_v1", "follow": 1.0}
        lines = [e for e in t["evidence"] if e.startswith("ride ±")]
        assert len(lines) == 1 and lines[0].endswith("following the rest of the band")
    assert "ride_follow" not in _track(p, "kick")
    assert not any("ride_follow" in t for t in old_plan["tracks"])
    planner.validate(p)
    for bad in ({"law": "band_follow_v0", "follow": 1.0}, {"law": "band_follow_v1", "follow": 1.0, "x": 1},
                {"law": "band_follow_v1", "follow": 1.5}, {"law": "band_follow_v1"}):
        q = copy.deepcopy(p)
        _track(q, "keys")["ride_follow"] = bad
        with pytest.raises(Exception):
            planner.validate(q)


def test_the_receipt_is_not_read_by_the_renderer(session):
    stems, p = session
    q = copy.deepcopy(p)
    for t in q["tracks"]:
        t.pop("ride_follow", None)
    a, _ = render.render(stems, p)
    b, _ = render.render(stems, q)
    assert np.array_equal(a, b)


def test_follow_zero_plans_exactly_as_before(session, old_plan):
    stems, _ = session
    # the pass is off: the plan is the old plan to the last evidence line
    again = _plan(stems, follow=0.0)
    strip = lambda p: {k: v for k, v in p.items() if k != "created_at"}  # noqa: E731
    assert strip(again) == strip(old_plan)
    assert all(not e.endswith("following the rest of the band") for t in old_plan["tracks"] for e in t["evidence"])


# ------------------------------------------------------------ the curve itself

def test_with_no_band_the_ride_is_the_old_one():
    n = int(SECONDS * SR)
    (stem,) = _stems([("Keys.wav", .05 * _section_env(n, 8.0) * _tone(330, n, 2))])
    old = planner._ride_curve(stem, 2.5)
    assert len(old) > 2  # a real, moving ride
    info = {}
    assert planner._ride_curve(stem, 2.5, None, 1.0, info=info) == old and not info
    assert planner._ride_curve(stem, 2.5, np.full(400, np.nan), 1.0, info=info) == old and not info
    lv = [planner._stem_levels(stem)]
    assert planner._rest_level(lv, 0, [0.0]) is None
    # and a lone stem in a plan has no receipt
    p = _plan([stem])
    assert "ride" in p["tracks"][0] and "ride_follow" not in p["tracks"][0]


def test_follow_zero_is_the_old_curve():
    n = int(SECONDS * SR)
    (stem,) = _stems([("Keys.wav", .05 * _section_env(n, 8.0) * _tone(330, n, 2))])
    old = planner._ride_curve(stem, 2.5)
    assert len(old) > 2
    _, db = dsp.short_term_db(stem.audio, SR, 1.0, 0.25)
    assert planner._ride_curve(stem, 2.5, db - 9.0 + np.linspace(-6, 6, db.size), 0.0) == old


def test_a_bimodal_song_holds_the_distance_not_the_two_medians():
    # the stem's verse is 52% of the song, the band's verse 48%: the stem's
    # own median is its verse level, the band's median its chorus level.
    # Their difference (2 dB) is not the distance the stem keeps (10 dB).
    k = 400
    times = 0.5 + 0.25 * np.arange(k)
    own = np.where(np.arange(k) < 208, -20.0, -12.0)
    rest = np.where(np.arange(k) < 192, -30.0, -22.0)
    pts = planner._ride_curve(None, 2.5, rest, 1.0, levels=(times, own))
    g = np.interp(times, *np.asarray(pts).T)
    assert np.all(np.abs(g[20:180]) < 0.3) and np.all(np.abs(g[220:380]) < 0.3)


def test_inf_and_nan_never_reach_the_ride():
    k = 300
    times = 0.5 + 0.25 * np.arange(k)
    own = np.where(np.arange(k) < 150, -20.0, -12.0)
    for rest in (np.full(k, np.inf), np.full(k, -np.inf), np.where(np.arange(k) % 7 == 0, np.nan, -25.0)):
        pts = planner._ride_curve(None, 2.5, rest, 1.0, levels=(times, own))
        assert pts and np.all(np.isfinite(np.asarray(pts)))


def _independent_rest(levels, own, gains):
    times = levels[own][0]
    total = np.zeros(times.size)
    for i, (t, db) in enumerate(levels):
        if i == own:
            continue
        p = 10 ** ((db + gains[i]) / 10)
        total += np.interp(times, t, p, left=0.0, right=0.0)
    rest = 10 * np.log10(np.maximum(total, 1e-30))
    return np.where(rest > rest.max() - 40.0, rest, np.nan)


def test_the_rest_by_hand():
    rng = np.random.default_rng(7)
    t = lambda k: 0.5 + 0.25 * np.arange(k)  # noqa: E731
    levels = [(t(200), rng.uniform(-40, -10, 200)),
              (t(150), rng.uniform(-30, -12, 150)),  # ends early
              (t(200), np.concatenate([rng.uniform(-25, -15, 100), np.full(100, -120.0)]))]  # goes quiet
    gains = [0.0, 3.0, -6.0]
    for own in range(3):
        got = planner._rest_level(levels, own, gains)
        want = _independent_rest(levels, own, gains)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], atol=1e-9)
    # past the end of the only other stem there is no band
    two = [(t(200), np.zeros(200)), (t(100), np.full(100, -10.0))]
    got = planner._rest_level(two, 0, [0.0, 0.0])
    assert np.all(np.isnan(got[101:])) and np.allclose(got[:99], -10.0)
    # 40 dB under the rest's loudest is no band
    two = [(t(10), np.zeros(10)), (t(10), np.array([-10.0] * 5 + [-55.0] * 5))]
    got = planner._rest_level(two, 0, [0.0, 0.0])
    assert np.all(np.isnan(got[5:])) and np.allclose(got[:5], -10.0)
    # a rest that never rises above -120 dB is silence
    assert planner._rest_level([(t(4), np.zeros(4)), (t(4), np.full(4, -125.0))], 0, [0.0, 0.0]) is None


def test_a_broken_file_does_not_wipe_the_band():
    n = int(20 * SR)
    a = .05 * _tone(330, n, 1)
    b = .08 * _tone(196, n, 2)
    b[1000] = np.nan
    stems = [analyze.Stem(name=nm, path=nm, sr=SR, audio=np.stack([x, x], 1).astype(np.float32))
             for nm, x in (("Keys.wav", a), ("Electric Guitar.wav", b))]
    levels = [planner._stem_levels(s) for s in stems]
    assert all(np.all(np.isfinite(db)) for _, db in levels)
    r = planner._rest_level(levels, 0, [0.0, 0.0])
    assert np.isfinite(r).sum() > 0.9 * r.size
    # and nothing overflows
    t = 0.5 + 0.25 * np.arange(10)
    with np.errstate(over="raise"):
        r = planner._rest_level([(t, np.zeros(10)), (t, np.full(10, 4000.0)), (t, np.full(10, -20.0))],
                                0, [0.0, 0.0, 6.0])
    assert r is not None and np.all(np.isfinite(r)) and np.allclose(r, 300.0)


def test_stems_at_different_rates_line_up_by_time():
    n48, n24 = int(30 * SR), int(30 * 24000)
    env48 = _section_env(n48, 10.0, at=15.0)
    env24 = _section_env(n24, 10.0, at=15.0, sr=24000)
    keys = _stems([("Keys.wav", .05 * _tone(330, n48, 1))])[0]
    g48 = _stems([("Electric Guitar.wav", .08 * env48 * _tone(196, n48, 2))])[0]
    g24 = _stems([("Electric Guitar.wav", .08 * env24 * _tone(196, n24, 2, sr=24000))], sr=24000)[0]
    a = planner._rest_level([planner._stem_levels(keys), planner._stem_levels(g48)], 0, [0.0, 0.0])
    b = planner._rest_level([planner._stem_levels(keys), planner._stem_levels(g24)], 0, [0.0, 0.0])
    ok = ~np.isnan(a) & ~np.isnan(b)
    assert ok.sum() > 0.9 * a.size
    assert np.max(np.abs(a[ok] - b[ok])) < 0.5


# ------------------------------------------------------------ inside build_plan

def test_the_band_is_weighed_at_the_final_gains():
    # the guitar is steady; only the keys build.  A +15 dB trim on the keys
    # makes them the band (the guitar is brought along up to the cap); a
    # -15 dB trim takes them out of it (the guitar barely moves).
    n = int(SECONDS * SR)
    stems = _stems([("Electric Guitar.wav", .08 * _tone(196, n, 1)),
                    ("Keys.wav", .03 * _section_env(n, 10.0) * _tone(330, n, 2)),
                    ("Bass.wav", .08 * _tone(55, n, 3))])
    keys_id = analyze.track_id("Keys.wav")
    up = _track(_plan(stems, trims={keys_id: 15.0}), "guitar_electric")
    down = _track(_plan(stems, trims={keys_id: -15.0}), "guitar_electric")
    assert _ride_at(up, 40.0) - _ride_at(up, 8.0) > 3.0
    assert abs(_ride_at(down, 40.0) - _ride_at(down, 8.0)) < 1.0


def test_capture_group_members_keep_no_ride_and_no_receipt():
    n = int(SECONDS * SR)
    band = _section_env(n, CHORUS_DB)
    stems = _stems([("Guitar Mic1.wav", .08 * band * _tone(196, n, 1)),
                    ("Guitar Mic2.wav", .06 * band * _tone(196, n, 1)),
                    ("Keys.wav", .05 * _tone(330, n, 2)),
                    ("Bass.wav", .1 * band * _tone(55, n, 3))])
    ids = [analyze.track_id(s.name) for s in stems[:2]]
    p = _plan(stems, capture_groups=[{"id": "gtr", "members": ids}])
    for t in p["tracks"]:
        if t["id"] in ids:
            assert "ride" not in t and "ride_follow" not in t
    keys = _track(p, "keys")
    assert keys["ride_follow"]["law"] == "band_follow_v1"
    planner.validate(p)


def test_the_vocal_fit_sees_the_followed_ride(monkeypatch):
    n = int(SECONDS * SR)
    rng = np.random.default_rng(9)
    syl, gap = int(0.38 * SR), int(0.02 * SR)
    env = np.zeros(n)
    for start in range(0, n - syl, syl + gap):  # sung syllables, 10 dB apart at most
        env[start:start + syl] = 10 ** (-rng.uniform(0.0, 10.0) / 20)
    stems = _session()
    stems += _stems([("Lead Vocal.wav", .25 * env * _tone(220, n, 9))])
    seen = {}
    real = planner._fit_vocal_comp

    def spy(stem, gain_db, ride, *a, **k):
        seen[stem.name] = copy.deepcopy(ride)
        return real(stem, gain_db, ride, *a, **k)

    monkeypatch.setattr(planner, "_fit_vocal_comp", spy)
    p = _plan(stems)
    lead = _track(p, "lead_vocal")
    assert lead["ride_follow"]["law"] == "band_follow_v1"
    assert seen["Lead Vocal.wav"] == lead["ride"]


def test_zero_intensity_still_rides_nothing():
    p = _plan(_session(), intensity=0.0)
    assert not any("ride" in t or "ride_follow" in t for t in p["tracks"])


# ------------------------------------------------------------ added after the mutation run

def test_the_distance_is_read_only_where_the_stem_plays():
    # the stem is silent for 60% of the song (the band plays throughout); the
    # distance it keeps is read where it plays, not over its silence
    k = 400
    times = 0.5 + 0.25 * np.arange(k)
    own = np.where(np.arange(k) < 240, -100.0, -12.0)
    rest = np.full(k, -22.0)
    pts = planner._ride_curve(None, 2.5, rest, 1.0, levels=(times, own))
    g = np.interp(times, *np.asarray(pts).T)
    assert np.all(np.abs(g[260:390]) < 0.3)


def test_the_rest_is_read_by_time_not_by_frame_number():
    # the other stem's frames are 0.5 s apart: frame i of it is at time
    # 0.5 + 0.5 i, which is frame 2 i of the own stem
    own_t = 0.5 + 0.25 * np.arange(200)
    other_t = 0.5 + 0.5 * np.arange(100)
    other_db = np.linspace(-40.0, -10.0, 100)
    got = planner._rest_level([(own_t, np.zeros(200)), (other_t, other_db)], 0, [0.0, 0.0])
    with np.errstate(divide="ignore"):  # past the other stem's end: -inf, and NaN in got
        want = 10 * np.log10(np.interp(own_t, other_t, 10 ** (other_db / 10), left=0.0, right=0.0))
    ok = ~np.isnan(got)
    assert ok.sum() > 150
    assert np.allclose(got[ok], want[ok], atol=1e-9)


def test_a_protected_group_is_weighed_at_its_protected_gain(monkeypatch):
    # two guitar mics that build by 12 dB.  The planner is made to stage them
    # 20 dB down (so, weighed there, they are no band and the steady keys are
    # not ridden); the protection puts them back at 0 dB, where they are the
    # band, so the keys must follow them.
    n = int(SECONDS * SR)
    build = _section_env(n, 12.0)
    stems = _stems([("Guitar Mic1.wav", .05 * build * _tone(196, n, 1)),
                    ("Guitar Mic2.wav", .04 * build * _tone(196, n, 1)),
                    ("Keys.wav", .05 * _tone(330, n, 2)),
                    ("Bass.wav", .1 * _tone(55, n, 3))])
    real = planner._gain_stage

    def staged(stems_, *a, **k):
        g = real(stems_, *a, **k)
        return {name: (v - 20.0 if name.startswith("Guitar Mic") else v) for name, v in g.items()}

    monkeypatch.setattr(planner, "_gain_stage", staged)
    ids = [analyze.track_id(s.name) for s in stems[:2]]
    grouped_plan = _plan(stems, capture_groups=[{"id": "gtr", "members": ids}])
    assert all(t["gain_db"] == 0.0 for t in grouped_plan["tracks"] if t["id"] in ids)
    grouped, free = _track(grouped_plan, "keys"), _track(_plan(stems), "keys")
    assert abs(_ride_at(free, 40.0) - _ride_at(free, 8.0)) < 0.75
    assert _ride_at(grouped, 40.0) - _ride_at(grouped, 8.0) > 3.0


def test_the_protection_strips_the_receipt_with_the_ride(monkeypatch):
    # make the pass (wrongly) follow every ridden track, the group members
    # too: the protection, which runs after it, must take ride and receipt away
    real = planner._follow_band

    def everyone(stems_, tracks, **kw):
        real(stems_, tracks, **kw)
        for t in tracks:
            if t.get("ride"):
                t["ride_follow"] = {"law": "band_follow_v1", "follow": 1.0}

    monkeypatch.setattr(planner, "_follow_band", everyone)
    n = int(SECONDS * SR)
    band = _section_env(n, CHORUS_DB)
    stems = _stems([("Guitar Mic1.wav", .08 * band * _tone(196, n, 1)),
                    ("Guitar Mic2.wav", .06 * band * _tone(196, n, 1)),
                    ("Keys.wav", .05 * _tone(330, n, 2)),
                    ("Bass.wav", .1 * band * _tone(55, n, 3))])
    ids = [analyze.track_id(s.name) for s in stems[:2]]
    p = _plan(stems, capture_groups=[{"id": "gtr", "members": ids}])
    members = [t for t in p["tracks"] if t["id"] in ids]
    assert members and all("ride" not in t and "ride_follow" not in t for t in members)
    assert "ride_follow" in _track(p, "keys")


def test_an_overflowing_window_reads_loud_not_silent():
    # float64 audio with an absurd sample: the window's power overflows to
    # +inf; that reads as the loudest possible (+300 dB), never as silence
    x = np.full((5 * SR, 2), 0.1)
    x[SR // 2] = 1e200
    stem = analyze.Stem(name="Keys.wav", path="Keys.wav", sr=SR, audio=x)
    with np.errstate(over="ignore", invalid="ignore"):
        _, db = planner._stem_levels(stem)
    assert np.all(np.isfinite(db)) and db.max() == 300.0
