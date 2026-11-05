"""R117: a balanced bus's reverb and delay sends follow its balance move.

The balance step moves the whole vocal (drum, bass) bus to its target after
bus processing.  Before R117 only the dry bus moved: a vocal pulled down 6 dB
kept its reverb where it was and came out 6 dB wetter (three songs measured
in R116), and drums cut 12 dB kept their plate.  New plans write
``balance.sends = post_fader_v1``: the returns of that bus's tracks are summed
apart and join the fx bus at the bus's balance gain, as a post-fader send on
a console.  Plans without the field render as before (test_replay_r115.py).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _session(seconds=12):
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(3)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    sings = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    stems = []

    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    add("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t) + .02 * rng.standard_normal(t.size))
    add("Lead Vocal.wav", .1 * sings * np.sin(2 * np.pi * 440 * t))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def session():
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    return stems, p


def _without_follow(p):
    q = copy.deepcopy(p)
    for name in ("vocal", "drum", "bass"):
        bal = (q["buses"].get(name) or {}).get("balance")
        if bal:
            bal.pop("sends", None)
    return q


def test_new_plans_ask_every_balanced_bus_to_carry_its_sends(session):
    _, p = session
    planner.validate(p)
    for name in ("vocal", "drum", "bass"):
        assert p["buses"][name]["balance"]["sends"] == "post_fader_v1"


def test_the_followed_returns_join_the_fx_bus_at_the_fader():
    n = 1000
    buses = {"fx": np.ones((n, 2))}
    wet = {"vocal": np.full((n, 2), 0.5), "drum": np.full((n, 2), 0.25)}
    out = render._join_followed(buses, wet, {"vocal": -6.0}, n)
    # the vocal's return moved with its -6 dB; the drums' balance did not move
    expect = 1.0 + 0.5 * dsp.db_to_lin(-6.0) + 0.25
    assert np.allclose(buses["fx"], expect)
    assert out["vocal"]["gain_db"] == -6.0 and out["drum"]["gain_db"] == 0.0
    assert out["vocal"]["law"] == "post_fader_v1"
    assert not wet  # handed over, not kept


def test_a_session_without_an_fx_bus_gets_one():
    n = 10
    buses = {"vocal": np.zeros((n, 2))}
    render._join_followed(buses, {"vocal": np.ones((n, 2))}, {"vocal": 6.0}, n)
    assert np.allclose(buses["fx"], dsp.db_to_lin(6.0))


def test_the_receipt_uses_the_balance_moves(session):
    stems, p = session
    p = copy.deepcopy(p)
    # the planner gives this kick no plate; give it one so the drums follow too
    for t in p["tracks"]:
        if planner.ROLE_GROUP.get(t["role"]) == "drum":
            t.setdefault("sends", {})["plate"] = 0.05
    seen = {}
    real = render._join_followed

    def spy(buses, followed, fader_db, n):
        seen["fader"] = dict(fader_db)
        seen["energy"] = {k: float(np.sum(v ** 2)) for k, v in followed.items()}
        return real(buses, followed, fader_db, n)
    render._join_followed = spy
    try:
        _, rep = render.render(stems, p)
    finally:
        render._join_followed = real
    sf = rep["send_follow"]
    # the vocal and the kick send to the plate; bass sends nothing
    assert set(sf) == {"vocal", "drum"}
    assert sf["vocal"]["gain_db"] == pytest.approx(rep["vocal_balance"]["applied_db"], abs=0.006)
    # the band's receipt adds rounded per-round steps (up to 0.005 dB each,
    # six rounds); the returns follow the real move (next test)
    assert sf["drum"]["gain_db"] == pytest.approx(rep["part_balance"]["drum"]["applied_db"], abs=0.035)
    # the exact move, not the receipt's rounded one
    assert seen["fader"]["vocal"] == pytest.approx(rep["vocal_balance"]["applied_db"], abs=0.005)
    assert all(e > 0 for e in seen["energy"].values())
    # a real move happened in this session, so the test means something
    assert abs(sf["drum"]["gain_db"]) > 1.0


def test_a_bus_that_did_not_move_renders_like_before(session):
    """With the balance held still (both limits 0) following the fader changes
    nothing: the mix matches the plan without the field to rounding."""
    stems, p = session
    held = copy.deepcopy(p)
    for name in ("vocal", "drum", "bass"):
        held["buses"][name]["balance"]["max_cut_db"] = 0.0
        held["buses"][name]["balance"]["max_boost_db"] = 0.0
    a, rep_a = render.render(stems, held)
    b, rep_b = render.render(stems, _without_follow(held))
    assert "send_follow" in rep_a and "send_follow" not in rep_b
    assert np.max(np.abs(a - b)) < 1e-6
    assert rep_a["wet_ratio_db"] == pytest.approx(rep_b["wet_ratio_db"], abs=0.01)


def test_a_vocal_pulled_down_keeps_its_wet_dry():
    """The finding: a vocal the balance pulls down 6 dB used to come out 6 dB
    wetter.  Measured where the balance step sees it: the vocal's own return
    against the vocal dry, with and without following."""
    stems = _session()
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    # ask for a vocal far below where it sits: the balance pulls it down
    p["buses"]["vocal"]["balance"]["target_db"] = -10.0
    got = {}
    real = render._join_followed

    def spy(buses, followed, fader_db, n):
        got["vocal_wet_db"] = render._rms_db(followed["vocal"]) + fader_db["vocal"]
        got["vocal_dry_db"] = render._rms_db(buses["vocal"])
        got["move"] = fader_db["vocal"]
        return real(buses, followed, fader_db, n)
    render._join_followed = spy
    try:
        render.render(stems, p)
        follow = dict(got)
        got.clear()
        # the same plan with the vocal held where it was: no move
        q = copy.deepcopy(p)
        q["buses"]["vocal"]["balance"]["max_cut_db"] = 0.0
        render.render(stems, q)
        still = dict(got)
    finally:
        render._join_followed = real
    assert follow["move"] < -3.0
    ratio_follow = follow["vocal_wet_db"] - follow["vocal_dry_db"]
    ratio_still = still["vocal_wet_db"] - still["vocal_dry_db"]
    assert ratio_follow == pytest.approx(ratio_still, abs=0.05)


def test_drums_and_bass_alone_keep_their_returns():
    """When the band is only drums and bass the balance is skipped (R116):
    nothing moved, so the returns stay at 0 dB."""
    t = np.arange(SR * 10) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    stems = []
    for name, a in (("Kick.wav", .6 * kick), ("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    for rec in (rep.get("send_follow") or {}).values():
        assert rec["gain_db"] == 0.0


def test_an_unknown_send_law_is_refused(session):
    stems, p = session
    q = copy.deepcopy(p)
    q["buses"]["vocal"]["balance"]["sends"] = "pre_fader_v9"
    with pytest.raises(render.RenderError):
        render.render(stems, q)


def test_the_report_names_the_moves(session, tmp_path):
    stems, p = session
    _, rep = render.render(stems, p)
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    text = path.read_text()
    assert "reverb and delay sends (post-fader" in text
    assert f"vocal {rep['send_follow']['vocal']['gain_db']:+.1f} dB" in text


def test_the_vocal_delay_follows_the_fader_too(session):
    """The delay return of a followed bus joins with it, not with the shared
    delay: with the plate off, the vocal's followed return is its delay."""
    stems, p = session
    q = copy.deepcopy(p)
    vocal = next(t for t in q["tracks"] if t["role"] == "lead_vocal")
    assert q.get("delay") and vocal["sends"].get("delay", 0) > 0
    vocal["sends"]["plate"] = 0.0
    for t in q["tracks"]:
        if t is not vocal:
            t["sends"] = {}
    seen = {}
    real = render._join_followed

    def spy(buses, followed, fader_db, n):
        seen.update({k: float(np.sum(v ** 2)) for k, v in followed.items()})
        return real(buses, followed, fader_db, n)
    render._join_followed = spy
    try:
        _, rep = render.render(stems, q)
    finally:
        render._join_followed = real
    assert seen.get("vocal", 0.0) > 0.0
    assert rep["mix_moves"]["vocal_delay"]["law"] == "tempo_delay_v1"


def test_the_drums_follow_their_real_move_not_the_rounded_receipt(session):
    """The returns join at the gain the bus really moved: the band balance
    adds rounded per-round steps into its receipt, the bus moves by the
    unrounded ones (and the cap correction)."""
    stems, p = session
    seen = {}
    real_band, real_join = render._balance_band, render._join_followed

    def band(buses, specs, sr, warnings, **kw):
        before = buses["drum"].copy()
        out = real_band(buses, specs, sr, warnings, **kw)
        i = int(np.argmax(np.abs(before[:, 0])))
        seen["moved"] = 20 * np.log10(abs(buses["drum"][i, 0] / before[i, 0]))
        return out

    def join(buses, followed, fader_db, n):
        seen["fader"] = fader_db.get("drum")
        return real_join(buses, followed, fader_db, n)
    render._balance_band, render._join_followed = band, join
    try:
        render.render(stems, p)
    finally:
        render._balance_band, render._join_followed = real_band, real_join
    assert abs(seen["moved"]) > 1.0
    assert seen["fader"] == pytest.approx(seen["moved"], abs=1e-6)

