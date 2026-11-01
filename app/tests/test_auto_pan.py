"""Where the planner puts a track when more than one of them shares a role.

A role carries one pan number, so two tracks with the same role used to land
on top of each other -- and the fan that was supposed to separate them fanned
around the role's own offset, which pushed a group of identical parts even
further to one side instead of spreading them.  The rules these tests pin down:

* a file name that commits to a side ("guitar_01_L") is believed and held on
  that side, and is never part of a fan;
* the rest of the group is spread symmetrically around the centre, so the
  group's automatic members balance each other;
* a name is only read as a side when it says so in whole tokens -- "loop",
  "layer", "lead", "room" and "rights" are not sides;
* a name that mentions both sides is not guessed at;
* centre roles (kick, snare, bass, sub, lead vocal, full drums) stay centred;
* the placement does not depend on the order the tracks arrive in;
* a pan the user set by hand is still the last word;
* a source the user declared to be balanced stereo at full balance
  preservation gets no automatic placement of its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from automix import analyze, plan as planner, render

SR = 48000


# ------------------------------------------------------------------ helpers

def _tracks(*specs):
    """Planner tracks as build_plan makes them: (file name, role)."""
    return [
        {
            "id": analyze.track_id(name),
            "role": role,
            "pan": planner.PAN.get(role, 0.0),
            "evidence": [],
        }
        for name, role in specs
    ]


def _place(*specs):
    """Run the placement helper; return {track id: pan}."""
    tracks = _tracks(*specs)
    planner._spread_duplicate_pans(tracks)
    return {t["id"]: float(t["pan"]) for t in tracks}


def _evidence(*specs):
    """Run the placement helper; return {track id: evidence lines}."""
    tracks = _tracks(*specs)
    planner._spread_duplicate_pans(tracks)
    return {t["id"]: list(t["evidence"]) for t in tracks}


def _stem(name, role, freq=220.0, dur=1.0, amp=0.25):
    """One in-memory stem, measured the way the pipeline expects."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    st = analyze.Stem(
        name=name,
        path=name,
        audio=np.stack([x, x], axis=1).astype(np.float32),
        sr=SR,
        role=role,
        role_source="keyword",
    )
    analyze.measure(st)
    return st


def _session(*specs):
    """Stems for a plan: (file name, role), measured and aligned."""
    stems = [_stem(name, role) for name, role in specs]
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


def _plan(*specs, **kwargs):
    preset = planner.load_preset("pop_2020s")
    return planner.build_plan(_session(*specs), preset, **kwargs)


def _plan_pans(plan):
    return {t["id"]: float(t["pan"]) for t in plan["tracks"]}


def _lr_db(pans):
    """Left over right energy in dB, for identical mono panned by the renderer."""
    n = 8000
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / 8000)
    dual = np.column_stack([tone, tone])
    mix = sum((render._pan(dual.copy(), p) for p in pans), start=np.zeros_like(dual))
    power = np.mean(mix ** 2, axis=0)
    return float(10 * np.log10(power[0] / power[1]))


# --------------------------------------------------- one member, one role

def test_a_single_member_keeps_its_role_base():
    """One track of a role has nothing to spread away from."""
    assert _place(("synth_a.wav", "synth")) == {"synth_a": planner.PAN["synth"]}
    assert _place(("keys_a.wav", "keys")) == {"keys_a": planner.PAN["keys"]}
    assert _place(("guitar_a.wav", "guitar_electric")) == {
        "guitar_a": planner.PAN["guitar_electric"]
    }
    assert _place(("pad_a.wav", "pad")) == {"pad_a": planner.PAN["pad"]}


def test_a_single_member_is_not_moved_by_its_name():
    """A lone track keeps the role's number even if it says left or right."""
    assert _place(("guitar_01_L.wav", "guitar_electric")) == {
        "guitar_01_L": planner.PAN["guitar_electric"]
    }


# ---------------------------------------------- automatic members spread

def test_two_three_and_four_automatic_members_spread_around_the_centre():
    """The spread is symmetric about 0 and steps by 0.2."""
    for role in ("synth", "keys", "guitar_electric", "pad"):
        for count in (2, 3, 4):
            names = [(f"part_{i + 1}.wav", role) for i in range(count)]
            wanted = [round((i - (count - 1) / 2.0) * 0.2, 2) for i in range(count)]
            got = _place(*names)
            assert list(got.values()) == wanted, (role, count, got)


def test_the_spread_is_around_the_centre_not_around_the_role_base():
    """A role that sits off centre does not drag its duplicates off centre too."""
    assert planner.PAN["guitar_electric"] < 0.0
    assert planner.PAN["fx"] > 0.0
    assert list(_place(("gtr_a.wav", "guitar_electric"), ("gtr_b.wav", "guitar_electric")).values()) == [
        -0.1,
        0.1,
    ]
    assert list(_place(("fx_a.wav", "fx"), ("fx_b.wav", "fx")).values()) == [-0.1, 0.1]


def test_a_large_group_stays_inside_the_bounds():
    """Twelve members: finite, inside -1..1, still symmetric."""
    names = [(f"part_{i + 1:02d}.wav", "synth") for i in range(12)]
    pans = list(_place(*names).values())
    assert all(np.isfinite(p) for p in pans)
    assert all(-1.0 <= p <= 1.0 for p in pans)
    assert sorted(pans) == sorted(-p for p in pans)


def test_one_remaining_automatic_member_is_centred():
    """Named sides took the flanks; the leftover sits in the middle."""
    pans = _place(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02_R.wav", "guitar_electric"),
        ("guitar_03.wav", "guitar_electric"),
    )
    assert pans == {"guitar_01_L": -0.4, "guitar_02_R": 0.4, "guitar_03": 0.0}


# ---------------------------------------------------------- reading names

def test_a_side_is_only_read_as_a_whole_token():
    """'loop', 'layer', 'lead', 'room' and 'rights' are not sides."""
    for name in (
        "synth_loop_a.wav",
        "synth_layer_b.wav",
        "guitar_lead.wav",
        "vocal_room.wav",
        "gtr_rights.wav",
        "keys_reverb.wav",
    ):
        side, ambiguous = planner._side_hint(analyze.track_id(name))
        assert (side, ambiguous) == (0, False), name


def test_a_side_is_read_whatever_the_case_or_delimiter():
    for name, wanted in (
        ("guitar_01_L.wav", -1),
        ("GUITAR_02_l.wav", -1),
        ("gtr-03-left.wav", -1),
        ("gtr.04.right.wav", 1),
        ("gtr 05 left.wav", -1),
        ("gtr 06 R.wav", 1),
        ("GTR_07_Left.wav", -1),
    ):
        side, ambiguous = planner._side_hint(analyze.track_id(name))
        assert side == wanted, name
        assert not ambiguous, name


def test_a_name_that_mentions_both_sides_is_not_guessed():
    """Both sides in one name is an answer to no question; place it automatically."""
    side, ambiguous = planner._side_hint(analyze.track_id("guitar_L_R.wav"))
    assert side == 0
    assert ambiguous

    pans = _place(("guitar_02.wav", "guitar_electric"), ("guitar_L_R.wav", "guitar_electric"))
    assert pans == {"guitar_02": -0.1, "guitar_L_R": 0.1}
    evidence = _evidence(("guitar_02.wav", "guitar_electric"), ("guitar_L_R.wav", "guitar_electric"))
    assert any("both sides" in line for line in evidence["guitar_L_R"])


def test_a_named_side_is_held_and_is_not_fanned():
    """Two tracks that both say left stay left; they are not spread."""
    pans = _place(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02_L.wav", "guitar_electric"),
        ("guitar_03_L.wav", "guitar_electric"),
    )
    assert set(pans.values()) == {-0.4}


def test_an_all_right_group_is_held_too():
    pans = _place(
        ("guitar_01_R.wav", "guitar_electric"),
        ("guitar_02_R.wav", "guitar_electric"),
    )
    assert set(pans.values()) == {0.4}


def test_named_members_are_not_part_of_the_automatic_spread():
    """A named side does not take a slot from the automatic members."""
    pans = _place(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02.wav", "guitar_electric"),
        ("guitar_03.wav", "guitar_electric"),
    )
    assert pans == {"guitar_01_L": -0.4, "guitar_02": -0.1, "guitar_03": 0.1}


def test_centre_roles_are_never_spread():
    """Kick, snare, bass, sub, lead vocal and full drums stay where they are."""
    for role in ("kick", "snare", "bass", "sub_808", "lead_vocal", "drums_full"):
        names = [(f"{role}_{i + 1}.wav", role) for i in range(3)]
        pans = _place(*names)
        assert set(pans.values()) == {0.0}, role
        assert all(not lines for lines in _evidence(*names).values()), role


# ------------------------------------------------------- order and members

def test_the_placement_does_not_depend_on_the_order_of_the_list():
    """The same tracks get the same places however they are handed over."""
    specs = [
        ("synth_a.wav", "synth"),
        ("synth_b.wav", "synth"),
        ("synth_c.wav", "synth"),
        ("guitar_01_L.wav", "guitar_electric"),
    ]
    straight = _place(*specs)
    for rotation in range(len(specs)):
        shuffled = specs[rotation:] + specs[:rotation]
        assert _place(*shuffled) == straight, shuffled


def test_the_helper_leaves_the_callers_list_alone():
    """The helper places tracks; it does not reorder or replace them.

    Ordering the caller's list would be a side effect on the plan's own track
    order, so only the automatic members are ordered, and only locally.
    """
    tracks = _tracks(("synth_b.wav", "synth"), ("synth_a.wav", "synth"))
    identities = [id(t) for t in tracks]
    planner._spread_duplicate_pans(tracks)
    assert [t["id"] for t in tracks] == ["synth_b", "synth_a"]
    assert [id(t) for t in tracks] == identities
    # the placement still follows the track id, whichever order they came in
    assert {t["id"]: float(t["pan"]) for t in tracks} == {
        "synth_a": -0.1,
        "synth_b": 0.1,
    }
    # the names that only look like a side take their slots by id as well:
    # layer_b sorts before loop_a, so it takes the left one
    false_left = _tracks(("synth_loop_a.wav", "synth"), ("synth_layer_b.wav", "synth"))
    planner._spread_duplicate_pans(false_left)
    assert {t["id"]: float(t["pan"]) for t in false_left} == {
        "synth_layer_b": -0.1,
        "synth_loop_a": 0.1,
    }


def test_a_member_that_is_filtered_out_does_not_hold_a_slot():
    """Three tracks spread three ways; two of them spread two ways."""
    three = _place(("synth_a.wav", "synth"), ("synth_b.wav", "synth"), ("synth_c.wav", "synth"))
    assert sorted(three.values()) == [-0.2, 0.0, 0.2]
    two = _place(("synth_a.wav", "synth"), ("synth_b.wav", "synth"))
    assert sorted(two.values()) == [-0.1, 0.1]


def test_every_placement_is_finite_and_inside_the_bounds():
    names = [(f"part_{i + 1:02d}.wav", "synth") for i in range(20)]
    for pan in _place(*names).values():
        assert np.isfinite(pan)
        assert -1.0 <= pan <= 1.0


# ----------------------------------------------------------- in a real plan

def test_a_plan_spreads_a_duplicate_role_group_around_the_centre():
    plan = _plan(
        ("synth_a.wav", "synth"),
        ("synth_b.wav", "synth"),
        ("synth_c.wav", "synth"),
    )
    assert _plan_pans(plan) == {"synth_a": -0.2, "synth_b": 0.0, "synth_c": 0.2}


def test_a_plan_holds_a_named_side_and_centres_the_rest():
    plan = _plan(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02.wav", "guitar_electric"),
    )
    assert _plan_pans(plan) == {"guitar_01_L": -0.4, "guitar_02": 0.0}


def test_a_plan_keeps_its_track_order_and_its_stem_pairing():
    """Local automatic placement keeps the caller's track order and each
    track paired with its original stem."""
    stems = _session(("synth_b.wav", "synth"), ("synth_a.wav", "synth"))
    preset = planner.load_preset("pop_2020s")
    plan = planner.build_plan(stems, preset)
    assert [t["id"] for t in plan["tracks"]] == ["synth_b", "synth_a"]
    assert [s["id"] for s in plan["source"]["stems"]] == ["synth_b", "synth_a"]
    lufs = {s.name: s.metrics["lufs"] for s in stems}
    for track in plan["tracks"]:
        assert track["level_match"]["source_lufs"] == pytest.approx(
            lufs[track["source_file"]], abs=1e-9
        )


def test_a_dropped_silent_stem_does_not_take_a_slot():
    """An excluded stem is not a member of the group any more."""
    preset = planner.load_preset("pop_2020s")
    stems = [
        _stem("synth_a.wav", "synth"),
        _stem("synth_b.wav", "synth"),
        _stem("synth_c.wav", "synth"),
    ]
    stems[2].audio[:] = 0.0
    analyze.measure(stems[2])
    assert stems[2].metrics.get("silent")
    analyze.align(stems)
    plan = planner.build_plan(stems, preset)
    assert _plan_pans(plan) == {"synth_a": -0.1, "synth_b": 0.1}
    assert [e["id"] for e in plan["source"]["excluded_stems"]] == ["synth_c"]


# --------------------------------------------------- the user's pan is last

def test_a_manual_pan_of_zero_survives_the_automatic_spread():
    plan = _plan(
        ("tom_1.wav", "toms"),
        ("tom_2.wav", "toms"),
        pans={"tom_1": 0.0, "tom_2": 0.0},
    )
    assert _plan_pans(plan) == {"tom_1": 0.0, "tom_2": 0.0}


def test_a_manual_pan_at_the_extremes_survives_the_automatic_spread():
    plan = _plan(
        ("tom_1.wav", "toms"),
        ("tom_2.wav", "toms"),
        pans={"tom_1": -1.0, "tom_2": 1.0},
    )
    assert _plan_pans(plan) == {"tom_1": -1.0, "tom_2": 1.0}


def test_a_manual_pan_beats_a_named_side():
    plan = _plan(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02.wav", "guitar_electric"),
        pans={"guitar_01_L": 0.7},
    )
    assert _plan_pans(plan) == {"guitar_01_L": 0.7, "guitar_02": 0.0}


# ------------------------------------------------------- what it sounds like

def test_two_automatically_placed_duplicates_stay_symmetric_in_the_audio():
    """Identical mono, panned by the real renderer: the pair must balance."""
    pans = list(_place(("synth_a.wav", "synth"), ("synth_b.wav", "synth")).values())
    assert abs(_lr_db(pans)) < 0.01


def test_three_automatically_placed_duplicates_stay_symmetric_in_the_audio():
    pans = list(
        _place(
            ("synth_a.wav", "synth"),
            ("synth_b.wav", "synth"),
            ("synth_c.wav", "synth"),
        ).values()
    )
    assert abs(_lr_db(pans)) < 0.01


def test_a_named_left_group_still_leans_left():
    """Naming a side is a request, and the audio obeys it."""
    pans = list(
        _place(
            ("guitar_01_L.wav", "guitar_electric"),
            ("guitar_02_L.wav", "guitar_electric"),
        ).values()
    )
    assert _lr_db(pans) > 3.0


# ------------------------------------------- a declared balanced stereo source

def _stereo_stem(name, role, freq=220.0, dur=1.0, amp=0.25):
    """One stem whose two channels really differ -- a printed stereo image.

    ``_stem`` writes the same signal into both channels, so every file it
    makes is dual mono: one source, two identical copies.  A mono file reads
    that way too -- the loader duplicates a mono channel at unity -- so only a
    stem like this one can show the planner that the source is a stereo
    print, and only a stem like this one may lose an automatic placement.
    """
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    y = amp * (0.85 * np.sin(2 * np.pi * freq * t)
               + 0.15 * np.sin(2 * np.pi * 1.5 * freq * t))
    st = analyze.Stem(
        name=name,
        path=name,
        audio=np.stack([x, y], axis=1).astype(np.float32),
        sr=SR,
        role=role,
        role_source="keyword",
    )
    analyze.measure(st)
    return st


def _stereo_session(*specs):
    """Stems for a plan whose channels really differ: (file name, role)."""
    stems = [_stereo_stem(name, role) for name, role in specs]
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    return stems


def _stereo_plan(*specs, **kwargs):
    preset = planner.load_preset("pop_2020s")
    return planner.build_plan(_stereo_session(*specs), preset, **kwargs)


def _plan_evidence(plan):
    """The evidence lines of every track in a plan: {track id: [lines]}."""
    return {t["id"]: list(t["evidence"]) for t in plan["tracks"]}


@pytest.mark.parametrize("role,name", [
    ("synth", "arp"),
    ("keys", "chords"),
    ("guitar_electric", "gtr"),
])
def test_balanced_stereo_declared_stems_get_no_automatic_role_offset(role, name):
    """A declared balanced stereo print keeps its own L/R; no offset is added.

    The role's own number (the arp sits left, the chords sit left of centre)
    is an L/R offset on top of a source the user already declared balanced,
    and nobody asked for it.
    """
    plan = _stereo_plan((f"{name}.wav", role), source_balance_profile="balanced_stems")
    assert plan["style"]["source_balance"]["requested"] == "balanced_stems"
    assert plan["style"]["balance_preserve"] == 1
    assert _plan_pans(plan) == {name: 0.0}
    evidence = _plan_evidence(plan)[name]
    assert any("automatic placement suppressed" in line for line in evidence)
    assert any("balanced stereo" in line for line in evidence)


def test_balanced_stereo_declared_stems_get_no_automatic_spread():
    """Two parts of one role are not spread apart on a declared stereo print."""
    plan = _stereo_plan(
        ("synth_a.wav", "synth"),
        ("synth_b.wav", "synth"),
        source_balance_profile="balanced_stems",
    )
    assert _plan_pans(plan) == {"synth_a": 0.0, "synth_b": 0.0}
    # the spread really would have applied: the same stems under the default
    # declaration are placed apart
    plain = _stereo_plan(("synth_a.wav", "synth"), ("synth_b.wav", "synth"))
    assert set(_plan_pans(plain).values()) != {0.0}


def test_balanced_stereo_declared_stems_get_no_automatic_named_side():
    """A side read out of a file name is an automatic hint, not a request."""
    plan = _stereo_plan(
        ("guitar_01_L.wav", "guitar_electric"),
        ("guitar_02.wav", "guitar_electric"),
        source_balance_profile="balanced_stems",
    )
    assert _plan_pans(plan) == {"guitar_01_L": 0.0, "guitar_02": 0.0}
    evidence = _plan_evidence(plan)["guitar_01_L"]
    assert any("automatic placement suppressed" in line for line in evidence)


def test_balanced_stereo_dual_mono_material_keeps_its_placement():
    """Identical L/R is one source, not a stereo print: nothing is suppressed.

    A mono file reads as two identical channels, so this is also the mono
    case: a single source carries no stereo image that could be preserved.
    """
    plan = _plan(("arp.wav", "synth"), source_balance_profile="balanced_stems")
    assert _plan_pans(plan) == {"arp": planner.PAN["synth"]}
    assert not any("balanced stereo" in line
                   for line in _plan_evidence(plan)["arp"])


@pytest.mark.parametrize("kwargs", [
    pytest.param({"source_balance_profile": "auto"}, id="auto"),
    pytest.param({"source_balance_profile": "balanced_stems",
                  "balance_preserve": 0.5}, id="balanced_at_half"),
    pytest.param({"source_balance_profile": "raw_tracks",
                  "balance_preserve": 1.0}, id="raw_at_full_preserve"),
])
def test_balanced_stereo_other_declared_intents_keep_the_placement(kwargs):
    """Only the balanced-stems declaration at full preservation suppresses.

    The automatic profile, a balanced declaration at half preservation and a
    raw declaration at full preservation are three other statements about the
    source, and none of them is "this is already balanced stereo".
    """
    plan = _stereo_plan(("arp.wav", "synth"), **kwargs)
    assert _plan_pans(plan) == {"arp": planner.PAN["synth"]}


def test_balanced_stereo_manual_pan_still_wins():
    """The user's own pan is applied after the automatic one and still wins."""
    plan = _stereo_plan(
        ("arp.wav", "synth"),
        source_balance_profile="balanced_stems",
        pans={"arp": 0.4},
    )
    assert _plan_pans(plan) == {"arp": 0.4}
    evidence = _plan_evidence(plan)["arp"]
    assert any("automatic placement suppressed" in line for line in evidence)
    assert any("manual pan" in line for line in evidence)


def test_balanced_stereo_capture_group_pan_still_wins():
    """A declared capture group keeps its own common pan and its own override.

    A group holds one physical source, so its members are mono or dual mono
    by the group's own rule -- they keep the old placement, and the group's
    pan and the user's per-track pan are applied exactly as before.
    """
    plan = _plan(
        ("mic_a.wav", "guitar_electric"),
        ("mic_b.wav", "guitar_electric"),
        source_balance_profile="balanced_stems",
        capture_groups=[{"id": "room", "members": ["mic_a", "mic_b"], "pan": -0.3}],
        pans={"mic_a": 0.5},
    )
    assert _plan_pans(plan) == {"mic_a": 0.5, "mic_b": -0.3}
    assert any("keeps its own pan" in w for w in plan["warnings"])


def test_balanced_stereo_changes_only_the_pan_and_the_evidence():
    """Nothing else about the plan moves: same order, same pairing, same audio.

    The two plans declare the same balance in two different ways, so the
    gains, the EQ, the width and the loudness matching are the same numbers
    in both; only the automatic placement and the lines that explain it are
    different.
    """
    stems = _stereo_session(("arp.wav", "synth"), ("chords.wav", "keys"))
    original = [np.array(s.audio, copy=True) for s in stems]
    preset = planner.load_preset("pop_2020s")
    plain = planner.build_plan(stems, preset, source_balance_profile="auto",
                               balance_preserve=1.0)
    balanced = planner.build_plan(stems, preset,
                                  source_balance_profile="balanced_stems")

    # the planner never touches the audio it was given
    for stem, before in zip(stems, original):
        assert np.array_equal(stem.audio, before)

    # the track order and the source pairing are the same
    assert [t["id"] for t in balanced["tracks"]] == ["arp", "chords"]
    assert [t["id"] for t in balanced["tracks"]] == [t["id"] for t in plain["tracks"]]
    assert [t["source_file"] for t in balanced["tracks"]] == ["arp.wav", "chords.wav"]
    for one, other in zip(plain["tracks"], balanced["tracks"]):
        assert one.keys() == other.keys()
        for key in one:
            if key in ("pan", "evidence"):
                continue
            assert one[key] == other[key], key

    assert _plan_pans(plain) == {"arp": planner.PAN["synth"],
                                 "chords": planner.PAN["keys"]}
    assert _plan_pans(balanced) == {"arp": 0.0, "chords": 0.0}
