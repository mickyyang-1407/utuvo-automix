"""Overhead and room microphones are roles of their own (R109).

Reported 2026-11-02: "鼓組要有OH和Room，加進去".  Before, an overhead pair was
filed as cymbals (HPF 400 Hz, panned 0.30 to one side) and a room mic fell
through to the whole drum kit.  Names only; no audio is read.
"""

from __future__ import annotations

import pytest

from automix import analyze, plan as planner


@pytest.mark.parametrize("name", [
    "overhead.wav", "Overheads.wav", "OH L.wav", "19. OH Right.wav", "Drum OHs.wav",
    "drums_oh.wav", "Over Head.wav", "頂部麥.wav", "鼓 頂麥 L.wav",
])
def test_overhead_names(name):
    assert analyze._keyword_role(name) == "overheads"


@pytest.mark.parametrize("name", [
    "room.wav", "Drum Room.wav", "Rooms.wav", "Room L.wav", "Kit RM.wav",
    "21. Room Mono.wav", "房間麥.wav", "鼓 環境麥 R.wav",
])
def test_room_names(name):
    assert analyze._keyword_role(name) == "room"


@pytest.mark.parametrize("name, want", [
    # words that only contain the letters are not microphones
    ("john vocal.wav", "lead_vocal"),
    ("mushroom pad.wav", "pad"),
    ("harmony.wav", "backing_vocal"),
    ("warm keys.wav", "keys"),
    ("ohm synth.wav", "synth"),
    # an explicit instrument still wins
    ("Kick Room.wav", "kick"),
    ("Snare Room.wav", "snare"),
    ("vocal room.wav", "lead_vocal"),
    ("ride.wav", "cymbals"),
    ("crash.wav", "cymbals"),
    # a bare room or top word in Chinese is not a microphone
    ("房間鋼琴.wav", "piano"),
    ("piano room.wav", "piano"),
    ("Strings Room L.wav", "strings"),
    ("Guitar OH.wav", "guitar_electric"),
    ("頂部合成器.wav", "synth"),
])
def test_other_names_keep_their_roles(name, want):
    assert analyze._keyword_role(name) == want


def test_both_are_drum_roles_with_their_own_treatment():
    for role in ("overheads", "room"):
        assert role in analyze.ROLES and role in analyze.DRUM_ROLES
        assert planner.ROLE_GROUP[role] == "drum"
        # a stereo pair keeps its own image
        assert planner.PAN[role] == 0.0
    assert planner.HPF_HZ["overheads"] < planner.HPF_HZ["cymbals"]
    assert planner.REVERB_WEIGHT["room"] == 0.0


def test_every_style_sets_a_level_for_both():
    from pathlib import Path
    styles = sorted(p.stem for p in (Path(__file__).resolve().parents[2] / "presets").glob("*.json"))
    assert len(styles) == 14
    for style in styles:
        gains = planner.load_preset(style)["role_gains_db"]
        # overheads take over what cymbals used to carry; the room sits under them
        assert gains["overheads"] == gains["cymbals"], style
        offset = {"jazz_natural": -1.0, "acoustic": -1.0, "ambient_space": -1.0,
                  "cinematic": -1.0, "heavy_rock_metal": -2.0, "rock": -2.0}.get(style, -3.0)
        assert gains["room"] == pytest.approx(gains["overheads"] + offset), style
