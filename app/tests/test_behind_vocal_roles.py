"""A vocal that says it stands behind the lead is a backing vocal (R131).

"Back Vox", "Backup Vocal", "BG Vox", "vox_back_l" carry a vocal word, so the
keyword table read them as the lead vocal: three "vox_back_c/l/r" tracks of a
real session were mixed as three more leads.  Names only; no audio is read.
"""

from __future__ import annotations

import pytest

from automix import analyze


@pytest.mark.parametrize("name", [
    "vox_back_c.wav", "vox_back_l.wav", "vox_back_r.wav", "Back Vox.wav", "Back Vocals L.wav",
    "Vocal Back.wav", "Backup Vox.wav", "backup vocal 2.wav", "Vox Back Up.wav", "BG Vox.wav",
    "bg vocal.wav", "Bkg Vox.wav", "Bkgd Vocals.wav", "Bck Vox.wav", "Gang Vox.wav", "Crowd Vox.wav",
    "Vox Stack.wav", "Vocal Stacks.wav", "BackVox 2.wav", "GangVocals.wav", "BgVox_L.wav",
    "Gang.wav", "Gang 2.wav", "Backups.wav", "BgdVox.wav", "StackVox.wav", "Vox Background.wav",
    "Vocal Harmonies.wav",
    # "Vx" is the word vox, not the word lead
    "Vx Back.wav", "Vx Harmonies.wav",
])
def test_a_vocal_behind_the_lead_is_a_backing_vocal(name):
    assert analyze._keyword_role(name) == "backing_vocal"


@pytest.mark.parametrize("name", [
    # a name that also says lead keeps the lead
    "Lead Vox Back.wav", "Main Vox Back.wav", "LV back.wav", "Ld Vox bkg.wav", "Lead Vocal Harmonies.wav",
    "Lead Harmonies.wav", "Lead Vx Back.wav",
    # words left out on purpose
    "Vox Chorus.wav", "Chorus Vox.wav", "2nd Vox.wav", "Vox Hi.wav", "Vox Low.wav",
    "Vox Dbl.wav", "Lead Vox Double.wav", "vox_lead_double.wav", "Vocal Group.wav", "Vox bk.wav",
    # the letters inside another word are not the word
    "Talkback Vox.wav", "Throwback Vocal.wav", "Feedback Vox.wav", "Bgm Vox.wav", "Haystack Vox.wav",
    # plain leads
    "Vox.wav", "Lead.wav", "Vocal 1.wav",
])
def test_the_lead_stays_the_lead(name):
    assert analyze._keyword_role(name) == "lead_vocal"


@pytest.mark.parametrize("name, want", [
    # the rule only touches names read as a lead vocal
    ("Back Kick.wav", "kick"), ("Back Snare.wav", "snare"), ("Bass Back.wav", "bass"),
    ("Piano Back.wav", "piano"), ("Gang Gtr.wav", "guitar_electric"), ("Backups Gtr.wav", "guitar_electric"),
    ("Back Beat.wav", None), ("Talkback.wav", None), ("Stack 1.wav", None), ("Crowd.wav", None),
    ("Group 1.wav", None), ("Backing.wav", None),
    # backing vocals that were already read right
    ("BV 1-1.wav", "backing_vocal"), ("Lead BV.wav", "backing_vocal"), ("Vox Harm.wav", "backing_vocal"),
    ("BGVs.wav", "backing_vocal"),
])
def test_everything_else_reads_as_before(name, want):
    assert analyze._keyword_role(name) == want


def test_a_name_that_was_read_only_changes_in_two_ways():
    """A name the old reading understood keeps its role, except lead -> backing vocal and
    backing vocal -> the instrument whose harmony line it is.  Checked over every order of one to three of the words below."""
    import itertools
    words = ["kick", "sn", "top", "bass", "di", "vox", "vocal", "lead", "main", "bv", "oh", "room", "gtr", "eg",
             "synth", "piano", "harmony", "harmonies", "back", "bg", "gang", "stack", "double", "dbl", "choir",
             "harm", "sax", "pad", "keys", "hh", "tom", "ride", "l", "r", "1", "2",
             "ld", "lv", "vx", "background", "bgd", "bkg", "backup", "crowd"]
    names = [" ".join(c) + ".wav" for n in (1, 2, 3) for c in itertools.permutations(words, n)]
    changed = set()
    for name in names:
        before, after = analyze._keyword_role_before_r131(name), analyze._keyword_role(name)
        if before is None or after == before:
            continue
        changed.add((before, after in analyze.VOCAL_ROLES))
        assert (before, after) == ("lead_vocal", "backing_vocal") or (
            before == "backing_vocal" and after not in analyze.VOCAL_ROLES and after is not None), (name, before, after)
    assert changed == {("lead_vocal", True), ("backing_vocal", False)}  # both ways really occur in the sweep


# ---------------------------------------------------------------- R131: a recorder's own track names

#: the session this was reported from: 49 tracks, 19 roles set by hand.  Names
#: as exported (track.playlist_take), and the role the user ended up with.
SESSION = {
    "01 EG Mono_01.wav": "guitar_electric", "06 EG Stereo.01_01.wav": "guitar_electric",
    "AG 01_Intro 0101_01.wav": "guitar_acoustic", "AG 02_01.wav": "guitar_acoustic",
    "AG_Mute 01_01.wav": "guitar_acoustic", "BV 1-1_01.wav": "backing_vocal", "BV 3-2_01.wav": "backing_vocal",
    "Bass Amp.Edit_02.wav": "bass", "Bass Clean.Edit_02.wav": "bass", "Dr_Cymbal_Rolls_01.wav": "cymbals",
    "FT.13_02.wav": "toms", "HH.13_02.wav": "hats", "KI.13_02.wav": "kick", "KO1.13_02.wav": "kick",
    "KO2.13_02.wav": "kick", "Lead01_01.wav": "lead_vocal", "Lead02_01.wav": "lead_vocal",
    "OHL.13_02.wav": "overheads", "OHR.13_02.wav": "overheads", "PN.1_01.wav": "piano",
    "Ride.13_02.wav": "cymbals", "RoomL.13_02.wav": "room", "RoomR.13_02.wav": "room",
    "SNB.13_02.wav": "snare", "SNM.13_02.wav": "snare", "SNT.13_02.wav": "snare", "Synth 01_01.wav": "synth",
    "T1.13_02.wav": "toms", "T2.13_02.wav": "toms", "unisL.04_02.wav": "backing_vocal",
    "unisR.04_02.wav": "backing_vocal",
}


@pytest.mark.parametrize("name, want", sorted(SESSION.items()))
def test_a_recorders_track_names_are_read(name, want):
    assert analyze._keyword_role(name) == want


def test_those_names_needed_the_new_reading():
    """The hand-set ones said nothing before -- otherwise this test proves nothing."""
    silent = [n for n in SESSION if analyze._keyword_role_before_r131(n) is None]
    assert len(silent) >= 17 and "SNT.13_02.wav" in silent and "AG 02_01.wav" in silent


@pytest.mark.parametrize("name, want", [
    ("KI.wav", "kick"), ("Ki 2.wav", "kick"), ("KO2.wav", "kick"), ("K.wav", "kick"),
    ("SNT.wav", "snare"), ("T1.wav", "toms"), ("T3 L.wav", "toms"), ("TT1.wav", "toms"), ("OHL.wav", "overheads"),
    ("RoomL.wav", "room"), ("RmR.wav", "room"), ("03 SNB.wav", "snare"), ("AG.wav", "guitar_acoustic"),
    ("AG2.wav", "guitar_acoustic"), ("AG Intro.wav", "guitar_acoustic"), ("PN.wav", "piano"),
    ("unisL.wav", "backing_vocal"), ("ukelele.wav", "guitar_acoustic"), ("Uke 2.wav", "guitar_acoustic"),
])
def test_recorder_codes(name, want):
    assert analyze._keyword_role(name) == want


@pytest.mark.parametrize("name", [
    # a drum-mic code with words after it is not a code
    "KO Theme.wav", "KI Song.wav", "SNT Demo.wav", "sn.wav", "SN_12345.wav", "T1 Topography.wav", "OHL Mix.wav",
    # a lone "T" is too little; the letters inside a word are not a code
    "T.wav", "Rt.wav", "Lt.wav", "T-Rex.wav", "Again.wav", "Agnes.wav", "Pneuma.wav", "Snap.wav", "Unisex.wav",
    # dots that are not a take suffix
    "v1.2 mix.wav", "Song ft. Guest.wav", "mix.wav", "Track 1.wav", "A - test2.wav",
])
def test_names_that_still_say_nothing(name):
    assert analyze._keyword_role(name) is None


@pytest.mark.parametrize("name, want", [
    # a name that was read before reads the same, dot or not
    ("Kick.L.wav", "kick"), ("Bass.DI.wav", "bass"), ("Lead01.02_03.wav", "lead_vocal"),
    ("Unison Synth.wav", "synth"), ("SD 2.wav", "snare"), ("HH.wav", "hats"), ("FT.wav", "toms"),
    # the dot is a separator only for a name that said nothing: with it, this one would read "bass gtr"
    # (a bass).  Left as it was read -- this round does not re-read names that already had a role.
    ("Bass.Gtr.wav", "guitar_electric"),
])
def test_names_read_before_are_unchanged(name, want):
    assert analyze._keyword_role_before_r131(name) == want and analyze._keyword_role(name) == want


# ---------------------------------------------------------------- R131: "harmony" beside an instrument

@pytest.mark.parametrize("name, want", [
    ("guitar_electric_solo_harmony.wav", "guitar_electric"), ("guitar_acoustic_lead_harmony.wav", "guitar_acoustic"),
    ("Synth Harmony.wav", "synth"), ("Harmony Gtr L.wav", "guitar_electric"),
])
def test_an_instruments_harmony_line_is_the_instrument(name, want):
    assert analyze._keyword_role_before_r131(name) == "backing_vocal"  # the old reading
    assert analyze._keyword_role(name) == want


@pytest.mark.parametrize("name", [
    "Harmony.wav", "Harmony 2.wav", "Hi Harmony L.wav", "Vox Harmony.wav", "Lead Vox Harmony.wav",
    "Harmony Vocal Gtr Room.wav", "BV Harmony.wav", "Harmony Vox.wav",
])
def test_a_harmony_with_no_instrument_or_with_a_vocal_word_is_a_backing_vocal(name):
    assert analyze._keyword_role(name) == "backing_vocal"
