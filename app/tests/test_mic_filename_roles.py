"""Filename aliases for explicit close microphones (R37).

A session recorded into a real room arrives with names like "1. SubKick.wav"
or "5. Sn Top 57.wav".  The keyword table only ever saw instrument names, so
"SubKick" fell through to the generic "sub" substring (an 808) and the snare
positions came out as "toms" or as nothing at all.

These tests pin the *name* side of the fix.  No real recordings are used:
every stem below is a synthetic array built inside the test, and nothing is
read from or written to disk.

The contract that matters:

* ``SubKick`` is a physical kick mic and outranks the generic "sub";
* an explicit 808 / subbass label still outranks "SubKick";
* ``Sn Top`` / ``Sn Btm`` / ``Sn Bottom`` are snare positions;
* an explicit bass / vocal / 808 label still outranks the snare position;
* words that merely start the same way ("Subkickstarter", "sn_topography",
  "SnTopSecret"), serial numbers ("SN_12345") and microphone model numbers
  ("SM57", "Beta 91") are not instruments;
* Chinese names keep their precedence;
* the NWE tom positions ("T1 Top NWE", "Floor NWE") are toms, but only when
  the whole basename has that shape;
* names outside the aliases keep their old result and the spectral fallback.
"""

from __future__ import annotations

import numpy as np

from automix import analyze

SR = 48000


def test_bass_di_filename_aliases_name_the_bass_source():
    for name in (
        "BassDI_CONS.wav", "BassDI.wav", "Bass DI CONS.wav",
        "bass_di_cons.wav", "BASS-DI-CONS.WAV", "Bass.DI_CONS.wav",
        "01BassDI_CONS.wav", "BassDI3.wav",
    ):
        assert analyze._keyword_role(name) == "bass", name


def test_bass_di_lookalikes_keep_the_prior_keyword_result():
    # Root measured these R71 values; the new alias must not expand into them.
    for name, want in (
        ("BassDinosaur.wav", "snare"),
        ("BassDither.wav", "snare"),
        ("embassDI.wav", "snare"),
        ("Bass_Dinosaur.wav", "bass"),
        ("Bass D I.wav", "bass"),
    ):
        assert analyze._keyword_role(name) == want, name


def test_bass_di_alias_keeps_explicit_competing_label_priorities():
    for name, want in (
        ("808 BassDI.wav", "sub_808"),
        ("sub bass BassDI.wav", "sub_808"),
        ("lead vocal BassDI.wav", "lead_vocal"),
        ("kick BassDI.wav", "kick"),
        ("snare BassDI.wav", "snare"),
        ("主唱 BassDI.wav", "lead_vocal"),
        ("木吉他 BassDI.wav", "guitar_acoustic"),
    ):
        assert analyze._keyword_role(name) == want, name


def test_bass_di_filename_reaches_detect_roles_without_audio_changes():
    # A bright synthetic tone need not sound like a bass: the explicit name wins.
    stem = _stem("BassDI_CONS.wav", _stereo(_tone(4000.0, dur=1.0)))
    before = stem.audio.copy()
    analyze.detect_roles([stem])
    assert (stem.role, stem.role_source) == ("bass", "keyword")
    assert np.array_equal(stem.audio, before)
    assert (stem.sr, stem.name, stem.path) == (SR, "BassDI_CONS.wav", "BassDI_CONS.wav")


def test_reverse_backing_vocal_filename_aliases():
    for name in (
        "vox_backing_chorus_l.wav", "vox_backing_chorus_r.wav",
        "vox_backing_verse.wav", "VOX-BACKING-CHORUS-L.WAV",
        "Vox  Backing  Verse.wav", "Vocal_Backing_02.wav",
        "VOCALS-BACKING-03.wav",
    ):
        assert analyze._keyword_role(name) == "backing_vocal", name


def test_reverse_backing_vocal_aliases_keep_leads_and_instruments():
    for name, want in (
        ("vox_lead.wav", "lead_vocal"),
        ("lead_vox.wav", "lead_vocal"),
        ("vocal_lead.wav", "lead_vocal"),
        ("backing_guitar.wav", "guitar_electric"),
        ("guitar_backing.wav", "guitar_electric"),
        ("backing_piano.wav", "piano"),
        ("backing_track.wav", None),
        ("backing.wav", None),
    ):
        assert analyze._keyword_role(name) == want, name


def test_reverse_backing_vocal_aliases_keep_explicit_priorities():
    for name, want in (
        ("主唱_vox_backing.wav", "lead_vocal"),
        ("808_vox_backing.wav", "sub_808"),
        ("vox_backing_bass_guitar.wav", "backing_vocal"),
    ):
        assert analyze._keyword_role(name) == want, name


def test_reverse_backing_vocal_filenames_reach_detect_roles_as_keywords():
    # A bright synthetic tone must not override the filename-declared voices.
    names = ("vox_backing_chorus_l.wav", "vox_backing_chorus_r.wav",
             "vox_backing_verse.wav", "vox_lead.wav")
    stems = [_stem(name, _stereo(_tone(4000.0, dur=1.0))) for name in names]
    before = [s.audio.copy() for s in stems]
    analyze.detect_roles(stems)
    assert [s.role for s in stems] == ["backing_vocal"] * 3 + ["lead_vocal"]
    assert all(s.role_source == "keyword" for s in stems)
    for stem, audio in zip(stems, before):
        assert np.array_equal(stem.audio, audio)


def test_explicit_bass_guitar_names_outrank_generic_guitar():
    for name in (
        "Hang Low - Bass Guitar.mp3", "Bass Guitar.wav", "Bass_Guitar.wav",
        "bass-guitar.wav", "BASS GUITAR.wav", "Bass Gtr.wav", "Bass_Gtr.wav",
        "BassGuitar.wav", "BassGtr.wav", "01BassGtr.wav", "Bass Guitar 01.wav",
        "Acoustic Bass Guitar.wav", "Electric Bass Guitar.wav",
    ):
        assert analyze._keyword_role(name) == "bass", name


def test_bass_guitar_compound_keeps_explicit_source_priority():
    for name, want in (
        ("808 Bass Guitar.wav", "sub_808"),
        ("Sub Bass Guitar.wav", "sub_808"),
        ("Lead Vocal Bass Guitar.wav", "lead_vocal"),
        ("Backing Vox Bass Guitar.wav", "backing_vocal"),
        ("Bass Drum Guitar.wav", "kick"),
        ("Snare Bass Guitar.wav", "snare"),
        ("主唱 Bass Guitar.wav", "lead_vocal"),
        ("木吉他 Bass Guitar.wav", "guitar_acoustic"),
    ):
        assert analyze._keyword_role(name) == want, name


def test_bass_guitar_compound_does_not_borrow_from_longer_words():
    # Preserve the prior generic result for unrelated/ambiguous labels.
    for name, want in (
        ("Guitar.wav", "guitar_electric"),
        ("Gtr.wav", "guitar_electric"),
        ("Acoustic Guitar.wav", "guitar_acoustic"),
        ("Guitar Bass.wav", "guitar_electric"),
        ("bass guitarcase.wav", "guitar_electric"),
        ("embass guitar.wav", "guitar_electric"),
        ("bassgtronomy.wav", "guitar_electric"),
        ("Bass.wav", "bass"),
        ("SubKick.wav", "kick"),
        ("Sn Top 57.wav", "snare"),
    ):
        assert analyze._keyword_role(name) == want, name


def test_real_bass_guitar_filename_reaches_detect_roles_as_a_keyword():
    # A bright tone does not need to resemble a bass: the explicit source label
    # must decide, rather than accidentally passing via spectral refinement.
    t = np.arange(SR, dtype=np.float64) / SR
    bright = (.2 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    stem = analyze.Stem(name="Hang Low - Bass Guitar.mp3", path="not-read.mp3",
                        audio=np.column_stack([bright, bright]), sr=SR)
    analyze.detect_roles([stem])
    assert stem.role == "bass"
    assert stem.role_source == "keyword"


def _tone(freq, dur=2.0, amp=0.3, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _drum(freq, dur=2.0, sr=SR, decay=40.0):
    """A short percussive tone: enough shape to look like a drum."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.exp(-((t % 0.5) * decay))
    return (env * np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)


def _stereo(x):
    return np.stack([x, x], axis=1).astype(np.float32)


def _stem(name, audio, sr=SR):
    return analyze.Stem(name=name, path=name, audio=audio, sr=sr)


# ------------------------------------------------------------------ SubKick

def test_mic_filename_subkick_is_a_physical_kick():
    """The real session names, plus the spellings people actually type."""
    cases = {
        "1. SubKick.wav": "kick",
        "SubKick.wav": "kick",
        "sub kick.wav": "kick",
        "Sub-Kick_Out.wav": "kick",
        "SUBKICK.wav": "kick",
        "3.SubKick.wav": "kick",
        "Sub Kick 01.wav": "kick",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_subkick_keeps_the_explicit_808_priority():
    """A stem that also says 808 or subbass is an 808, not a kick mic."""
    cases = {
        "808.wav": "sub_808",
        "808_SubKick.wav": "sub_808",
        "SubKick_808.wav": "sub_808",
        "SubKick_subbass.wav": "sub_808",
        "sub bass.wav": "sub_808",
        "subbass.wav": "sub_808",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_subkick_lookalikes_are_not_a_kick():
    """Longer words must not borrow the SubKick meaning.

    These keep the old generic answer, which is the whole point: the alias
    fires on the physical name and on nothing else.
    """
    for name in (
        "Subkickstarter.wav",
        "Sub Kicker.wav",
        "resubkick.wav",
    ):
        got = analyze._keyword_role(name)
        assert got == "sub_808", (name, got)


# -------------------------------------------------------------- snare positions

def test_mic_filename_snare_positions_are_a_snare():
    """Both snare positions, in every spelling the session used."""
    cases = {
        "5. Sn Top 57.wav": "snare",
        "6. Sn Top Condnsr.wav": "snare",
        "7. Sn Btm 57.wav": "snare",
        "05_Sn.Top_57.wav": "snare",
        "sn bottom.wav": "snare",
        "sn top.wav": "snare",
        "SnTop57.wav": "snare",
        "SN_BTM.wav": "snare",
        "3SnTop.wav": "snare",
        "3SnBtm.wav": "snare",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_snare_lookalikes_are_not_a_snare():
    """A bare "sn", a serial number and a longer word are not a snare."""
    for name in (
        "sn.wav",
        "SN_12345.wav",
        "sn_topography.wav",
        "SnTopSecret.wav",
    ):
        got = analyze._keyword_role(name)
        assert got is None, (name, got)


def test_mic_filename_snare_keeps_a_competing_source_label():
    """An explicit bass / 808 / vocal label still wins over the position."""
    cases = {
        "sn top bass.wav": "bass",
        "sn top 808.wav": "sub_808",
        "sn top lead vocal.wav": "lead_vocal",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_mixed_aliases_keep_the_original_keyword_result():
    """Two different microphones in one name do not say which one it is.

    The word order must not decide the instrument: both spellings keep the
    result the ordinary keyword table produced before the aliases existed.
    """
    cases = {
        "SubKick SnTop.wav": "sub_808",
        "SnTop SubKick.wav": "sub_808",
        "sub-kick_sn-btm.wav": "sub_808",
        "sn_btm_sub-kick.wav": "sub_808",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_repeated_same_role_aliases_still_resolve():
    """Saying the same thing twice is not ambiguous."""
    cases = {
        "SnTop SnBtm.wav": "snare",
        "SnBtm SnTop.wav": "snare",
        "SubKick SubKick Out.wav": "kick",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_microphone_models_are_not_instruments():
    """A mic model on its own says nothing about the instrument."""
    for name in ("SM57.wav", "M81.wav", "57.wav", "3. Beta 91 Internal.wav"):
        got = analyze._keyword_role(name)
        assert got is None, (name, got)


# ------------------------------------------------------------- precedence, scope

def test_mic_filename_chinese_names_keep_their_priority():
    """Chinese names are more specific than the new English aliases."""
    cases = {
        "主唱_SubKick.wav": "lead_vocal",
        "底鼓 SubKick.wav": "kick",
        "军鼓 SubKick.wav": "snare",
        "小鼓 Sn Top.wav": "snare",
        "和聲 Sn Top 57.wav": "backing_vocal",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_mic_filename_ordinary_names_keep_their_old_roles():
    """Everything outside the aliases keeps the role it had before."""
    cases = {
        "kick.wav": "kick",
        "snare.wav": "snare",
        "tom.wav": "toms",
        # R109: overheads became a role of their own (requested: "鼓組要有OH和Room")
        "overhead.wav": "overheads",
        "19. OH Left.wav": "overheads",
        "2. D6 KickPort.wav": "kick",
        "bass.wav": "bass",
        "guitar ac.wav": "guitar_acoustic",
        "backing vocal.wav": "backing_vocal",
        "lead vocal.wav": "lead_vocal",
        "hihat.wav": "hats",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


# --------------------------------------------------------------- detect_roles

def test_mic_filename_detect_roles_uses_the_alias_not_the_timbre():
    """The name decides; the spectrum is not allowed to argue with it."""
    kick = _stem("1. SubKick.wav", _stereo(_tone(440.0)))       # bright, not a kick
    snare = _stem("5. Sn Top 57.wav", _stereo(_tone(55.0)))      # low, not a snare
    before = [(s.audio.copy(), s.sr, s.name) for s in (kick, snare)]

    analyze.detect_roles([kick, snare])

    assert (kick.role, kick.role_source) == ("kick", "keyword"), kick.role_source
    assert (snare.role, snare.role_source) == ("snare", "keyword"), snare.role_source
    for stem, (audio, sr, name) in zip((kick, snare), before):
        assert np.array_equal(stem.audio, audio)
        assert stem.sr == sr
        assert stem.name == name


def test_mic_filename_detect_roles_leaves_unknown_names_to_the_spectrum():
    """An unknown name still gets the old spectral fallback, not a guess."""
    unknown = _stem("3. Beta 91 Internal.wav", _stereo(_tone(440.0)))
    known = _stem("bass.wav", _stereo(_drum(60.0)))
    before = unknown.audio.copy()

    analyze.detect_roles([unknown, known])

    assert unknown.role_source == "spectral", unknown.role_source
    assert unknown.role != "kick", unknown.role
    assert np.array_equal(unknown.audio, before)
    assert unknown.sr == SR
    assert unknown.name == "3. Beta 91 Internal.wav"


def test_ukulele_filename_uses_acoustic_role_before_spectral_fallback():
    for name in ("ukulele-KaraSquare.wav", "UKULELE 02.wav"):
        stem = _stem(name, _stereo(_tone(440.0)))
        analyze.detect_roles([stem])
        assert (stem.role, stem.role_source) == ("guitar_acoustic", "keyword")


def test_ukulele_filename_keeps_explicit_priority_and_word_boundaries():
    for name, expected in (
        ("lead vocal ukulele.wav", "lead_vocal"),
        ("主唱 ukulele.wav", "lead_vocal"),
        ("ukulele bass guitar.wav", "bass"),
        # was "keys" through a bare "ep" substring; since R113 "ep" (electric
        # piano) only counts as a whole word, so a made-up word is nothing
        ("ukulelephant.wav", None),
        ("preukulele.wav", None),
        ("Luke.wav", None),
    ):
        assert analyze._keyword_role(name) == expected


# ------------------------------------------------------- NWE tom microphones

#: The NWE session names its tom microphones by position, not by instrument:
#: "T1 Top  NWE", "T2 Top NWE", "T3 Top NWE" and "Floor NWE".  None of them
#: carries an instrument word -- "Top" is not "tom", and "Floor" names the
#: floor-tom position rather than a drum kit -- so the ordinary keyword table
#: cannot see a tom in them.  They used to come out as "other" (or as whatever
#: the spectrum happened to guess), which broke the Top/Bottom pair grouping.
NWE_TOM_NAMES = (
    "T1 Top  NWE.wav",
    "T2 Top NWE.wav",
    "T3 Top NWE.wav",
    "Floor NWE.wav",
)


def test_nwe_tom_names_are_toms():
    """The four real names of the NWE tom microphones."""
    for name in NWE_TOM_NAMES:
        got = analyze._keyword_role(name)
        assert got == "toms", (name, got)


def test_nwe_tom_bottom_positions_stay_toms():
    """The Bottom names already read as toms; they must keep reading so."""
    for name in ("T1 Bottom NWE.wav", "T2 Bottom NWE.wav", "T3 Bottom NWE.wav"):
        got = analyze._keyword_role(name)
        assert got == "toms", (name, got)


def test_nwe_tom_btm_position_is_a_tom():
    """"Btm" is the same position as "Bottom", spelled the short way."""
    for name in ("T1 Btm NWE.wav", "T2 Btm NWE.wav", "T3 Btm NWE.wav"):
        got = analyze._keyword_role(name)
        assert got == "toms", (name, got)


def test_nwe_tom_names_need_the_whole_basename():
    """A longer word or a competing label is not a tom position.

    The NWE marker is what makes the name a tom name: on its own neither "T1"
    nor "Floor" says anything, and the whole normalised basename has to have
    the position shape.
    """
    cases = {
        "Noise Floor NWE.wav": "fx",        # the noise label wins
        "T1 Topography NWE.wav": None,       # "Top" is only the start of a word
        "Floor NWE Synthetic.wav": "synth",  # an explicit instrument label wins
        "T1 Synth.wav": "synth",
        "T4 Top NWE.wav": None,              # only T1/T2/T3 are tom positions
        "T1 Top.wav": None,                  # no NWE marker
        "Floor.wav": None,                   # no NWE marker
        "T1 Top NWE.wav.bak": None,          # not a wav name we know
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_nwe_tom_names_keep_the_explicit_competing_labels():
    """An explicit instrument label still outranks the NWE position."""
    cases = {
        "808 T1 Top NWE.wav": "sub_808",
        "T1 Top NWE 808.wav": "sub_808",
        "lead vocal T1 Top NWE.wav": "lead_vocal",
        "bass T1 Top NWE.wav": "bass",
        "snare T1 Top NWE.wav": "snare",
        "kick Floor NWE.wav": "kick",
        "主唱 T1 Top NWE.wav": "lead_vocal",
        "底鼓 Floor NWE.wav": "kick",
    }
    for name, want in cases.items():
        got = analyze._keyword_role(name)
        assert got == want, (name, got, want)


def test_nwe_tom_names_accept_a_leading_recording_number():
    """The session numbers its takes; the number is not part of the name."""
    for name in ("01 T1 Top NWE.wav", "1. T1 Top NWE.wav", "02_Floor_NWE.wav"):
        got = analyze._keyword_role(name)
        assert got == "toms", (name, got)


def test_nwe_tom_names_reach_detect_roles_without_audio_changes():
    """The name decides; the spectrum is not allowed to argue with it."""
    stems = [_stem(name, _stereo(_tone(440.0))) for name in NWE_TOM_NAMES]
    before = [s.audio.copy() for s in stems]

    analyze.detect_roles(stems)

    assert [s.role for s in stems] == ["toms"] * len(NWE_TOM_NAMES)
    assert all(s.role_source == "keyword" for s in stems)
    for stem, audio in zip(stems, before):
        assert np.array_equal(stem.audio, audio)
        assert stem.sr == SR
        assert stem.name in NWE_TOM_NAMES
