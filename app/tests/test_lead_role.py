"""A track called "Lead" is the lead vocal (R110).

Reported 2026-11-02: "你就是把我的lead判別成synth".  "lead" sat in the synth
row of the keyword table, so a vocal named "Lead.wav" was mixed as a synth:
style gain 6 dB under the vocal, a 40 Hz HPF, no vocal bus -- and the vocal
balance step of R109 never ran because the session had no vocal at all.
Names only; no audio is read.
"""

from __future__ import annotations

import pytest

from automix import analyze


@pytest.mark.parametrize("name", [
    "Lead.wav", "lead 1.wav", "01 Lead.wav", "Lead_Comp.wav", "Lead Dbl.wav",
    "My Lead.wav", "LEAD-final.wav", "Ld.wav", "LV.wav", "LV2.wav", "Vx.wav",
    "Ld Vox.wav", "Main Vox.wav", "Lead Vocal.wav", "LeadVox.wav", "主唱.wav",
])
def test_the_lead_is_the_vocal(name):
    assert analyze._keyword_role(name) == "lead_vocal"


@pytest.mark.parametrize("name, want", [
    # a lead that names its instrument stays that instrument
    ("Lead Synth.wav", "synth"),
    ("seoul square lead.wav", "synth"),
    ("Saw Lead.wav", "synth"),
    ("Lead-Saw.wav", "synth"),
    ("SawLead.wav", "synth"),
    ("Supersaw.wav", "synth"),
    ("Lead Gtr.wav", "guitar_electric"),
    ("guitar_lead.wav", "guitar_electric"),
    ("Lead Keys.wav", "keys"),
    ("Lead Piano.wav", "piano"),
    ("Brass Lead.wav", "brass"),
    ("Lead BV.wav", "backing_vocal"),
    # the letters inside another word are not a name
    ("leader.wav", None),
    ("misled.wav", None),
    ("LVL meter.wav", None),
    # the rest of the synth row is unchanged
    ("arp.wav", "synth"),
    ("pluck 2.wav", "synth"),
    ("synth bass.wav", "bass"),  # as before: the bass row comes first
])
def test_other_leads_keep_their_instrument(name, want):
    assert analyze._keyword_role(name) == want


@pytest.mark.parametrize("name, want", [
    # short or common vocal words, whole words only (R110)
    ("Ad Libs.wav", "backing_vocal"),  # "bs" inside "libs" used to make it a bass
    ("Adlib.wav", "backing_vocal"),
    ("Harm 1.wav", "backing_vocal"),
    ("Hi Harms.wav", "backing_vocal"),
    ("Choir.wav", "backing_vocal"),
    ("Backing.wav", None),  # a backing track is the band, as before
    ("Oohs.wav", "backing_vocal"),
    ("Rap.wav", "lead_vocal"),
    ("Topline.wav", "lead_vocal"),
    # a named instrument in the same name still wins
    ("Choir Pad.wav", "pad"),
    ("Strings Choir.wav", "strings"),
    ("Backing Keys.wav", "keys"),
    ("Harm Gtr.wav", "guitar_electric"),
    ("Choir Synth.wav", "synth"),
    # the letters inside other words are not vocal words
    ("Harmonica.wav", None),
    ("Rapid Arp.wav", "synth"),
    ("Trap Drums.wav", "drums_full"),
    # a microphone position outranks a vocal word
    ("Rap Sn Top 57 L.wav", "snare"),
])
def test_short_vocal_words(name, want):
    assert analyze._keyword_role(name) == want


def test_detect_roles_takes_the_lead_from_the_name():
    import numpy as np
    sr = 48000
    t = np.arange(sr) / sr
    # a bright synthetic tone: the spectrum alone would never call it a vocal
    tone = np.stack([np.sin(2 * np.pi * 2000 * t)] * 2, axis=1) * 0.2
    stems = [analyze.Stem(name=n, path=n, audio=tone.copy(), sr=sr) for n in ("Lead.wav", "Ad Libs.wav")]
    analyze.detect_roles(stems)
    assert [(s.role, s.role_source) for s in stems] == [
        ("lead_vocal", "keyword"), ("backing_vocal", "keyword")]
