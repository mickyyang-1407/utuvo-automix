"""Every instrument name and abbreviation UTUVO Prep writes or reads.

Requested 2026-11-04: "參考UTUVO Prep，把一些樂器的簡寫都要納入進去，不然我幾乎
都要自己分類".  The list is Prep 1.0.10's music naming presets (expand /
shorten / group prefix): every name they match and every name they write, so a
session renamed by Prep, or named the way Prep expects, is sorted here
without hand work.  Where a mix needs something else than Prep's delivery
group, the mix wins and it says so: a double bass is mixed as a bass (Prep
files it under strings), because a strings high-pass would take its
fundamental away.
"""
import pytest

from automix import analyze

EXPECTED = {
    # drums
    "BD": "kick", "Kick": "kick", "Kick 1": "kick", "Kick In": "kick", "Kick Out": "kick",
    "K In": "kick", "K Out": "kick", "K Sub": "kick", "Kick Sub": "kick",
    "SD": "snare", "Snare": "snare", "Snare 1": "snare", "Snare Top": "snare", "Snare Bottom": "snare",
    "Snr": "snare", "Snr 1": "snare", "Snr Top": "snare", "Snr Bot": "snare", "Sn T": "snare",
    "Sn B": "snare", "Side Snare": "snare",
    "Tom": "toms", "Toms": "toms", "Tom 1": "toms", "Tom 2": "toms", "Tom1": "toms", "Tom2": "toms",
    "Toms 1": "toms", "Toms 2": "toms", "Floor Tom": "toms", "Floor Tom 1": "toms", "Rack Tom": "toms",
    "FT": "toms",
    "HH": "hats", "Hats": "hats", "Hi Hat": "hats", "HiHat": "hats",
    "Ride": "cymbals",
    "OH": "overheads", "Overhead": "overheads", "Overheads": "overheads",
    "Rm": "room", "Room": "room", "Rooms": "room", "Room Mono": "room", "Room Stereo": "room",
    "Crotch": "room",
    "Drums": "drums_full",
    "Perc": "percussion", "Percussion": "percussion", "AuxPerc": "percussion",
    "Aux Percussion": "percussion", "Shaker": "percussion", "Tambourine": "percussion",
    # bass
    "Bass": "bass", "Bs": "bass", "Bass DI": "bass", "Bs DI": "bass", "Bass Amp": "bass",
    "Electric Bass": "bass", "Upright Bass": "bass", "Synth Bass": "bass",
    "Double Bass": "bass", "Db": "bass", "Cb": "bass",
    # guitars
    "Guitar": "guitar_electric", "Gtr": "guitar_electric", "GTR": "guitar_electric",
    "gtr": "guitar_electric", "EGtr": "guitar_electric", "EGtrs": "guitar_electric",
    "ElecGtr": "guitar_electric", "Electric Guitar": "guitar_electric",
    "Electric Guitars": "guitar_electric", "Guitar Amp": "guitar_electric",
    "Guitar DI": "guitar_electric", "Pedal Steel": "guitar_electric", "Lap Steel": "guitar_electric",
    "AGtr": "guitar_acoustic", "AGtrs": "guitar_acoustic", "AcGt": "guitar_acoustic",
    "AcGtr": "guitar_acoustic", "Acoustic Guitar": "guitar_acoustic",
    "Acoustic Guitars": "guitar_acoustic", "Banjo": "guitar_acoustic", "Mandolin": "guitar_acoustic",
    "Ukulele": "guitar_acoustic", "Sitar": "guitar_acoustic", "Oud": "guitar_acoustic",
    # keys
    "Piano": "piano", "Pno": "piano", "PNO": "piano", "Grand Piano": "piano",
    "Electric Piano": "keys", "EPiano": "keys", "Keys": "keys", "Organ": "keys",
    "Harpsichord": "keys", "Celesta": "keys", "Accordion": "keys",
    # synths
    "Synth": "synth", "Syn": "synth", "SY": "synth", "Synth Lead": "synth", "Syn Lead": "synth",
    "Synth Arp": "synth", "Synth Arps": "synth", "Syn Arps": "synth", "Synth Pulse": "synth",
    "Arp": "synth", "Arpeggio": "synth", "Loop": "synth", "LOOP": "synth",
    "Pad": "pad", "Synth Pad": "pad", "Syn Pad": "pad",
    # vocals
    "Vocal": "lead_vocal", "Vox": "lead_vocal", "V": "lead_vocal", "LV": "lead_vocal",
    "LV-D": "lead_vocal", "Lead Vocal": "lead_vocal", "Lead Vox": "lead_vocal",
    "LeadVox": "lead_vocal", "LdVox": "lead_vocal", "Lead Vocal Double": "lead_vocal",
    "LdVocOnly": "lead_vocal", "LdVocWet": "lead_vocal", "Ld Voc Only": "lead_vocal",
    "Ld Voc Wet": "lead_vocal",
    "BGV": "backing_vocal", "BGVs": "backing_vocal", "BV": "backing_vocal",
    "Backing Vocal": "backing_vocal", "Backing Vocals": "backing_vocal",
    "Background Vocal": "backing_vocal", "Background Vocals": "backing_vocal",
    "BackingVox": "backing_vocal", "BkVox": "backing_vocal", "BgvOnly": "backing_vocal",
    "BgvWet": "backing_vocal", "Bgv Only": "backing_vocal", "Bgv Wet": "backing_vocal",
    "Harm": "backing_vocal", "Harmony": "backing_vocal", "Choir": "backing_vocal",
    "Dbl": "backing_vocal", "Double": "backing_vocal",
    # strings
    "Strings": "strings", "Strs": "strings", "String Section": "strings", "Violin": "strings",
    "Vln": "strings", "vn": "strings", "Viola": "strings", "Vla": "strings", "va": "strings",
    "Cello": "strings", "Vc": "strings", "Harp": "strings", "Hp": "strings",
    # brass (Prep files the saxes with the brass too)
    "Brass": "brass", "Brass Section": "brass", "Horns": "brass", "Trumpet": "brass", "Tpt": "brass",
    "Trombone": "brass", "Tbn": "brass", "Tuba": "brass", "Tba": "brass", "French Horn": "brass",
    "Hn": "brass", "FHn": "brass", "Cornet": "brass", "Crt": "brass", "Flugelhorn": "brass",
    "Fgh": "brass", "Euphonium": "brass", "Euph": "brass", "Sax": "brass", "Saxophone": "brass",
    "Alto Sax": "brass", "Tenor Sax": "brass", "Baritone Sax": "brass",
    # woodwinds
    "Woodwinds": "woodwinds", "Flute": "woodwinds", "Fl": "woodwinds", "Alto Flute": "woodwinds",
    "Afl": "woodwinds", "Bass Flute": "woodwinds", "Bfl": "woodwinds", "Piccolo": "woodwinds",
    "Picc": "woodwinds", "Clarinet": "woodwinds", "Cl": "woodwinds", "Bass Clarinet": "woodwinds",
    "Bcl": "woodwinds", "Contrabass Clarinet": "woodwinds", "Cbcl": "woodwinds", "Oboe": "woodwinds",
    "Ob": "woodwinds", "English Horn": "woodwinds", "Eh": "woodwinds", "Bassoon": "woodwinds",
    "Bsn": "woodwinds", "Bn": "woodwinds", "Contrabassoon": "woodwinds", "Cbn": "woodwinds",
    "Recorder": "woodwinds",
    # effects
    "FX": "fx", "SFX": "fx", "Sound Effects": "fx",
}


@pytest.mark.parametrize("name,role", sorted(EXPECTED.items()))
def test_prep_name_is_sorted(name, role):
    assert analyze._keyword_role(name + ".wav") == role


@pytest.mark.parametrize("name,role", [
    # the same names as they arrive from a real session: numbers, takes, case
    ("03 Tpt 1.wav", "brass"), ("Vln I.wav", "strings"), ("vc_div.wav", "strings"),
    ("FL 2.wav", "woodwinds"), ("Snr Top 57.wav", "snare"), ("K_In.wav", "kick"),
    ("Syn Lead 2.wav", "synth"), ("Lead Tpt.wav", "brass"), ("Perc 3.wav", "percussion"),
    # an abbreviation never beats a word that names something else
    ("LV Db.wav", "lead_vocal"), ("Lead Dbl.wav", "lead_vocal"), ("Kick FL.wav", "kick"),
    ("Drum Loop.wav", "drums_full"), ("Double Bass.wav", "bass"),
    # short abbreviations inside longer words are not abbreviations
    ("Cloud.wav", None), ("Loud.wav", None), ("Harpsichord.wav", "keys"),
    ("Vacation.wav", None), ("Flat.wav", None), ("Obi.wav", None),
])
def test_abbreviations_in_real_names(name, role):
    assert analyze._keyword_role(name) == role


@pytest.mark.parametrize("name,role", [
    # review round 1: common words that are not instrument labels here
    ("CHANNEL-TEST-12ch-loop.wav", None), ("Song ft. Guest.wav", None), ("SYS.wav", None),
    ("dolby-own-dc-power2-double-minus13p450.wav", None),
    # ... and still are when they stand alone
    ("Loop 2.wav", "synth"), ("FT.wav", "toms"), ("Double 2.wav", "backing_vocal"),
    # a vocal word never beats a named instrument, abbreviated or not
    ("Harm Tpt.wav", "brass"), ("Choir Vln.wav", "strings"),
    # the side of a stereo pair does not hide an abbreviation
    ("01 Cb L.wav", "bass"), ("Db R.wav", "bass"), ("Db Left 2.wav", "bass"),
    # Chinese: a bass clarinet is not a bass, a tambourine is not a drum kit
    ("低音單簧管.wav", "woodwinds"), ("长笛.wav", "woodwinds"), ("鈴鼓.wav", "percussion"),
    ("沙鈴 2.wav", "percussion"), ("低音.wav", "bass"), ("鼓組.wav", "drums_full"),
])
def test_review_round_one_names(name, role):
    assert analyze._keyword_role(name) == role


def test_woodwinds_can_make_room_for_the_vocal():
    from automix import unmask
    assert "woodwinds" in unmask.TARGET_ROLES
