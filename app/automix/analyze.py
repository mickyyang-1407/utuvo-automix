"""Stem loading, measurement and role guessing.

Deliberately dependency-free: ``wave`` from the standard library reads the
WAV files, numpy does the maths, and ffmpeg is only shelled out to when the
user drops something that is not already a WAV.  That keeps the MVP runnable
on a bare Python install, which matters more right now than codec breadth.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field

import numpy as np

from . import dsp

#: every role a preset may carry a gain for, in a stable order
ROLES = (
    "kick",
    "snare",
    "hats",
    "toms",
    "cymbals",
    "overheads",
    "room",
    "drums_full",
    "percussion",
    "bass",
    "sub_808",
    "lead_vocal",
    "backing_vocal",
    "guitar_electric",
    "guitar_acoustic",
    "keys",
    "piano",
    "synth",
    "pad",
    "strings",
    "brass",
    "woodwinds",
    "fx",
    "other",
)

#: filename keywords -> role.  Longer, more specific phrases are tested first so
#: that "backing vocal" wins over "vocal" and "sub bass" wins over "bass".
_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sub 808", "808", "sub bass", "subbass", "sub"), "sub_808"),
    (("backing vocal", "backing vox", "backingvox", "background vocal", "background vox", "bkvox",
      "vox backing", "vocal backing", "vocals backing", "bv", "harmony", "vox bg", "bgv"), "backing_vocal"),
    (("lead vocal", "lead vox", "ldvoc", "vocal", "vox", "voice", "sing", "vox lead"), "lead_vocal"),
    (("kick", "bd", "bass drum"), "kick"),
    (("snare", "sd", "rim"), "snare"),
    (("hi hat", "hihat", "hat", "hh"), "hats"),
    (("tom",), "toms"),
    # overheads and room mics are drum-kit microphones, not instruments: an
    # overhead pair is the whole kit from above, a room mic the kit in the
    # room.  Short words, so they match as whole words only (_WHOLE_WORD).
    (("overheads", "overhead", "over head", "ohs", "oh"), "overheads"),
    (("rooms", "room", "rm"), "room"),
    (("cymbal", "ride", "crash"), "cymbals"),
    (("drum", "drums", "drumbus", "kit"), "drums_full"),
    (("percussion", "shaker", "tambourine", "conga", "bongo", "cajon", "cowbell"), "percussion"),
    (("guitar ac", "acoustic guitar", "ac gtr", "acgtr", "agtr", "acoustic", "gtr ac",
      "banjo", "mandolin", "sitar"), "guitar_acoustic"),
    (("guitar", "gtr", "electric guitar", "eg", "gtr el", "pedal steel", "lap steel"), "guitar_electric"),
    # before the piano row: "upright" alone is a piano, an upright bass is not
    (("upright bass", "double bass"), "bass"),
    (("electric piano", "epiano"), "keys"),
    (("piano", "pno", "grand", "upright"), "piano"),
    (("keys", "rhodes", "wurli", "organ", "ep", "harpsichord", "celesta", "accordion"), "keys"),
    (("pad", "syn pad", "atmos", "texture"), "pad"),
    (("string", "violin", "cello", "viola", "harp"), "strings"),
    # before brass and bass: an English horn is a woodwind, and a bassoon or a
    # bass clarinet is not a bass
    (("woodwind", "bassoon", "bsn", "bass clarinet", "bass flute", "clarinet", "flute", "oboe",
      "piccolo", "recorder", "english horn"), "woodwinds"),
    (("brass", "horn", "trumpet", "trombone", "tuba", "cornet", "flugel", "euphonium", "sax"), "brass"),
    (("bass", "bassline", "bs"), "bass"),
    # "lead" alone is not here: in a vocal session "Lead" is the lead vocal
    # (see _LEAD_VOCAL_RE); a synth lead names its sound ("square lead")
    (("synth", "syn lead", "square", "supersaw", "sawtooth", "saw lead", "lead saw", "sawlead", "leadsaw",
      "arp", "pluck"), "synth"),
    (("fx", "sfx", "sound effect", "effects", "riser", "impact", "noise"), "fx"),
)

#: Chinese filename keywords -> role.  Both traditional and simplified
#: spellings, because the same file travels between them.  Without this a
#: stem called "主唱.wav" came out as "other" and the user had to set every
#: role by hand.  Checked before the English table: Chinese names are more
#: specific than the short English tokens ("sd", "bd") they might contain.
_KEYWORDS_ZH: tuple[tuple[tuple[str, ...], str], ...] = (
    (("主唱", "主旋律", "獨唱", "独唱"), "lead_vocal"),
    (("和聲", "和声", "和音", "合聲", "合声"), "backing_vocal"),
    (("人聲", "人声", "聲部", "声部"), "lead_vocal"),
    # before 低音 (bass) and 鼓 (drums): a bass clarinet is a woodwind, a
    # tambourine is percussion
    (("單簧管", "单簧管", "長笛", "长笛", "雙簧管", "双簧管", "低音管", "巴松", "短笛", "木管"), "woodwinds"),
    (("打擊樂", "打击乐", "沙鈴", "沙铃", "鈴鼓", "铃鼓", "手鼓", "康加"), "percussion"),
    (("底鼓", "大鼓", "腳鼓", "脚鼓"), "kick"),
    (("軍鼓", "军鼓", "小鼓"), "snare"),
    (("踩鈸", "踩镲", "踩叉"), "hats"),
    (("吊镲", "吊鈸", "叮叮", "鈹", "镲"), "cymbals"),
    (("桶鼓", "嗵鼓"), "toms"),
    # microphone words only: a bare "房間" or "頂部" also names a room piano
    # or a top-line synth
    (("頂部麥", "顶部麦", "頂麥", "顶麦"), "overheads"),
    (("房間麥", "房间麦", "環境麥", "环境麦"), "room"),
    (("鼓組", "鼓组", "整套鼓", "鼓"), "drums_full"),
    (("電吉他", "电吉他"), "guitar_electric"),
    (("木吉他", "民謠吉他", "民谣吉他"), "guitar_acoustic"),
    (("吉他",), "guitar_electric"),
    (("鋼琴", "钢琴"), "piano"),
    (("電鋼", "电钢", "電子琴", "电子琴", "鍵盤", "键盘"), "keys"),
    (("貝斯", "贝斯", "低音"), "bass"),
    (("弦樂", "弦乐", "小提琴", "大提琴"), "strings"),
    (("銅管", "铜管", "小號", "小号", "薩克斯", "萨克斯"), "brass"),
    (("合成器", "合成"), "synth"),
    (("氛圍", "氛围", "鋪底", "铺底"), "pad"),
    (("環境", "环境", "效果", "特效", "音效"), "fx"),
)

#: roles that belong to the drum group (they get the drum compression amount)
DRUM_ROLES = frozenset({"kick", "snare", "hats", "toms", "cymbals", "overheads", "room", "drums_full",
                        "percussion"})
#: roles whose keywords only count as whole words ("oh" is not in "john",
#: "room" is not in "mushroom", "rm" is not in "harmony")
_WHOLE_WORD = frozenset({"overheads", "room"})
#: single keywords that are whole words only, whatever their row: "eg" is in
#: "arpeggio" and "leg", "ep" in "deep" and "step", "harp" in "sharp"
_WHOLE_WORD_TOKENS = frozenset({"eg", "ep", "harp"})
VOCAL_ROLES = frozenset({"lead_vocal", "backing_vocal"})

#: The decoder policies a load (and a plan) may name.
#:
#: ``float32_v2`` is the current default.  Like ``float32_v1`` it decodes a
#: source that ffmpeg has to decode to 32-bit float PCM, so a 24-bit file
#: keeps every bit it has.  What it adds is the channel rule: a mono or stereo
#: file is decoded at its own channel count and then goes through the same
#: channel handling as a WAV, so a mono FLAC is duplicated at unity exactly
#: like a mono WAV.  ``float32_v1`` let ffmpeg turn mono into stereo, and
#: ffmpeg's mono-to-stereo matrix puts 1/sqrt(2) on each side -- the same
#: samples loaded 3.01 dB quieter from a FLAC than from a WAV, which moved the
#: source balance of any session that mixed formats.  Sources with more than
#: two channels are still downmixed by ffmpeg, as before.
#:
#: ``float32_v1`` and ``legacy_pcm16_v1`` (ffmpeg to 16-bit PCM, the oldest
#: behaviour) are kept unchanged so a plan written with either still replays
#: exactly the way it did.
#:
#: The name describes the *decoder's output precision*.  It is not a promise
#: that a lossy codec was undone: an MP3 is still an MP3 after it is decoded,
#: it just is no longer quantised again on the way in.  It is not a promise of
#: lossless preservation either -- the samples are held as float32.
DECODE_POLICY_FLOAT32 = "float32_v1"
DECODE_POLICY_FLOAT32_V2 = "float32_v2"
DECODE_POLICY_LEGACY_PCM16 = "legacy_pcm16_v1"
DECODE_POLICIES = (DECODE_POLICY_FLOAT32, DECODE_POLICY_FLOAT32_V2, DECODE_POLICY_LEGACY_PCM16)

#: the policy a loader uses when the caller does not name one
DEFAULT_DECODE_POLICY = DECODE_POLICY_FLOAT32_V2

#: the policy a plan that does not record one is replayed with
LEGACY_DECODE_POLICY = DECODE_POLICY_LEGACY_PCM16

#: ffmpeg output codec per policy.  The codec is looked up here and never
#: taken from the caller: an arbitrary codec string would be an arbitrary
#: decoder, and this ticket is about making the decoder explicit.
_FFMPEG_CODEC = {
    DECODE_POLICY_FLOAT32: "pcm_f32le",
    DECODE_POLICY_FLOAT32_V2: "pcm_f32le",
    DECODE_POLICY_LEGACY_PCM16: "pcm_s16le",
}


class StemError(RuntimeError):
    pass


def check_decode_policy(decode_policy: object) -> str:
    """Return a valid decoder-policy name, or refuse.

    The name is checked before anything is opened or decoded, so a typo can
    never silently pick a precision: an unknown name, ``None`` or a non-string
    is an error, not a fallback to some other policy.
    """
    if not isinstance(decode_policy, str) or decode_policy not in DECODE_POLICIES:
        raise StemError(
            f"unknown decode policy {decode_policy!r}; "
            f"expected one of {', '.join(DECODE_POLICIES)}"
        )
    return decode_policy


@dataclass
class Stem:
    """One dropped file, decoded to stereo float32 at its native rate.

    ``decode_policy`` records which decoder produced the samples when the
    loader knows it (``float32_v2``, ``float32_v1`` or ``legacy_pcm16_v1``).  The field is
    appended last, so stems that are built by hand -- and the existing
    positional callsites -- keep working; it stays ``None`` when nothing
    loaded the file and the samples' origin is unknown.
    """

    name: str
    path: str
    audio: np.ndarray
    sr: int
    role: str = "other"
    role_source: str = "keyword"
    metrics: dict = field(default_factory=dict)
    #: length of the file as loaded; ``duration`` follows the audio, which
    #: ``align`` pads to the session length
    source_duration: float | None = None
    #: which decoder produced these samples; ``None`` when unknown
    decode_policy: str | None = None

    def __post_init__(self) -> None:
        if self.source_duration is None:
            self.source_duration = self.audio.shape[0] / float(self.sr)

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / float(self.sr)

    def mono(self) -> np.ndarray:
        return self.audio.mean(axis=1)


def _parse_wav(data: bytes, path: str) -> tuple[int, int, int, int, bytes]:
    """Return (format_tag, channels, sample_rate, bits, pcm_bytes).

    Written by hand because the stdlib's ``wave`` refuses IEEE-float WAVs
    outright ("unknown format: 3"), and 32-bit float is one of the standard
    things a DAW exports.  Chunks are walked rather than assumed to be in a
    fixed order, and WAVE_FORMAT_EXTENSIBLE is unwrapped to its real format.
    """
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise StemError(f"{os.path.basename(path)}: not a RIFF/WAVE file")

    fmt_body = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + 8
        end = body + size
        if end > len(data):
            end = len(data)  # truncated/streaming file: take what is there
        if cid == b"fmt ":
            fmt_body = data[body:end]
        elif cid == b"data":
            payload = data[body:end]
        pos = body + size + (size & 1)
        if size == 0xFFFFFFFF:  # streaming size marker
            pos = body + (len(data) - body)

    if fmt_body is None or len(fmt_body) < 16:
        raise StemError(f"{os.path.basename(path)}: missing fmt chunk")
    if payload is None:
        raise StemError(f"{os.path.basename(path)}: missing data chunk")

    tag = int.from_bytes(fmt_body[0:2], "little")
    channels = int.from_bytes(fmt_body[2:4], "little")
    sample_rate = int.from_bytes(fmt_body[4:8], "little")
    bits = int.from_bytes(fmt_body[14:16], "little")

    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag is the subformat GUID
        if len(fmt_body) < 26:
            raise StemError(f"{os.path.basename(path)}: truncated extensible header")
        tag = int.from_bytes(fmt_body[24:26], "little")

    if channels < 1 or sample_rate <= 0 or bits <= 0:
        raise StemError(
            f"{os.path.basename(path)}: implausible WAV header "
            f"(channels={channels}, rate={sample_rate}, bits={bits})"
        )
    return tag, channels, sample_rate, bits, payload


def _decode_pcm(tag: int, bits: int, payload: bytes, path: str) -> np.ndarray:
    name = os.path.basename(path)
    if tag == 3:  # IEEE float
        if bits == 32:
            return np.frombuffer(payload, dtype="<f4").astype(np.float64)
        if bits == 64:
            return np.frombuffer(payload, dtype="<f8").astype(np.float64)
        raise StemError(f"{name}: unsupported float width {bits}")

    if tag != 1:
        raise StemError(f"{name}: unsupported WAV format tag {tag} (PCM or float only)")

    if bits == 8:
        return (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if bits == 24:
        b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 0x800000, v - 0x1000000, v)
        return v.astype(np.float64) / 8388608.0
    if bits == 32:
        return np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    raise StemError(f"{name}: unsupported PCM width {bits}")


def _read_wav_frames(path: str) -> tuple[np.ndarray, int]:
    """Read any common WAV into (frames, channels) float64 at its own channel count."""
    with open(path, "rb") as fh:
        data = fh.read()

    tag, n_ch, sr, bits, payload = _parse_wav(data, path)
    samples = _decode_pcm(tag, bits, payload, path)

    # drop a trailing partial frame rather than reshaping into nonsense
    n_frames = samples.size // n_ch
    samples = samples[: n_frames * n_ch].reshape(-1, n_ch) if n_frames else np.zeros((0, n_ch))
    return samples, sr


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read any common WAV into stereo float32."""
    return _to_stereo(*_read_wav_frames(path))


def _to_stereo(samples: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """The WAV channel rule: mono duplicated at unity, surround cut to two."""
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    elif samples.shape[1] > 2:
        samples = samples[:, :2]  # downmix surround to the first two channels

    # float32 is what a DAW holds audio in, and it halves the footprint of a
    # 12-stem, 3-minute session.  24-bit source material has less resolution
    # than float32 carries, so nothing is lost.
    return np.ascontiguousarray(samples, dtype=np.float32), sr


#: where ffmpeg usually lives when the App is opened from Finder, which gives
#: it a PATH without the package managers' folders (Apple silicon Homebrew,
#: Intel Homebrew, MacPorts)
FFMPEG_FALLBACKS = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/local/bin/ffmpeg")


def _ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return next((p for p in FFMPEG_FALLBACKS if os.path.exists(p)), None)


def _ffmpeg_available() -> bool:
    return _ffmpeg_path() is not None


def _decode_with_ffmpeg(
    path: str, decode_policy: str = DEFAULT_DECODE_POLICY
) -> tuple[np.ndarray, int]:
    """Decode a non-WAV source with ffmpeg, under the named policy.

    ``legacy_pcm16_v1`` writes 16-bit PCM exactly as before, ``float32_v1``
    writes 32-bit float PCM; both let ffmpeg convert the channels to stereo.
    ``float32_v2`` first decodes a tenth of a second to learn the source's
    channel count.  A mono or stereo source is then decoded at its own count
    and takes the WAV channel rule, so a mono file is duplicated at unity
    instead of through ffmpeg's 1/sqrt(2) mono-to-stereo matrix.  A source with
    more channels is decoded with ``-ac 2``, exactly as ``float32_v1`` does.  The sample rate, the argument order and the
    temporary file are the same for every policy, so the timeline does not
    move.
    """
    policy = check_decode_policy(decode_policy)
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise StemError(
            f"{os.path.basename(path)}: not a WAV and ffmpeg was not found "
            "(install it, e.g. `brew install ffmpeg`, or export the stems as WAV)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "decoded.wav")

        def run(channels: list[str]) -> None:
            cmd = [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                path,
                *channels,
                "-c:a",
                _FFMPEG_CODEC[policy],
                out,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.exists(out):
                raise StemError(
                    f"{os.path.basename(path)}: ffmpeg failed: {proc.stderr.strip()[:300]}"
                )

        channels = ["-ac", "2"]
        if policy == DECODE_POLICY_FLOAT32_V2:
            # a tenth of a second is enough to learn the channel count; the
            # whole file is then decoded once, never at a surround width.  A
            # stream that changes its layout later is decoded at the width it
            # starts with (ffmpeg converts the rest to it).
            run(["-t", "0.1"])
            with open(out, "rb") as fh:
                n_ch = _parse_wav(fh.read(), path)[1]
            if n_ch <= 2:
                channels = []
        run(channels)
        return _read_wav(out)


AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def load_stem_file(
    path: str, decode_policy: str = DEFAULT_DECODE_POLICY
) -> tuple[np.ndarray, int]:
    """Load one audio file under the named decoder policy.

    The policy is validated before the file is opened, WAV or not, so an
    invalid name is refused instead of quietly picking a precision.  The WAV
    path is the same under both policies: a WAV carries its own format in its
    header and is read by it, so the policy only decides what happens to the
    files ffmpeg has to decode.
    """
    policy = check_decode_policy(decode_policy)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return _read_wav(path)
    return _decode_with_ffmpeg(path, policy)


def load_stems(
    stem_dir: str, decode_policy: str = DEFAULT_DECODE_POLICY
) -> list[Stem]:
    """Load every audio file in a directory, sorted by name for repeatability.

    Every stem that comes out of here carries the policy it was loaded with,
    so a plan built from these stems can record how they were decoded.  The
    sample rate, the channel handling, the order and the resampling of odd
    rates are unchanged.
    """
    policy = check_decode_policy(decode_policy)
    if not os.path.isdir(stem_dir):
        raise StemError(f"not a directory: {stem_dir}")
    names = sorted(
        n
        for n in os.listdir(stem_dir)
        if os.path.splitext(n)[1].lower() in AUDIO_EXTS and not n.startswith(".")
    )
    if not names:
        raise StemError(f"no audio files in {stem_dir}")

    stems: list[Stem] = []
    for name in names:
        path = os.path.join(stem_dir, name)
        audio, sr = load_stem_file(path, policy)
        if audio.shape[0] == 0:
            raise StemError(f"{name}: empty file")
        stems.append(
            Stem(name=name, path=path, audio=audio, sr=sr, decode_policy=policy)
        )

    # the mixer needs one common sample rate; resample the odd ones out
    rates = {s.sr for s in stems}
    if len(rates) > 1:
        target = max(rates)
        for s in stems:
            if s.sr != target:
                s.audio = dsp.resample_fft(s.audio, s.sr, target)
                s.resampled_from = s.sr
                s.sr = target
    return stems


# --------------------------------------------------------------- role guess

#: Bounded aliases for explicit source and microphone names (R37/R73).
#:
#: ``SubKick`` names one specific kick microphone, so it must outrank the
#: bare generic ``sub`` substring; ``Sn Top`` / ``Sn Btm`` name the two
#: snare positions.  Both patterns are anchored on purpose: longer words that
#: merely start with the same letters ("Subkickstarter", "sn_topography",
#: "SnTopSecret"), serial numbers ("SN_12345") and microphone model numbers
#: ("SM57", "Beta 91") are *not* instruments and must not be guessed.
#:
#: The lookbehind allows a leading recording number ("3SnTop") and the
#: lookahead allows a trailing number or separator ("SnTop57", "Sub-Kick_Out")
#: while refusing a trailing letter.
_SUBKICK_ALIAS_RE = re.compile(r"(?<![a-z])(?:sub[\s._/-]*kick|(?:kick|k)[\s._/-]*sub)(?![a-z])")
_SNARE_ALIAS_RE = re.compile(r"(?<![a-z])sn[\s._/-]*(?:top|bottom|btm)(?![a-z])")
#: Mask the internal "sd" in an explicit Bass DI label, not in BassDinosaur.
_BASS_DI_ALIAS_RE = re.compile(r"(?<![a-z])bass[\s._/-]*di(?![a-z])")

#: The NWE session names its tom microphones by position instead of by
#: instrument: "T1 Top NWE", "T2 Btm NWE", "Floor NWE".  Such a name carries no
#: instrument word at all -- "Top" is not "tom", and "Floor" names the
#: floor-tom position rather than a drum kit -- so the ordinary keyword table
#: cannot see a tom in it and the stem used to end up as "other".
#:
#: The whole normalised basename has to have that shape and it has to carry
#: the NWE marker: a longer word that merely contains the position ("T1
#: Topography NWE") or a competing label in front of it ("Noise Floor NWE") is
#: not a tom name.  A recording number in front of the name is allowed,
#: because the session numbers its takes.
#: A track called just "Lead" (or "Ld", "LV", "Vx", with a number or a take
#: name around it) is the lead vocal.  Real sessions name it that way, and
#: "lead" used to sit in the synth row, so every such vocal was mixed as a
#: synth: 6 dB down, synth HPF, no vocal bus.  Only reached when no instrument
#: word in the name claimed it first ("Lead Gtr", "Lead Synth", "Square Lead").
_LEAD_VOCAL_RE = re.compile(r"(?<![a-z])(?:lead|ld|lv|vx)(?![a-z])")

#: Vocal words that are too short or too common to test as substrings ("bs"
#: inside "libs" made "Ad Libs" a bass).  Whole words only, and a named
#: instrument in the same name still wins ("Choir Pad", "Backing Keys").
_VOCAL_WORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    # not "backing": a backing track is the band (test_mic_filename_roles)
    (("adlib", "adlibs", "ad lib", "ad libs", "harm", "harms", "choir",
      "ooh", "oohs", "aah", "aahs"), "backing_vocal"),
    (("rap", "topline"), "lead_vocal"),
)


#: Session abbreviations, from UTUVO Prep's music naming presets (the names
#: it reads and the short names it writes).  Too short to test as substrings
#: ("fl" is in "flat", "oud" in "loud"), so whole words only, and only after
#: the keyword table found nothing: a word that names something else always
#: wins ("Kick FL" is a kick).  They are checked before the lone "Lead" rule,
#: so "Lead Tpt" is a trumpet, not a lead vocal.
_ABBREVIATIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("k in", "k out", "k sub"), "kick"),
    (("snr", "sn t", "sn b"), "snare"),
    (("crotch",), "room"),
    (("perc", "auxperc", "tamb"), "percussion"),
    (("acgt", "oud"), "guitar_acoustic"),
    (("syn", "sy"), "synth"),
    (("strs", "vln", "vla", "vc", "vn", "va", "harp", "hp"), "strings"),
    (("tpt", "tbn", "tba", "hn", "fhn", "crt", "fgh", "euph"), "brass"),
    (("fl", "afl", "bfl", "picc", "cl", "bcl", "cbcl", "ob", "eh", "bsn", "bn", "cbn"), "woodwinds"),
)
#: Words that mean something else beside another word ("LV Db" is a lead
#: vocal double, not a double bass; "Song ft. Guest" is a featuring credit;
#: "12ch-loop" a test file; "double" a sample format): only a name that is
#: nothing but the word -- with a take number or a side (L/R) -- counts.
_ALONE_RE = re.compile(r"(?:\d+ )?(db|cb|v|ft|loop|double)(?: (?:\d+|l|r|left|right))*")
_ALONE_ROLE = {"db": "bass", "cb": "bass", "v": "lead_vocal", "ft": "toms", "loop": "synth",
               "double": "backing_vocal"}
#: Vocal abbreviations that only count when no lead word is there: "Lead Dbl"
#: is the lead vocal's double, a lone "Dbl" a backing double.
_LATE_VOCAL: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dbl",), "backing_vocal"),
    (("voc",), "lead_vocal"),
)


def _abbreviation(word: str, key: str) -> bool:
    """An abbreviation standing on its own: no plural ("SYS" is not two "SY")."""
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z])", key) is not None


def _whole_word(word: str, key: str) -> bool:
    """``word`` standing on its own in ``key`` (a plural "s" is allowed)."""
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}s?(?![a-z])", key) is not None


def _names_an_instrument(key: str) -> bool:
    """True when a non-vocal instrument word stands on its own in ``key``
    ("Harm Tpt" is a trumpet, whatever "Harm" says)."""
    return any(
        role not in VOCAL_ROLES and any(_whole_word(w, key) for w in words)
        for words, role in _KEYWORDS
    ) or any(any(_abbreviation(w, key) for w in words) for words, _role in _ABBREVIATIONS)


_NWE_TOM_RE = re.compile(
    r"^(?:\d+\.? )?(?:t[123] (?:top|bottom|btm)|floor) nwe(?: \d+)?$"
)


def _alias_role(key: str) -> tuple[str | None, str]:
    """Match the bounded source/microphone aliases in a normalised name.

    Returns the alias role (``None`` when nothing matched) together with the
    name with every alias span blanked out.  The ordinary keyword table then
    runs against what is left, so an explicit competing label ("808",
    "subbass", "bass", "lead vocal") keeps its original priority, while the
    substring noise that lives *inside* the alias itself ("sub" in
    "SubKick", "tom" in "sn bottom", "sd" in "BassDI") disappears with the span.

    Two *different* aliases in one name ("SubKick SnTop") are ambiguous: the
    name does not say which instrument it is, so the alias layer stands down
    and the ordinary keyword result is kept instead of letting the word order
    pick a role.  Repeating the same alias is not ambiguous and still
    resolves ("SnTop SnBtm" is a snare, "SubKick SubKick" is a kick).
    """
    spans: list[tuple[int, int, str]] = []
    for pattern, role in ((_SUBKICK_ALIAS_RE, "kick"), (_SNARE_ALIAS_RE, "snare"), (_BASS_DI_ALIAS_RE, "bass")):
        for m in pattern.finditer(key):
            spans.append((m.start(), m.end(), role))
    if not spans:
        return None, key
    if len({role for _, _, role in spans}) > 1:
        return None, key
    spans.sort()
    parts: list[str] = []
    pos = 0
    for start, end, _role in spans:
        parts.append(key[pos:start])
        parts.append(" ")
        pos = end
    parts.append(key[pos:])
    return spans[0][2], "".join(parts)


def _keyword_role(name: str) -> str | None:
    key = os.path.splitext(name)[0].lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    for words, role in _KEYWORDS_ZH:
        for w in words:
            if w in key:
                return role
    whole = key
    # a microphone position ("Sn Top", "SubKick", "BassDI") is an instrument too
    if _alias_role(key)[0] is None:
        for words, role in _VOCAL_WORDS:
            if any(_whole_word(w, key) for w in words) and not _names_an_instrument(key):
                return role
    alias, key = _alias_role(key)
    for index, (words, role) in enumerate(_KEYWORDS):
        # A named bass guitar is a bass even when it is acoustic.  Check the
        # compound at the guitar priority, after explicit vocal/drum/808 labels;
        # a bass-like substring inside another word is not an instrument name.
        if role in ("guitar_acoustic", "guitar_electric") and re.search(
            r"(?<![a-z])bass ?(?:guitar|gtr)(?![a-z])", key
        ):
            return "bass"
        if role == "guitar_acoustic" and re.search(r"(?<![a-z])ukulele(?![a-z])", key):
            return role
        for w in words:
            if role in _WHOLE_WORD:
                if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z])", key):
                    # "piano room" is a piano recorded in a room, not a drum
                    # room mic: a named instrument further down the table wins
                    if any(
                        later_role not in DRUM_ROLES and any(lw in key for lw in later_words)
                        for later_words, later_role in _KEYWORDS[index + 1:]
                    ):
                        break
                    return role
            elif w in _WHOLE_WORD_TOKENS:
                if _whole_word(w, key):
                    return role
            elif w in key:
                return role
    # A named position beats no name at all: "T1 Top NWE" is a tom even though
    # no keyword in it says so.  By now every explicit label in the name
    # (808, bass, lead vocal, ...) has already had its say, and the
    # whole-basename test keeps the position from swallowing anything else.
    if _NWE_TOM_RE.match(whole):
        return "toms"
    if alias is None:
        alone = _ALONE_RE.fullmatch(key)
        if alone:
            return _ALONE_ROLE[alone.group(1)]
        for words, role in _ABBREVIATIONS:
            if any(_abbreviation(w, key) for w in words):
                return role
        if _LEAD_VOCAL_RE.search(key):
            return "lead_vocal"
        for words, role in _LATE_VOCAL:
            if any(_abbreviation(w, key) for w in words):
                return role
    return alias


#: band edges matching the preset's tonal tilt vocabulary
BAND_EDGES = (20.0, 160.0, 500.0, 2000.0, 6000.0, 18000.0)
BAND_NAMES = ("low", "low_mid", "mid", "presence", "air")


def _hum_hz(seg: np.ndarray, sr: int) -> float:
    """Mains hum frequency if the track has one, else 0.

    Deliberately only reports the frequency: whether it is safe to notch is a
    decision for the planner, which knows the track's role.
    """
    try:
        return float(detect_hum(seg, sr)["hz"])
    except Exception:  # noqa: BLE001 - a metering nicety must never fail a mix
        return 0.0


def _band_levels(seg: np.ndarray, sr: int, spectrum=None) -> dict:
    """Per-band level in dBFS, measured on the spectrum."""
    m = seg.mean(axis=1) if seg.ndim == 2 else seg
    if m.shape[0] < 2048:
        return {k: -120.0 for k in ("low", "low_mid", "mid", "presence", "air")}
    if spectrum is not None:
        freqs, spec = spectrum
    else:
        freqs, spec = _power_spectrum(m, sr)
    out = {}
    edges = BAND_EDGES
    names = ("low", "low_mid", "mid", "presence", "air")
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        band = spec[(freqs >= lo) & (freqs < hi)]
        if band.size == 0:
            out[name] = -120.0
            continue
        # mean power in the band -> equivalent sine RMS, so it is comparable
        # across bands of different width
        power = float(band.mean()) / max(1, m.shape[0]) ** 2
        out[name] = round(dsp.lin_to_db(float(np.sqrt(power))), 1)
    return out


def _band_fractions(audio: np.ndarray, sr: int, exclude_hz: float = 0.0) -> dict:
    """Fractional energy per band; the five values sum to about 1."""
    frac = dsp.band_energy(audio, sr, edges=BAND_EDGES, exclude_hz=exclude_hz)
    return {name: round(float(v), 4) for name, v in zip(BAND_NAMES, frac)}


def _low_peak_hz(
    mono: np.ndarray,
    sr: int,
    lo: float = 30.0,
    hi: float = 200.0,
    exclude_hz: float = 0.0,
) -> float:
    """Dominant frequency of the low end, where kick and bass fight.

    Used to place the low-end carve: cutting at the kick's actual fundamental
    is worth far more than cutting "somewhere around 60 Hz".
    """
    n = mono.shape[0]
    if n < 2048:
        return 0.0
    seg = mono * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    if exclude_hz > 0:
        spec = spec.copy()
        spec[np.abs(freqs - exclude_hz) <= 3.0] = 0.0
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    band = spec[m]
    if float(band.sum()) < 0.02 * float(spec.sum()):
        return 0.0  # nothing meaningful down here; do not report a fake fundamental
    return float(freqs[m][int(np.argmax(band))])


def _flatness(mono: np.ndarray, sr: int) -> float:
    """Spectral flatness, 0 = tonal, 1 = noise.  Tells a snare from a note."""
    n = mono.shape[0]
    if n < 2048:
        return 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))


def detect_hum(seg: np.ndarray, sr: int) -> dict:
    """Look for mains hum: a steady 50/60 Hz tone that keeps going in the gaps.

    A bass note at 50 Hz is narrow too, so a single spectrum cannot tell them
    apart.  The difference is *when*: hum is there during the quiet passages as
    well, a played note is not.  So the check looks at the frames where the
    track is quietest and asks whether the hum bin is still strong there.

    The frame spectra are computed **once**, in batches, and both candidate
    frequencies read from the same numbers.  Doing it the obvious way -- an
    FFT per frame per candidate -- meant ~1900 FFTs for a two-minute stem and
    made measuring the second slowest step of the whole pipeline.
    """
    out = {"hz": 0.0, "quiet_db": -120.0, "loud_db": -120.0}
    if seg.shape[0] < int(0.5 * sr):
        return out
    mono = seg.mean(axis=1).astype(np.float64)

    frame = int(0.25 * sr)
    hop = int(0.125 * sr)
    if mono.shape[0] < frame + hop:
        return out
    n_frames = 1 + (mono.shape[0] - frame) // hop
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)

    def bin_at(hz):
        idx = np.argmin(np.abs(freqs - hz))
        lo, hi = max(0, idx - 1), min(freqs.size, idx + 2)
        return lo, hi

    candidates = []
    for hz in (50.0, 60.0):
        lo, hi = bin_at(hz)
        near = (freqs >= hz - 20.0) & (freqs <= hz + 20.0) & (np.abs(freqs - hz) > 3.0)
        candidates.append({"hz": hz, "lo": lo, "hi": hi, "near": near})

    # One batched pass over the frames; only per-frame scalars are kept, the
    # spectra themselves are discarded as soon as they are summarised.
    total = np.empty(n_frames)
    hum = {c["hz"]: np.empty(n_frames) for c in candidates}
    band = {c["hz"]: np.empty(n_frames) for c in candidates}
    floor = {c["hz"]: np.empty(n_frames) for c in candidates}
    peak = np.empty(n_frames)
    outside = {c["hz"]: np.empty(n_frames) for c in candidates}

    from numpy.lib.stride_tricks import sliding_window_view

    views = sliding_window_view(mono, frame)[::hop][:n_frames]
    chunk = 64
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        spec = np.abs(np.fft.rfft(views[start:stop] * window, axis=1)) ** 2
        total[start:stop] = spec.sum(axis=1) + 1e-20
        peak[start:stop] = spec.max(axis=1)
        for c in candidates:
            lo, hi = c["lo"], c["hi"]
            hum[c["hz"]][start:stop] = spec[:, lo:hi].sum(axis=1)
            band[c["hz"]][start:stop] = spec[:, lo:hi].mean(axis=1)
            if np.any(c["near"]):
                floor[c["hz"]][start:stop] = np.median(spec[:, c["near"]], axis=1)
            else:
                floor[c["hz"]][start:stop] = 0.0
            sides = np.concatenate(
                [spec[:, max(0, lo - 3) : lo], spec[:, hi : hi + 3]], axis=1
            )
            outside[c["hz"]][start:stop] = (
                sides.max(axis=1) if sides.size else np.zeros(stop - start)
            )

    best = {"hz": 0.0, "quiet_db": -120.0, "loud_db": -120.0}
    for c in candidates:
        hz = c["hz"]
        # Tonality: a hum line stands well above its neighbours in the same
        # frame.  Noise floor and reverb tails do not.  Without this the
        # "still there in the gaps" test fires on any track with a noise floor
        # -- measured on a realistic guitar stem, a 60 Hz bin read 52 dB
        # "peakier" than its neighbours when in fact the whole low end was flat.
        quiet_idx = np.argsort(total)[: max(1, n_frames // 10)]
        ratios = []
        audible = []
        for i in quiet_idx:
            fl = float(floor[hz][i])
            if fl > 0:
                ratios.append(float(band[hz][i]) / fl)
            # a "hum" 40 dB below the loudest thing in the frame is inaudible;
            # this is what separates a real hum line from the spectral leakage
            # of a bright tone (a pure 440 Hz sine leaked into the 50 Hz bin 94
            # dB down and looked like hum on every other test)
            bd = float(band[hz][i])
            pk = float(peak[i]) or 1e-30
            audible.append(10 * np.log10((bd or 1e-30) / pk))
            # if the "hum" bin sits right next to the loudest bin it is the
            # skirt of a note, not a separate line (a 55 Hz bass reads 60 Hz)
            if float(outside[hz][i]) > 4.0 * max(bd, 1e-30):
                audible[-1] = -120.0
        tonality_db = float(10 * np.log10(np.median(ratios))) if ratios else 0.0
        audible_db = float(np.median(audible)) if audible else -120.0

        quiet_db = float(10 * np.log10(np.median(hum[hz][quiet_idx]) + 1e-20))
        loud_db = float(10 * np.log10(np.percentile(hum[hz], 90) + 1e-20))
        # ... and it has to be audible at all.  The other tests compare the
        # hum with the rest of the *same* frame, so a line that is inaudible
        # everywhere can still look like hum.  Measured on a real vocal stem
        # (MUSDB18) that had no energy below 70 Hz worth speaking of, the
        # 60 Hz bin sat 107 dB below the stem's peak and was still called
        # hum; on a real bass stem the same bin was 56 dB down.  A hum line
        # worth notching is nowhere near that far away.
        stem_peak_db = float(10 * np.log10(float(peak.max()) + 1e-30))
        if quiet_db - stem_peak_db < -50.0:
            continue
        # hum stays within ~12 dB of its loudest in the quietest frames; a played
        # note drops away with the rest of the music
        if (
            loud_db > -120.0
            and quiet_db > loud_db - 12.0
            and tonality_db >= 8.0
            and audible_db > -40.0
        ):
            if quiet_db > best["quiet_db"]:
                best = {"hz": hz, "quiet_db": quiet_db, "loud_db": loud_db}
    return best


def _envelope(mono: np.ndarray, sr: int, hop_s: float = 0.5) -> list[float]:
    """Coarse RMS envelope, normalised to its own median (scale free)."""
    hop = max(1, int(hop_s * sr))
    n = mono.shape[0]
    if n < hop * 4:
        return []
    frames = n // hop
    seg = mono[: frames * hop].astype(np.float64).reshape(frames, hop)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    med = float(np.median(rms)) or 1e-9
    return [round(float(x / med), 3) for x in rms]


def _power_spectrum(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-band power spectrum, computed once and shared by every meter.

    Before this, ``measure`` ran four separate 8.6M-point FFTs per stem for
    low-peak, sibilance, band levels and band fractions -- about 2.4 s of the
    3.5 s a three-minute stem cost.  They are all asking the same question.
    """
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.shape[0], 1.0 / sr)
    return freqs, spec


def _sibilance(seg: np.ndarray, sr: int, spectrum=None) -> float:
    """How much of the vocal-band energy sits in the 5-9 kHz sibilance zone.

    A voice with harsh S sounds carries far more of its presence-band energy up
    there than one without.  It is a crude index on purpose: it only decides
    whether to turn the de-esser on, and the de-esser itself is bounded.
    """
    if seg.shape[0] < 2048:
        return 0.0
    mono = seg.mean(axis=1) if seg.ndim == 2 else seg
    if spectrum is not None:
        freqs, spec = spectrum
    else:
        freqs, spec = _power_spectrum(mono, sr)
    high = float(spec[(freqs >= 5000.0) & (freqs <= 9000.0)].sum())
    body = float(spec[(freqs >= 200.0) & (freqs <= 4000.0)].sum())
    return high / body if body > 0 else 0.0


def _body_resonance(mono: np.ndarray, sr: int, lo: float = 140.0, hi: float = 420.0) -> float:
    """How peaked the 140-420 Hz region is: a drum body resonates, noise does not.

    Peak-versus-median inside the band, so it is scale free.  The strongest bin
    in a hi-hat band is just the loudest slice of the noise, while a snare has
    an actual tuned body there.
    """
    n = mono.shape[0]
    if n < 2048:
        return 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = spec[(freqs >= lo) & (freqs <= hi)]
    if band.size == 0:
        return 0.0
    return float(band.max() / (np.median(band) + 1e-20))


def _envelope_crest_db(audio: np.ndarray, sr: int, frame_s: float = 0.05) -> float:
    """How peaky the amplitude envelope is.

    A sustained source (a bass note, a pad) keeps its envelope near its
    median, so this is small; a struck source (a kick, a snare) spends most
    of its time near silence with a few large peaks, so this is large.

    Measured on 11 real bass stems and 11 real drum-kit stems (MUSDB18):
    the basses sat at 4.6-16.6 dB (median 9.0) and the kits at 13.4-33.7 dB
    (median 19.3).  The old stand-in, a whole-file crest of 8 dB, called
    every one of those basses a kick drum.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    env = np.abs(mono)
    hop = max(1, int(frame_s * sr))
    n = len(env) // hop
    if n < 2:
        return 0.0
    frames = env[: n * hop].reshape(n, hop).max(axis=1)
    med = float(np.median(frames))
    if med <= 1e-9:
        return 60.0
    return 20.0 * float(np.log10((float(frames.max()) + 1e-12) / med))


def _timbre_features(mono: np.ndarray, sr: int) -> dict:
    """Three things that tell the roles apart, measured on the loud frames.

    ``sustain``   how long the loud parts last.  A bass note *holds*; a kick
                  is a transient.  This is what separates a bass played once
                  per bar (a huge crest factor) from a drum.
    ``runs``      how many loud parts there are.  A bass *line* has notes; a
                  single held tone or a one-shot hit has exactly one, and
                  neither is a bass line.
    ``harm``      how periodic the loud part is.  A voice is a vibrating
                  column of air; a drum membrane is not.

    Measured on 44 real stems (11 tracks x 4 roles, MUSDB18): the bass stems
    sat at an average centroid of 187-757 Hz and the drum kits at 2560-7633 Hz;
    the vocal stems were the most periodic (0.67-0.91) and the kits the
    least (0.31-0.56); a bass's loud parts lasted 0.11-1.05 s and a kit's
    0.05-0.13 s.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    frame = int(0.05 * sr)
    hop = max(1, int(0.025 * sr))
    out = {"sustain": 0.0, "harm": 0.0, "runs": 0}
    n = 1 + (len(mono) - frame) // hop
    if n < 2:
        return out
    views = sliding_window_view(mono.astype(np.float64), frame)[::hop][:n]
    window = np.hanning(frame)
    spec = np.abs(np.fft.rfft(views * window, axis=1)) ** 2
    energy = spec.sum(axis=1)
    thr = float(energy.max()) * 10 ** (-12.0 / 10.0)
    loud = energy >= thr
    if not loud.any():
        return out

    # how long the loud parts last
    runs: list[int] = []
    cur = 0
    for flag in loud:
        if flag:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    out["sustain"] = float(np.median(runs)) * hop / float(sr)
    out["runs"] = len(runs)

    # how periodic the loud part is
    peaks = []
    lo = max(1, int(sr / 400))
    hi = min(int(sr / 40), frame - 1)
    for i in np.flatnonzero(loud)[:60]:
        seg = views[i] - views[i].mean()
        ac = np.correlate(seg, seg, "full")[frame - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        if hi > lo:
            peaks.append(float(ac[lo:hi].max()))
    if peaks:
        out["harm"] = float(np.median(peaks))
    return out


def _spectral_role(audio: np.ndarray, sr: int) -> str:
    """Fallback when the filename says nothing useful."""
    start, end = active_span(audio, sr)
    if (end - start) < int(0.25 * sr):
        # a 150 ms one-shot does not contain enough evidence to call it a vocal;
        # "other" is dull but honest, and the preset treats it conservatively
        return "other"
    span = audio[start:end]
    mono = span.mean(axis=1)
    centroid = dsp.spectral_centroid(mono, sr)
    b = _band_fractions(mono, sr)
    low = b["low"] + b["low_mid"]
    mid = b["mid"]
    high = b["presence"] + b["air"]
    crest = dsp.crest_db(span)
    percussive = crest > 8.0
    flat = _flatness(mono, sr)
    low_hz = _low_peak_hz(mono, sr)
    # how much energy is above 2 kHz, how peaky the envelope is, and how
    # centred the stem is -- all measured on real stems, see the module notes
    hf = b["presence"] + b["air"]
    env_crest = _envelope_crest_db(span, sr)
    corr = dsp.stereo_correlation(span)
    # a voice leaves air (sibilance) far above its presence band; a mixed
    # accompaniment of guitars and keys does not
    air_over_presence = b["air"] / b["presence"] if b["presence"] > 1e-9 else 0.0
    tf = _timbre_features(mono, sr)

    # Only answer when the evidence is strong.  A wrong confident role is worse
    # than "other": it applies a vocal's gain, HPF and reverb to a synth pad,
    # and the user has to notice and undo it.  Anything doubtful returns "other"
    # and the UI lets them set the role by hand.

    # A steady 50/60 Hz mains line invents a low end: measured on a real
    # keyboard DI, the hum alone pushed the low band over the threshold and
    # the strongest "fundamental" to 50 Hz, so the track came out as a kick
    # drum.  If the low end *is* the hum, look again without it.
    if low > 0.55 and low_hz and low_hz < 110.0:
        if abs(low_hz - 50.0) < 4.0 or abs(low_hz - 60.0) < 4.0:
            detected = float(detect_hum(span, sr).get("hz", 0.0) or 0.0)
            if detected > 0 and abs(detected - low_hz) < 6.0:
                b = _band_fractions(mono, sr, exclude_hz=detected)
                low = b["low"] + b["low_mid"]
                low_hz = _low_peak_hz(mono, sr, exclude_hz=detected)
                hf = b["presence"] + b["air"]
                env_crest = _envelope_crest_db(span, sr)
                air_over_presence = (
                    b["air"] / b["presence"] if b["presence"] > 1e-9 else 0.0
                )

    # 1. A voice: no low end at all, centred in the stereo field, periodic,
    #    and it carries air (sibilance) well above its presence band.
    #    Measured on the real vocal stems: less than 10.5% of the energy in
    #    the low band, an autocorrelation peak of 0.67-0.91 (the drum kits
    #    never went above 0.56), an L/R correlation of at least 0.58, and an
    #    air-to-presence ratio of at least 0.036.  The accompaniment stems
    #    that come closest (a reggae organ bed, a sparse guitar) failed on
    #    the air ratio.
    if (
        b["low"] < 0.12
        and tf["harm"] >= 0.65
        and corr > 0.5
        and air_over_presence > 0.02
    ):
        return "lead_vocal"

    # 2. A bass: dark, band-limited, and it *holds* its notes.  The hold is
    #    what separates it from a kick -- a bass played once per bar has a
    #    huge crest factor, but its loud parts still last, while a kick is a
    #    transient.  Measured on the real bass stems: a loud centroid below
    #    136 Hz (the drum kits were all above 2560 Hz), no energy above 2 kHz
    #    at all, and loud parts lasting at least 0.11 s.
    if (
        low > 0.5
        and centroid < 800.0
        and hf < 0.005
        and tf["sustain"] >= 0.10
        and tf["runs"] >= 2
    ):
        return "bass"

    # 3. A whole kit rather than one drum: short transients and a bright
    #    average.  The accompaniment stems that used to be caught by the old
    #    "low end with something bright on top" rule were a sustained bed
    #    (14.7 s of loud parts) and two percussive stems whose loud part was
    #    darker than any real kit.
    if tf["sustain"] < 0.13 and centroid > 2500.0 and tf["runs"] >= 3:
        return "drums_full"

    # 4. A low end with a low fundamental that the rules above did not claim.
    if low > 0.55 and low_hz and low_hz < 110.0:
        # A bass stem is band-limited: measured on 11 real bass stems,
        # none had any energy above 2 kHz (presence+air <= 0.001), while
        # every real drum-kit stem did (>= 0.005).  A kick has a beater
        # click and a snare has a body; a bass note has neither.
        if hf < 0.002 and env_crest < 17.0:
            return "bass"
        if percussive:
            return "kick"
        # 55 Hz is an ordinary bass fundamental (A1); an 808 sits lower
        return "sub_808" if low_hz < 50.0 else "bass"

    # 5. A snare has a body around 200 Hz; a hi-hat is the same noise with the
    #    body filtered away, so the presence of low-mid decides between them
    if percussive and flat > 0.2:
        if b["low_mid"] > 0.12 or _body_resonance(mono, sr) > 40.0:
            return "snare"
        if high > 0.35 and centroid > 5000.0:
            return "hats"
    if not percussive and mid > 0.45 and 250.0 <= centroid <= 2500.0:
        # sustained and mid-forward: a pad is the safe read, a vocal is not
        return "pad"
    return "other"


def active_span(audio: np.ndarray, sr: int, floor_db: float = -60.0) -> tuple[int, int]:
    """First and last sample that is actually carrying signal.

    Stems are rarely the same length and rarely start on the downbeat: a vocal
    that enters at bar 9, or a cymbal with a 2 s tail, both measure wrong if
    you average loudness over the padded file.  Measuring on this span is what
    keeps a late-entering vocal from being pushed 6 dB too loud.
    """
    mono = np.max(np.abs(audio), axis=1) if audio.ndim == 2 else np.abs(audio)
    idx = np.flatnonzero(mono > dsp.db_to_lin(floor_db))
    if idx.size == 0:
        return (0, 0)
    return (int(idx[0]), int(idx[-1]) + 1)


def is_silent(stem: "Stem", floor_db: float = -60.0) -> bool:
    a, b = active_span(stem.audio, stem.sr, floor_db)
    return b - a < int(0.02 * stem.sr)


def _role_preview(audio: np.ndarray, sr: int, preview_s: float) -> np.ndarray:
    """A sample of the *whole* active span, not just its beginning.

    The role of a stem is a property of its timbre, not of the arrangement,
    but the first 30 seconds of a song are often a sparse intro: measured on
    real multitracks, a bass stem whose first 30 seconds held one long note
    was judged "kick", while the same stem judged over the whole track was
    judged "bass" correctly.  So the preview is three windows spread across
    the active span -- the same amount of audio, a fairer sample.
    """
    start, end = active_span(audio, sr)
    total = end - start
    want = int(preview_s * sr)
    if total <= want:
        return audio[start:end]
    n_win = 3
    win = max(1, want // n_win)
    if total < win:
        return audio[start:end]
    out = []
    for i in range(n_win):
        pos = start + int((total - win) * i / (n_win - 1))
        out.append(audio[pos : pos + win])
    return np.concatenate(out, axis=0)


def _preview_features(stem: "Stem", preview_s: float = 30.0) -> dict | None:
    """Cheap features on a preview of the active region, for context decisions."""
    seg = _role_preview(stem.audio, stem.sr, preview_s)
    if len(seg) < int(0.05 * stem.sr):
        return None
    mono = seg.mean(axis=1)
    bands = _band_fractions(mono, stem.sr)
    return {
        "low": bands["low"] + bands["low_mid"],
        "high": bands["presence"] + bands["air"],
        "mid": bands["mid"],
        "crest": dsp.crest_db(seg),
        "low_hz": _low_peak_hz(mono, stem.sr),
    }


def track_id(name: str) -> str:
    """The stable id a stem gets in the plan: basename without extension.

    Anything that is not a word character, a dot or a dash becomes an
    underscore.  A file name may legally hold a newline or a path
    separator, and an id that carries one breaks the plan's readability and
    the UI's layout; letters of any script (Chinese included) are kept.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    cleaned = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return cleaned or "track"


def quick_metrics(stem: "Stem", preview_s: float = 30.0) -> dict:
    """Cheap, honest pre-flight numbers for the UI.

    Meant to be shown *before* a render so a clipped or empty stem is obvious
    while the user is still looking at the file list.  Everything here is
    measured on a preview of the active region, which is why it is quick and
    why it does not claim to be an integrated loudness reading.

    The preview samples the whole active region, not its first seconds.  A
    file whose head carries nothing but a weak residue -- above the -60 dB
    floor, so the active region starts there -- would otherwise be measured
    on that silence and reported as very quiet.
    """
    start, end = active_span(stem.audio, stem.sr)
    span = stem.audio[start:end]
    active_s = max(0.0, (end - start) / float(stem.sr))

    if span.shape[0] == 0:
        return {
            "silent": True,
            "active_s": 0.0,
            "peak_db": -120.0,
            "lufs": -70.0,
            "crest_db": 0.0,
            "clipped_ratio": 0.0,
            "low_peak_hz": 0.0,
            "sibilance": 0.0,
        }

    preview = _role_preview(stem.audio, stem.sr, preview_s)
    lufs = (
        dsp.momentary_lufs(preview, stem.sr)
        if preview.shape[0] < int(0.4 * stem.sr)
        else dsp.loudness_lufs(preview, stem.sr)
    )
    return {
        "silent": False,
        "active_s": round(active_s, 2),
        "peak_db": round(dsp.lin_to_db(float(np.max(np.abs(span)))), 2),
        "lufs": round(float(lufs), 2),
        "crest_db": round(dsp.crest_db(preview), 2),
        "clipped_ratio": round(float(np.mean(np.abs(stem.audio) >= 0.999)), 4),
        "low_peak_hz": round(_low_peak_hz(preview.mean(axis=1), stem.sr), 1),
        "sibilance": round(_sibilance(preview, stem.sr), 3),
    }


def refine_roles(stems: "list[Stem]", preview_s: float = 30.0) -> None:
    """Use the whole stem set to fill gaps the per-stem guess left behind.

    A single track cannot tell you it is the bass; the *set* can.  These rules
    only ever promote an ``other`` into a role that nobody else in the session
    already occupies, so a confident guess is never overwritten and an ambiguous
    set still produces ``other`` (which the UI lets the user fix).
    """
    have = {s.role for s in stems}
    others = [s for s in stems if s.role == "other"]
    if not others:
        return

    feats = {s.name: f for s in others if (f := _preview_features(s, preview_s))}
    if not feats:
        return

    def promote(name: str, role: str) -> None:
        for s in stems:
            if s.name == name:
                s.role = role
                s.role_source = "spectral+context"

    # nobody is the low end: the most low-dominant unknown track is it
    if not have & {"bass", "sub_808"}:
        name, f = max(feats.items(), key=lambda kv: kv[1]["low"])
        if f["low"] > 0.35 and f["low_hz"] > 0.0:
            promote(name, "sub_808" if f["low_hz"] < 50.0 else "bass")

    # nobody is the top end: the brightest percussive unknown track is the hats
    if not have & {"hats", "cymbals", "overheads"}:
        name, f = max(feats.items(), key=lambda kv: kv[1]["high"])
        if f["high"] > 0.35 and f["crest"] > 8.0:
            promote(name, "hats")


def detect_roles(stems: list[Stem], preview_s: float = 30.0) -> None:
    """Assign ``role`` in place: filename first, spectrum only as a fallback.

    The spectral fallback only looks at a preview of the active region.  Role is
    a property of timbre, not of arrangement, so 30 seconds carries the same
    information as five minutes -- and this lets the UI show detected roles
    right after an upload instead of after a full analysis pass.
    """
    for s in stems:
        guess = _keyword_role(s.name)
        if guess is not None:
            s.role = guess
            s.role_source = "keyword"
            continue
        preview = _role_preview(s.audio, s.sr, preview_s)
        if len(preview) == 0:
            s.role, s.role_source = "other", "spectral"
            continue
        s.role = _spectral_role(preview, s.sr)
        s.role_source = "spectral"

    refine_roles(stems, preview_s)


# ------------------------------------------------------------------ metering

def measure(stem: Stem) -> dict:
    """Per-stem measurements used by the planner and printed in the receipt.

    Everything level-related is measured on the *active span* rather than the
    whole file.  Padding and long lead-ins would otherwise read as quietness,
    and gain staging would push the stem up until its first note was too loud.
    """
    a = stem.audio
    mono = stem.mono()
    sr = stem.sr

    start, end = active_span(a, sr)
    span = a[start:end]
    span_s = (end - start) / float(sr)
    silent = (end - start) < int(0.02 * sr)

    if not silent:
        # short one-shots have no 400 ms gating block; use momentary loudness
        # on the span instead of pretending the stem is silent
        lufs = (
            dsp.momentary_lufs(span, sr)
            if span_s < 0.4
            else dsp.loudness_lufs(span, sr)
        )
        crest = dsp.crest_db(span)
    else:
        lufs = -70.0
        crest = 0.0

    clip_ratio = float(np.mean(np.abs(a) >= 0.999))
    corr = dsp.stereo_correlation(a)

    # Spectral shape from the mono sum, unless the channels are out of phase:
    # an anti-phase stereo stem sums to (almost) zero, and analysing that
    # silence reports things like "centroid 0 Hz" for a perfectly good pad.
    if silent:
        spec_src = mono
    elif corr < -0.2:
        per_ch = np.sqrt(np.mean(span ** 2, axis=0))
        spec_src = span[:, int(np.argmax(per_ch))]
    else:
        spec_src = span.mean(axis=1)

    # Spectral meters run on a preview: a band fraction or a centroid is a
    # statistical property, and on a 3-minute stem the full-span FFTs cost
    # ~2.3 s each time (three of them per stem).  Loudness, true peak, crest and
    # clipping still use the whole active span.
    preview = spec_src if not silent else np.zeros((0,), dtype=np.float32)
    preview = preview[: int(30.0 * sr)]
    preview_2d = span[: int(30.0 * sr)] if not silent else np.zeros((0, 2), dtype=np.float32)
    spectrum = _power_spectrum(preview, sr) if preview.shape[0] >= 2048 else None

    m = {
        "role": stem.role,
        "role_source": stem.role_source,
        "duration_s": round(stem.duration, 3),
        "sample_rate": sr,
        "channels": int(a.shape[1]),
        "peak_db": round(dsp.lin_to_db(float(np.max(np.abs(a)))), 2),
        "crest_db": round(crest, 2),
        "lufs": round(lufs, 2),
        # per-stem true peak is a diagnostic, not a delivery number: 2x
        # oversampling is enough to spot a stem that is already at the ceiling,
        # and it halves the cost of this meter across a big session.  The
        # master's true peak is measured at the full 4x.
        "true_peak_db": round(dsp.true_peak_db(a, sr, oversample=2), 2),
        "silent": silent,
        "active_s": round(span_s, 3),
        "silence_ratio": round(1.0 - span_s / max(stem.duration, 1e-9), 3),
        "clipped_ratio": round(clip_ratio, 4),
        "low_peak_hz": round(_low_peak_hz(preview, sr), 1) if not silent else 0.0,
        "sibilance": round(_sibilance(preview_2d, sr, spectrum=spectrum), 3),
        "centroid_hz": round(dsp.spectral_centroid(preview, sr), 1) if not silent else 0.0,
        "band_energy": (
            _band_fractions(preview_2d, sr)
            if not silent
            else {name: 0.0 for name in ("low", "low_mid", "mid", "presence", "air")}
        ),
        "hum_hz": _hum_hz(span if not silent else np.zeros((0, 2), dtype=np.float32), sr),
        # absolute level per band: masking is about how loud two sources are in
        # the same band, not about what fraction of each source lives there
        "band_db": (
            _band_levels(preview_2d, sr, spectrum=spectrum)
            if not silent
            else {name: -120.0 for name in ("low", "low_mid", "mid", "presence", "air")}
        ),
        "correlation": round(corr, 3),
        "width": round(dsp.width_ratio(a), 3),
        # Coarse loudness envelope, computed on the *whole* file rather than on
        # the active span: trimming the lead-in made every pattern start at its
        # first hit, which aligned unrelated parts (a kick and a snare on a grid
        # measured r=0.998 and were reported as duplicates).
        "envelope": _envelope(a.mean(axis=1) if a.ndim == 2 else a, sr),
    }
    stem.metrics = m
    return m



def system_memory_mb() -> float | None:
    """Total physical RAM, best effort, no extra dependencies."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            return float(out.stdout.strip()) / 1e6
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - purely advisory
        return None
    return None


def estimate_memory_mb(stems: "list[Stem]") -> float:
    """Rough working-set estimate for mixing this session.

    The engine holds every stem, a handful of bus sums and a few temporaries.
    Measured on a real session, peak RSS lands around 4-5x the raw stem bytes,
    so that is what we report.  Better to refuse a session politely than to be
    killed by the OS halfway through a render.
    """
    n = max((s.audio.shape[0] for s in stems), default=0)
    ch = 2
    bytes_per_stem = n * ch * 4  # float32
    # R116: the balance meter holds a master-EQ'd copy of the four dry buses
    # (float64) and filters one at a time (a complex128 spectrum)
    meter = n * ch * (4 * 8 + 16)
    return (len(stems) * bytes_per_stem * 4.2 + meter) / 1e6


def check_memory(stems: "list[Stem]", budget_mb: float | None = None) -> None:
    need = estimate_memory_mb(stems)
    if budget_mb is None:
        return
    if need > budget_mb:
        raise StemError(
            f"this session needs about {need / 1000:.1f} GB of RAM to mix "
            f"(budget {budget_mb / 1000:.1f} GB); render it in two halves or "
            f"bounce the longest stems shorter"
        )


def align(stems: list[Stem]) -> None:
    """Bring every stem to one sample rate and one length.

    Resampling first matters: a 44.1 kHz stem summed into a 48 kHz session is
    8.8% fast and a semitone and a half sharp, and nothing downstream would
    notice.  The rate that most stems already use wins, so the majority of the
    session is untouched.
    """
    if not stems:
        return

    rates = [s.sr for s in stems]
    target = max(set(rates), key=lambda r: (rates.count(r), r))
    for s in stems:
        if s.sr == target:
            continue
        s.audio = dsp.resample_fft(s.audio, s.sr, target)
        s.resampled_from = s.sr
        s.sr = target
    rates = [s.sr for s in stems]

    n = max(s.audio.shape[0] for s in stems)
    for s in stems:
        if s.audio.shape[0] < n:
            pad = np.zeros((n - s.audio.shape[0], s.audio.shape[1]))
            s.audio = np.vstack([s.audio, pad])
