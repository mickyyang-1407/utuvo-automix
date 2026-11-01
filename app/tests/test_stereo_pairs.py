"""Behaviour of the declared stereo pair.

A stereo pair is two imported mono files that the user says are the left and
the right channel of one stereo microphone pair.  The same-source group puts
every member on one common pan; a pair keeps the two channels apart, and the
member order IS the channel identity.

Everything here is synthetic: short WAV files written into a tmp folder, no
audio from disk and no network.  The matrices are checked against the
renderer's own pan law, so the numbers the plan carries are the numbers the
renderer will use.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from automix import (
    analyze,
    capture_groups as capmod,
    cli,
    pipeline,
    plan as planner,
    render,
)

SR = 48000
TOL = 1e-7


# ------------------------------------------------------------------ fixtures

def _write(path, audio: np.ndarray, sr: int = SR) -> str:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return str(path)


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1)


def _load(folder: str, roles: dict | None = None) -> list:
    stems = analyze.load_stems(folder)
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    for s in stems:
        if roles and s.name in roles:
            s.role = roles[s.name]
    return stems


def _preset() -> dict:
    return planner.load_preset("pop_2020s")


def _plan(stems, **kw) -> dict:
    return planner.build_plan(stems, _preset(), **kw)


def _track(plan: dict, tid: str) -> dict:
    return next(t for t in plan["tracks"] if t["id"] == tid)


def _pair_dir(folder: Path) -> dict:
    """Two independently distinguishable channels plus a third track.

    The right channel is the left one delayed and polarity-inverted: a pair
    whose two members are the same signal would hide a wrong routing.
    """
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    left = 0.30 * np.sin(2 * np.pi * 220 * t)
    delay = 37
    right = 0.24 * np.concatenate([np.zeros(delay), -left[:-delay]])
    _write(str(folder / "left.wav"), _stereo(left))
    _write(str(folder / "right.wav"), _stereo(right))
    _write(str(folder / "drums.wav"), _stereo(0.20 * np.sin(2 * np.pi * 90 * t)))
    return {
        "left.wav": "guitar_electric",
        "right.wav": "guitar_electric",
        "drums.wav": "drums_full",
    }


@pytest.fixture()
def pair_stems(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    roles = _pair_dir(d)
    return _load(str(d), roles)


@pytest.fixture()
def pair_files(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    roles = _pair_dir(d)
    return d, roles


def _declaration(left="left", right="right", **kw) -> dict:
    decl = {
        "id": "pair",
        "members": [left, right],
        "mode": "stereo_pair",
        "pan": 0.0,
        "trim_db": 0.0,
    }
    decl.update(kw)
    return decl


def _facts(stems) -> list[dict]:
    return planner._capture_group_facts(stems)


# ------------------------------------------------------- the declaration


def test_a_pair_keeps_its_two_members_in_the_declared_order(pair_stems):
    groups = capmod.normalize([_declaration()], _facts(pair_stems))
    assert len(groups) == 1
    assert groups[0]["members"] == ["left", "right"]
    assert groups[0]["mode"] == "stereo_pair"
    assert groups[0]["trim_db"] == 0.0
    # the member order is the channel identity: it is never sorted
    swapped = capmod.normalize([_declaration(left="right", right="left")], _facts(pair_stems))
    assert swapped[0]["members"] == ["right", "left"]


def test_the_legacy_declaration_keeps_exactly_its_three_fields(pair_stems):
    groups = capmod.normalize(
        [{"id": "room", "members": ["left", "right"], "pan": -0.2}], _facts(pair_stems)
    )
    assert set(groups[0]) == {"id", "members", "pan"}
    assert "mode" not in groups[0]
    assert "trim_db" not in groups[0]


def test_a_pair_needs_exactly_two_members(pair_stems):
    facts = _facts(pair_stems)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration(left="left", right="left")], facts)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize(
            [{"id": "p", "members": ["left", "right", "drums"], "mode": "stereo_pair"}], facts
        )
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([{"id": "p", "members": ["left"], "mode": "stereo_pair"}], facts)


def test_a_pair_refuses_another_mode(pair_stems):
    facts = _facts(pair_stems)
    for mode in ("mono", "stereo", "STEREO_PAIR", 7, True):
        with pytest.raises(capmod.CaptureGroupError):
            capmod.normalize([_declaration(mode=mode)], facts)


def test_a_shared_trim_without_the_pair_mode_is_refused(pair_stems):
    facts = _facts(pair_stems)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize(
            [{"id": "room", "members": ["left", "right"], "trim_db": 1.0}], facts
        )


def test_a_present_null_mode_is_refused(pair_stems):
    """A mode that is present but is not the one literal mode is a
    declaration we cannot read back.  It is never read as "no mode": that
    would turn an ambiguous pair intent into a collapsed same-source group."""
    facts = _facts(pair_stems)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize(
            [{"id": "p", "members": ["left", "right"], "mode": None}], facts
        )
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize(
            [{"id": "p", "members": ["left", "right"], "mode": None, "trim_db": 0.0}],
            facts,
        )


def test_a_pair_refuses_a_bad_pan_or_trim(pair_stems):
    facts = _facts(pair_stems)
    for pan in (1.5, -1.5, float("nan"), float("inf"), "0.5", True):
        with pytest.raises(capmod.CaptureGroupError):
            capmod.normalize([_declaration(pan=pan)], facts)
    for trim in (12.5, -13.0, float("nan"), float("inf"), "0", True, 10 ** 400):
        with pytest.raises(capmod.CaptureGroupError):
            capmod.normalize([_declaration(trim_db=trim)], facts)


def test_a_pair_refuses_a_track_that_is_not_loaded(pair_stems):
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration(right="nowhere")], _facts(pair_stems))


def test_a_pair_refuses_mixed_roles(pair_files):
    folder, roles = pair_files
    roles["right.wav"] = "drums_full"
    stems = _load(str(folder), roles)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration()], _facts(stems))


def test_a_pair_refuses_mixed_sample_rates(pair_files):
    folder, roles = pair_files
    # the right channel arrives at another rate: it is not the same take
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    _write(str(folder / "right.wav"), _stereo(0.2 * np.sin(2 * np.pi * 220 * t)), sr=44100)
    stems = _load(str(folder), roles)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration()], _facts(stems))


def test_a_pair_refuses_one_sample_of_length(pair_files):
    folder, roles = pair_files
    n = int(1.0 * SR) + 1
    t = np.arange(n) / SR
    _write(str(folder / "right.wav"), _stereo(0.2 * np.sin(2 * np.pi * 220 * t)))
    stems = _load(str(folder), roles)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration()], _facts(stems))


def test_a_pair_refuses_a_silent_member(pair_files):
    folder, roles = pair_files
    n = int(1.0 * SR)
    _write(str(folder / "right.wav"), _stereo(np.full(n, 1e-6)))
    stems = _load(str(folder), roles)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration()], _facts(stems))


def test_a_pair_refuses_a_true_stereo_file(pair_files):
    folder, roles = pair_files
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    left = 0.25 * np.sin(2 * np.pi * 220 * t)
    right = 0.25 * np.sin(2 * np.pi * 330 * t)
    _write(str(folder / "right.wav"), np.stack([left, right], axis=1))
    stems = _load(str(folder), roles)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize([_declaration()], _facts(stems))


def test_a_track_can_only_be_in_one_group(pair_stems):
    facts = _facts(pair_stems)
    with pytest.raises(capmod.CaptureGroupError):
        capmod.normalize(
            [
                _declaration(),
                {"id": "other", "members": ["left", "drums"], "pan": 0.0},
            ],
            facts,
        )


# --------------------------------------------------------- what a pair does


def _routed(plan: dict, stems, members) -> np.ndarray:
    """Where the two members land in the stereo bus, from the plan's numbers."""
    tracks = {t["id"]: t for t in plan["tracks"]}
    audio = {
        analyze.track_id(s.name): np.asarray(s.audio[:, 0], dtype=float) for s in stems
    }
    n = len(audio[members[0]])
    out = np.zeros((n, 2))
    for tid in members:
        t = tracks[tid]
        gain = 10.0 ** (float(t["gain_db"]) / 20.0)
        x = np.stack([audio[tid], audio[tid]], axis=1)
        out += gain * render._pan(x, float(t["pan"]))
    return out


def _oracle(pan: float, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    root2 = math.sqrt(2.0)
    if pan == 0.0:
        return np.stack([left, right], axis=1)
    if pan == 0.5:
        return np.stack([left / root2, left / root2 + right], axis=1)
    if pan == -0.5:
        return np.stack([left + right / root2, right / root2], axis=1)
    if pan == 1.0:
        return np.stack([np.zeros_like(left), left + right], axis=1)
    if pan == -1.0:
        return np.stack([left + right, np.zeros_like(right)], axis=1)
    raise AssertionError(pan)


def _channels(stems) -> tuple[np.ndarray, np.ndarray]:
    audio = {analyze.track_id(s.name): np.asarray(s.audio[:, 0], dtype=float) for s in stems}
    return audio["left"], audio["right"]


def test_a_centred_pair_keeps_the_imported_left_and_right(pair_stems):
    left, right = _channels(pair_stems)
    assert not np.array_equal(left, right)      # the two channels are different
    plan = _plan(pair_stems, capture_groups=[_declaration()])
    routed = _routed(plan, pair_stems, ["left", "right"])
    expected = np.stack([left, right], axis=1)
    assert float(np.max(np.abs(routed - expected))) < TOL


def test_swapping_the_members_swaps_the_channels(pair_stems):
    left, right = _channels(pair_stems)
    plan = _plan(pair_stems, capture_groups=[_declaration(left="right", right="left")])
    routed = _routed(plan, pair_stems, ["right", "left"])
    expected = np.stack([right, left], axis=1)
    assert float(np.max(np.abs(routed - expected))) < TOL


def test_the_shared_position_moves_the_pair_on_the_analytic_matrices(pair_stems):
    left, right = _channels(pair_stems)
    for pan in (0.0, 0.5, -0.5, 1.0, -1.0):
        plan = _plan(pair_stems, capture_groups=[_declaration(pan=pan)])
        routed = _routed(plan, pair_stems, ["left", "right"])
        expected = _oracle(pan, left, right)
        assert float(np.max(np.abs(routed - expected))) < TOL, pan


def test_the_position_moves_the_pair_and_not_a_balance(pair_stems):
    left, right = _channels(pair_stems)
    plan = _plan(pair_stems, capture_groups=[_declaration(pan=0.5)])
    routed = _routed(plan, pair_stems, ["left", "right"])
    # at +0.5 the left member is in the middle and the right one is hard
    # right: the left channel carries a crossfeed of the left member
    assert float(np.max(np.abs(routed[:, 0] - left / math.sqrt(2.0)))) < TOL
    assert float(np.max(np.abs(routed[:, 1] - (left / math.sqrt(2.0) + right)))) < TOL


def test_the_pair_numbers_are_the_mapped_ones(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration(pan=0.5)])
    left = _track(plan, "left")
    right = _track(plan, "right")
    assert float(left["pan"]) == 0.0
    assert float(right["pan"]) == 1.0
    # the compensation for the endpoint law is kept at full precision
    assert float(left["gain_db"]) == pytest.approx(-3.010299956639812, abs=1e-12)
    assert float(right["gain_db"]) == pytest.approx(-3.010299956639812, abs=1e-12)


def test_the_shared_trim_moves_both_members(pair_stems):
    left, right = _channels(pair_stems)
    plan = _plan(pair_stems, capture_groups=[_declaration(trim_db=3.0)])
    routed = _routed(plan, pair_stems, ["left", "right"])
    scale = 10.0 ** (3.0 / 20.0)
    expected = np.stack([left, right], axis=1) * scale
    assert float(np.max(np.abs(routed - expected))) < TOL
    for tid in ("left", "right"):
        assert float(_track(plan, tid)["gain_db"]) == pytest.approx(
            -3.010299956639812 + 3.0, abs=1e-12
        )


def test_an_individual_pan_override_wins_and_is_reported(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration()], pans={"left": 0.0})
    left = _track(plan, "left")
    assert float(left["pan"]) == 0.0        # the explicit pan is the last word
    receipt = plan["capture_groups"]["groups"][0]
    assert any("keeps its own pan" in n for n in receipt["notes"])
    assert any("keeps its own pan" in w for w in plan["warnings"])
    # and the evidence says so on the member itself
    assert any("explicit pan" in e for e in left["evidence"])


def test_an_individual_trim_is_reported_with_the_actual_final_number(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration()], trims={"left": 2.0})
    left = _track(plan, "left")
    right = _track(plan, "right")
    assert float(left["gain_db"]) == pytest.approx(-3.010299956639812 + 2.0, abs=1e-12)
    assert float(right["gain_db"]) == pytest.approx(-3.010299956639812, abs=1e-12)
    assert any("manual trim" in w for w in plan["warnings"])
    assert any("manual trim" in e for e in left["evidence"])


def test_the_pair_receipt_is_ordered_and_honest(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration(pan=0.5, trim_db=1.5)])
    receipt = plan["capture_groups"]
    assert receipt["mode"] == "declared_capture_group"
    group = receipt["groups"][0]
    assert group["mode"] == "stereo_pair"
    assert group["members"] == ["left", "right"]
    assert group["protected_tracks"] == ["left", "right"]
    assert group["trim_db"] == 1.5
    assert float(group["pan"]) == 0.5
    notes = " ".join(group["notes"])
    assert "left" in notes and "right" in notes
    assert "3.010299956639812" in notes        # the real compensation, not -3.01
    # the boundary says what the pair does not do
    assert "stereo pair" in receipt["boundary"].lower()


def test_the_legacy_group_is_untouched_by_the_pair_mode(pair_stems):
    plan = _plan(
        pair_stems, capture_groups=[{"id": "room", "members": ["left", "right"], "pan": -0.25}]
    )
    group = plan["capture_groups"]["groups"][0]
    assert "mode" not in group
    assert "trim_db" not in group
    assert group["members"] == ["left", "right"]
    assert group["protected_tracks"] == ["left", "right"]
    assert float(group["pan"]) == -0.25
    for tid in ("left", "right"):
        t = _track(plan, tid)
        assert float(t["pan"]) == -0.25
        assert float(t["gain_db"]) == 0.0
    assert "stereo pair" not in plan["capture_groups"]["boundary"].lower()


def test_the_default_path_is_exactly_the_old_path(pair_stems):
    plain = _plan(pair_stems)
    empty = _plan(pair_stems, capture_groups=[])
    assert empty["tracks"] == plain["tracks"]
    assert "capture_groups" not in empty


# ------------------------------------------------- the schema and the engine


def test_a_pair_plan_validates_against_the_schema_and_the_engine(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration(pan=0.5, trim_db=-2.0)])
    planner.validate(plan)          # the shipped schema + the engine invariants


def _receipt_plan(plan: dict, group: dict) -> dict:
    broken = json.loads(json.dumps(plan))
    broken["capture_groups"]["groups"] = [group]
    return broken


def test_the_engine_refuses_a_partial_pair_claim(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration()])
    group = plan["capture_groups"]["groups"][0]
    broken = [
        {**group, "protected_tracks": ["left"]},            # one member of the pair
        {**group, "protected_tracks": ["left", "left"]},     # the same member twice
        {**group, "protected_tracks": ["left", "drums"]},   # a different pair
        {**group, "protected_tracks": ["left", "ghost"]},   # a track that is not there
    ]
    for bad in broken:
        with pytest.raises(planner.PlanError):
            planner._engine_invariants(_receipt_plan(plan, bad))
        # and the whole plan is refused too, schema first
        with pytest.raises(Exception):
            planner.validate(_receipt_plan(plan, bad))


def test_the_engine_refuses_a_malformed_pair_receipt(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration()])
    group = plan["capture_groups"]["groups"][0]
    without_trim = {k: v for k, v in group.items() if k != "trim_db"}
    broken = [
        without_trim,
        {**group, "trim_db": 30.0},
        {**group, "members": ["left", "right", "drums"]},
        {**group, "members": ["left", "left"]},
        {**group, "mode": None},
    ]
    for bad in broken:
        with pytest.raises(planner.PlanError):
            planner._engine_invariants(_receipt_plan(plan, bad))
        with pytest.raises(Exception):
            planner.validate(_receipt_plan(plan, bad))


def test_the_engine_refuses_a_shared_trim_without_a_mode(pair_stems):
    """A shared trim means nothing without the pair mode, so a receipt that
    keeps the trim and drops the mode is refused -- not read as a
    same-source group with a trim nobody asked for."""
    plan = _plan(pair_stems, capture_groups=[_declaration()])
    group = plan["capture_groups"]["groups"][0]
    without_mode = {k: v for k, v in group.items() if k != "mode"}
    with pytest.raises(planner.PlanError):
        planner._engine_invariants(_receipt_plan(plan, without_mode))
    with pytest.raises(Exception):
        planner.validate(_receipt_plan(plan, without_mode))


def test_the_schema_rejects_a_pair_receipt_with_a_bogus_mode(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration()])
    group = plan["capture_groups"]["groups"][0]
    with pytest.raises(Exception):
        planner.validate(_receipt_plan(plan, {**group, "mode": "stereo"}))


def test_an_old_plan_still_validates(pair_stems):
    plan = _plan(pair_stems, capture_groups=[{"id": "room", "members": ["left", "right"]}])
    planner.validate(plan)


# --------------------------------------------------------------- the receipt text


def test_the_cli_does_not_call_a_pair_one_common_pan(pair_stems):
    plan = _plan(pair_stems, capture_groups=[_declaration(pan=0.5, trim_db=1.0)])
    text = cli._plan_summary(plan)
    assert "stereo pair" in text
    assert "common pan" not in text
    assert "left left" in text and "right right" in text
    assert "pair position" in text


def test_the_cli_still_calls_a_same_source_group_one_common_pan(pair_stems):
    plan = _plan(pair_stems, capture_groups=[{"id": "room", "members": ["left", "right"]}])
    text = cli._plan_summary(plan)
    assert "pan" in text


def test_the_written_receipt_does_not_call_a_pair_one_common_pan(pair_stems, tmp_path):
    plan, total, report = pipeline.mix(pair_stems, _preset(), capture_groups=[_declaration()])
    assert total is not None
    path = str(tmp_path / "receipt.md")
    render.write_report(path, plan, report, pair_stems)
    text = Path(path).read_text(encoding="utf-8")
    assert "stereo pair" in text
    assert "common pan" not in text.split("## Capture groups")[-1].split("## Checks")[0]


# ------------------------------------------------------ the page's own guards


def _ui_functions() -> str:
    """The page's own declaration functions, read out of the page.

    The guard that blocks a mix and the payload that is sent are the ones
    the browser runs, so the test runs them too instead of describing them.
    """
    page = Path(__file__).resolve().parents[1] / "web" / "index.html"
    html = page.read_text(encoding="utf-8")
    return html.split("function captureGroupsPayload()", 1)[1].split(
        "async function saveGroups()", 1
    )[0]


@pytest.fixture(scope="module")
def node_bin():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    return node


def _ui_state(node: str, gain_text=None, pair: bool = True, saved_trim: float = 0.0) -> dict:
    """A page with one declared group, and the shared gain the user typed."""
    parts = [
        "groupByTrack['left']='pair';groupByTrack['right']='pair';",
        "groupPans['pair']=0;",
    ]
    if pair:
        parts += [
            "groupModes['pair']='stereo_pair';",
            "groupPairLeft['pair']='left';groupPairRight['pair']='right';",
            "groupPairTrims['pair']=0;",
        ]
        if gain_text is not None:
            parts.append("groupPairGains['pair']=" + json.dumps(gain_text) + ";")
        parts.append(
            "groupSavedDecl['pair']={id:'pair',members:['left','right'],pan:0,"
            "mode:'stereo_pair',trim_db:" + json.dumps(saved_trim) + "};"
        )
    snippet = "".join(parts) + (
        "console.log(JSON.stringify({problem:groupProblemText(),"
        "payload:captureGroupsPayload()}));"
    )
    js = (
        "let groupByTrack=Object.create(null),groupPans=Object.create(null);"
        "let groupModes=Object.create(null),groupPairLeft=Object.create(null);"
        "let groupPairRight=Object.create(null),groupPairTrims=Object.create(null);"
        "let groupPairGains=Object.create(null),groupSavedDecl=Object.create(null);"
        "let stems=['left.wav','right.wav'],mutedTracks=new Set();"
        "let roleInfo={'left.wav':{role:'guitar_electric'},"
        "'right.wav':{role:'guitar_electric'}};"
        "function fileNameOf(t){return t;}"
        "function tidOf(n){return n.replace(/\\.[^.]+$/,'');}\n"
        "function captureGroupsPayload()" + _ui_functions() + "\n" + snippet
    )
    proc = subprocess.run([node, "-e", js], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_an_out_of_range_shared_gain_blocks_the_mix(node_bin):
    result = _ui_state(node_bin, "13")
    assert result["problem"]
    assert "配對增益" in result["problem"]


def test_an_out_of_range_shared_gain_keeps_the_saved_pair(node_bin):
    """The invalid entry is not a zero and not the last valid value: the
    declaration the server holds stays exactly as it was."""
    result = _ui_state(node_bin, "13", saved_trim=0.0)
    assert result["payload"] == [
        {"id": "pair", "members": ["left", "right"], "pan": 0,
         "mode": "stereo_pair", "trim_db": 0.0}
    ]


def test_an_empty_shared_gain_blocks_the_mix_and_keeps_the_saved_pair(node_bin):
    result = _ui_state(node_bin, "", saved_trim=-2.5)
    assert result["problem"]
    assert result["payload"][0]["trim_db"] == -2.5


def test_a_valid_shared_gain_is_sent_and_blocks_nothing(node_bin):
    result = _ui_state(node_bin, "3")
    assert result["problem"] == ""
    assert result["payload"][0]["trim_db"] == 3.0


def test_a_legacy_draft_is_not_blocked_by_the_pair_guard(node_bin):
    result = _ui_state(node_bin, None, pair=False)
    assert result["problem"] == ""
    assert result["payload"] == [{"id": "pair", "members": ["left", "right"], "pan": 0}]
