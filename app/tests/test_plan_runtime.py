"""The plan validator has to be a gate, not a formality.

The questions here are all about what happens when the validator *cannot* do
its job: no jsonschema installed, no schema file, a corrupt schema, an
invalid schema, a NaN in the plan, a container of the wrong shape, a ``$ref``
that would need the network.  Every one of them has to end in a clear
refusal -- and a refusal must not blame the plan when the fault is the
installation.

The tests are self-contained: they build their own stems, their own plan and
their own fixtures.  No demo_stems, no App, no server, no real recording.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import wave

import numpy as np
import pytest

from automix import analyze, pipeline, plan as planner

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
SR = 48000
SCHEMA = os.path.join(os.path.dirname(HERE), "schemas", "mix_plan.schema.json")


def _tone(name: str, dur: float = 2.0, freq: float = 220.0):
    """A short synthetic tone -- ours, not borrowed from anywhere."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = 0.2 * np.sin(2 * np.pi * freq * t)
    st = analyze.Stem(
        name=name,
        path=name,
        audio=np.stack([x, x], axis=1).astype(np.float32),
        sr=SR,
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def _stems():
    stems = [_tone("vocal_lead.wav", 2.0, 220.0), _tone("guitar_elec.wav", 2.0, 110.0)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def a_plan():
    """A plan the engine accepts, built from our own two tones."""
    return planner.build_plan(_stems(), planner.load_preset("pop_2020s"))


def _write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# --- the validator itself ------------------------------------------------


def test_a_valid_plan_passes(a_plan):
    planner.validate(a_plan)


def test_the_default_schema_is_the_canonical_one():
    path = planner.default_schema_path()
    assert os.path.basename(path) == "mix_plan.schema.json"
    assert os.path.basename(os.path.dirname(path)) == "schemas"
    assert os.path.isfile(path)


def test_a_missing_validator_is_a_hard_stop_even_for_a_valid_plan(a_plan, monkeypatch):
    """No jsonschema is a broken installation, not a bad plan -- and it is
    never a silent pass either."""
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan)
    message = str(excinfo.value)
    assert "jsonschema" in message
    assert "not installed" in message


def test_a_missing_schema_is_a_hard_stop(a_plan, tmp_path):
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan, schema_path=str(tmp_path / "not-there.json"))
    assert "cannot be read" in str(excinfo.value)


def test_a_corrupt_schema_is_a_hard_stop(a_plan, tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text('{"type": "object"', encoding="utf-8")
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan, schema_path=str(path))
    assert "not valid JSON" in str(excinfo.value)


def test_a_schema_that_is_not_utf8_is_a_clear_refusal(a_plan, tmp_path):
    """A schema file that is not text is a broken resource, not a traceback."""
    path = tmp_path / "binary.schema.json"
    path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan, schema_path=str(path))
    message = str(excinfo.value)
    assert "not UTF-8 text" in message
    assert not isinstance(excinfo.value, UnicodeDecodeError)


def test_an_invalid_schema_is_a_hard_stop(a_plan, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"type": 12}', encoding="utf-8")
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan, schema_path=str(path))
    assert "not a valid JSON Schema" in str(excinfo.value)


def test_a_plan_that_is_not_an_object_is_refused():
    with pytest.raises(planner.PlanError):
        planner.validate(["not", "a", "plan"])


def test_a_nested_nan_is_refused_with_its_place(a_plan):
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["level_match"]["source_lufs"] = float("nan")
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(broken)
    message = str(excinfo.value)
    assert "non-finite" in message
    assert "$.tracks[0].level_match.source_lufs" in message


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_infinity_is_refused_inside_a_nested_array(a_plan, bad):
    broken = copy.deepcopy(a_plan)
    broken["style"]["notes"] = [1.0, bad]
    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(broken)
    message = str(excinfo.value)
    assert "non-finite" in message
    assert "$.style.notes[1]" in message


def test_a_huge_integer_is_finite_and_must_not_overflow(a_plan):
    """An integer with hundreds of digits is a number, not an overflow: the
    finite guard must not turn it into a crash."""
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["gain_db"] = 10 ** 400
    with pytest.raises(Exception) as excinfo:
        planner.validate(broken)
    assert not isinstance(excinfo.value, OverflowError)
    assert "non-finite" not in str(excinfo.value)


def test_a_container_of_the_wrong_shape_is_refused_without_a_traceback(a_plan):
    broken = copy.deepcopy(a_plan)
    broken["style"] = "not an object"
    with pytest.raises(Exception) as excinfo:
        planner.validate(broken)
    assert not isinstance(excinfo.value, (TypeError, KeyError, AttributeError))
    assert hasattr(excinfo.value, "json_path")


def test_an_out_of_range_value_is_refused_with_its_path(a_plan):
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["pan"] = 5.0
    with pytest.raises(Exception) as excinfo:
        planner.validate(broken)
    assert not isinstance(excinfo.value, (TypeError, KeyError, AttributeError))
    assert getattr(excinfo.value, "json_path", "") == "$.tracks[0].pan"


def test_validation_does_not_change_the_plan(a_plan):
    before = json.dumps(a_plan, sort_keys=True, default=str)
    planner.validate(a_plan)
    assert json.dumps(a_plan, sort_keys=True, default=str) == before


def test_a_refused_plan_is_not_changed_either(a_plan):
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["pan"] = 5.0
    before = json.dumps(broken, sort_keys=True, default=str)
    with pytest.raises(Exception):
        planner.validate(broken)
    assert json.dumps(broken, sort_keys=True, default=str) == before


def test_a_local_internal_ref_works(a_plan, tmp_path):
    """A schema that refers to itself is resolved from the schema, offline."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://utuvo.example/schemas/local-test.schema.json",
        "type": "object",
        "required": ["schema_version"],
        "properties": {"schema_version": {"$ref": "#/$defs/version"}},
        "$defs": {"version": {"type": "integer", "const": 1}},
    }
    path = tmp_path / "local.schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    planner.validate(a_plan, schema_path=str(path))


def test_a_remote_ref_is_refused_offline_without_touching_the_network(a_plan, tmp_path, monkeypatch):
    """A reference that is not inside the schema cannot be resolved, and the
    validator does not go out to the network to resolve it."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://utuvo.example/schemas/remote-test.schema.json",
        "type": "object",
        "properties": {"schema_version": {"$ref": "https://example.invalid/other.json#/v"}},
    }
    path = tmp_path / "remote.schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("the validator must never reach the network")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    with pytest.raises(planner.PlanError) as excinfo:
        planner.validate(a_plan, schema_path=str(path))
    message = str(excinfo.value)
    assert "cannot be resolved offline" in message
    assert "never fetches one over the network" in message
    # the network was never touched: the socket patch did not fire
    assert "AssertionError" not in message


def test_schema_data_that_looks_like_a_reference_is_not_a_reference(a_plan, tmp_path):
    """A literal ``$ref`` inside ``examples`` or ``const`` is data, not an
    active schema keyword: the schema is legal and must not be refused."""
    literal = {"$ref": "https://example.invalid/this-is-data"}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://utuvo.example/schemas/data-test.schema.json",
        "type": "object",
        "required": ["schema_version"],
        "properties": {
            "schema_version": {"type": "integer"},
            # an optional property whose const is a literal $ref object
            "note": {"const": literal},
        },
        "examples": [
            {"schema_version": 1, "note": literal},
        ],
    }
    path = tmp_path / "data.schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    planner.validate(a_plan, schema_path=str(path))


# --- the pipeline: every pass is checked before its renderer -------------


def test_each_pipeline_pass_is_checked_before_its_render(monkeypatch, tmp_path):
    """Two passes, two checks, and a check before each render."""
    from automix import reference as refmod

    calls = []
    real_render = pipeline.render.render
    real_validate = planner.validate

    def spy_render(stems, plan, *args, **kwargs):
        calls.append("render")
        return real_render(stems, plan, *args, **kwargs)

    def spy_validate(plan, schema_path=None):
        calls.append("validate")
        return real_validate(plan, schema_path)

    monkeypatch.setattr(pipeline.render, "render", spy_render)
    monkeypatch.setattr(planner, "validate", spy_validate)

    # a reference, so the pipeline really runs two passes
    n = int(12 * SR)
    t = np.arange(n) / SR
    x = np.concatenate([
        0.05 * np.sin(2 * np.pi * 220 * t[: n // 3]),
        0.2 * np.sin(2 * np.pi * 220 * t[: n // 3]),
        0.05 * np.sin(2 * np.pi * 220 * t[n - 2 * (n // 3):]),
    ])
    ref_path = tmp_path / "ref.wav"
    _write_wav(ref_path, np.stack([x, x], axis=1))
    ref = refmod.analyse(str(ref_path))

    pipeline.mix(_stems(), planner.load_preset("pop_2020s"), reference=ref, blend=0.7)

    assert calls == ["validate", "render", "validate", "render"]


def test_an_invalid_plan_never_reaches_the_renderer(monkeypatch):
    """A plan that does not satisfy the schema must not produce audio."""
    calls = []
    real_build = planner.build_plan

    def broken_build(*args, **kwargs):
        built = real_build(*args, **kwargs)
        built["tracks"][0]["pan"] = 5.0
        return built

    def spy_render(*args, **kwargs):
        calls.append("render")
        raise AssertionError("the renderer must not be reached")

    monkeypatch.setattr(planner, "build_plan", broken_build)
    monkeypatch.setattr(pipeline.render, "render", spy_render)

    with pytest.raises(Exception):
        pipeline.mix(_stems(), planner.load_preset("pop_2020s"))
    assert calls == []


# --- the App's own entry point -------------------------------------------


def _run_app(*args):
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "app.py"), *args],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_the_app_accepts_a_valid_plan_without_rendering_it(a_plan, tmp_path):
    path = tmp_path / "mixplan.json"
    planner.write_plan(a_plan, str(path))
    proc = _run_app("--validate-plan", str(path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "plan is valid" in proc.stdout
    assert "Traceback" not in proc.stderr
    # nothing was written next to the plan
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mixplan.json"]


def test_the_app_refuses_an_out_of_range_plan(a_plan, tmp_path):
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["pan"] = 5.0
    path = tmp_path / "broken.json"
    planner.write_plan(broken, str(path))
    proc = _run_app("--validate-plan", str(path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "pan" in proc.stderr
    # a plan that really breaks the schema is still called invalid
    assert "not a valid mix plan" in proc.stderr


def test_the_app_refuses_a_plan_with_a_nan(a_plan, tmp_path):
    broken = copy.deepcopy(a_plan)
    broken["tracks"][0]["gain_db"] = float("nan")
    path = tmp_path / "nan.json"
    # NaN is not JSON, but Python's parser accepts it -- that is the point
    path.write_text(json.dumps(broken), encoding="utf-8")
    proc = _run_app("--validate-plan", str(path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "non-finite" in proc.stderr


def test_the_app_refuses_a_file_that_is_not_a_plan(tmp_path):
    path = tmp_path / "not-a-plan.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    proc = _run_app("--validate-plan", str(path))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_the_app_rejects_unknown_flags():
    proc = _run_app("--not-a-flag")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "unknown argument" in proc.stderr


def test_the_app_does_not_call_a_valid_plan_invalid_when_the_validator_is_missing(a_plan, tmp_path):
    """A machine without jsonschema is a broken installation, not a bad plan:
    the reason has to survive, and the plan must not be called invalid."""
    shadow = tmp_path / "site-packages"
    shadow.mkdir()
    (shadow / "jsonschema.py").write_text(
        'raise ImportError("No module named \'jsonschema\'")\n', encoding="utf-8"
    )
    plan_path = tmp_path / "mixplan.json"
    planner.write_plan(a_plan, str(plan_path))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shadow)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "app.py"), "--validate-plan", str(plan_path)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    # the concrete installation reason is kept as it is ...
    assert "not installed" in proc.stderr
    assert "jsonschema" in proc.stderr
    # ... and the plan itself is not called invalid
    assert "invalid plan" not in proc.stderr
    assert "not a valid mix plan" not in proc.stderr
