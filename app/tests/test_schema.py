"""Schema conformance for every shape a plan can take.

`planner.validate` checks the invariants the engine relies on; this checks the
*contract* other consumers rely on (the committed JSON Schema).  The two are not
the same thing -- a field can be sane to the planner and still illegal to the
schema, which is exactly what happened with the hum notch's Q of 20.

The test is skipped when jsonschema is not installed; run it with
``uv run --with jsonschema python3 -m pytest tests/test_schema.py`` to include it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from automix import analyze, plan as planner

jsonschema = pytest.importorskip("jsonschema")

SR = 48000
SCHEMA = os.path.join(os.path.dirname(__file__), "..", "..", "schemas", "mix_plan.schema.json")


def _schema():
    with open(SCHEMA, encoding="utf-8") as fh:
        return json.load(fh)


def _tone(name, parts, dur=4.0, noise_bursts=None, hum=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in parts)
    if noise_bursts:
        rng = np.random.default_rng(0)
        for start in np.arange(0.4, dur - 0.4, 0.8):
            i = int(start * SR)
            d = int(0.12 * SR)
            x[i : i + d] += rng.standard_normal(d) * np.hanning(d) * noise_bursts
    if hum:
        x = x + hum * np.sin(2 * np.pi * 50.0 * t)
    st = analyze.Stem(
        name=name, path=name, audio=np.stack([x, x], axis=1).astype(np.float32), sr=SR
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


@pytest.fixture(scope="module")
def rich_plan():
    """A plan that exercises de-ess, hum notch, masking, trims and rides."""
    voice = _tone(
        "vocal_lead.wav",
        [(220, 0.3), (660, 0.1), (1320, 0.05), (2640, 0.03)],
        noise_bursts=0.8,
    )
    guitar = _tone(
        "guitar_elec.wav",
        [(110, 0.35), (220, 0.2), (660, 0.1), (1320, 0.06), (3300, 0.12)],
        hum=0.01,
    )
    stems = [voice, guitar]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    return planner.build_plan(
        stems, planner.load_preset("pop_2020s"), trims={"guitar_elec": -1.5}
    )


def test_the_engine_accepts_it(rich_plan):
    planner.validate(rich_plan)


def test_the_schema_accepts_it(rich_plan):
    jsonschema.validate(rich_plan, _schema())


def test_it_actually_contains_the_features(rich_plan):
    """Guard against the schema test passing because nothing was in the plan."""
    assert any(t.get("deess") for t in rich_plan["tracks"])
    assert any(b.get("purpose") == "hum_notch" for t in rich_plan["tracks"] for b in t.get("eq", []))
    assert any(t.get("trim_db") for t in rich_plan["tracks"])
    assert any(t.get("ride") for t in rich_plan["tracks"])


def test_a_reference_plan_validates(tmp_path):
    from automix import reference as refmod

    n = int(12 * SR)
    t = np.arange(n) / SR
    x = np.concatenate([
        0.05 * np.sin(2 * np.pi * 220 * t[: n // 3]),
        0.2 * np.sin(2 * np.pi * 220 * t[: n // 3]),
        0.05 * np.sin(2 * np.pi * 220 * t[: n - 2 * (n // 3)]),
    ])
    import struct
    import wave

    path = tmp_path / "ref.wav"
    pcm = (np.stack([x, x], 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    ref = refmod.analyse(str(path))
    stems = [_tone("pad.wav", [(220, 0.2), (440, 0.1)])]
    plan = planner.build_plan(
        stems, planner.load_preset("pop_2020s"), reference=ref, blend=0.7
    )
    jsonschema.validate(plan, _schema())
    assert plan["style"]["reference"]["mode"] == "global"
