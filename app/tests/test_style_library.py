"""The style library declares executable starting points, never artist recipes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from automix import analyze, cli, dsp, pipeline, plan, server

NEW = {'rnb_soul', 'jazz_natural', 'funk_disco', 'house_dance', 'heavy_rock_metal',
       'reggae_groove', 'ambient_space', 'indie_pop'}
LEGACY = {'acoustic', 'cinematic', 'lofi', 'pop_2020s', 'rock', 'trap'}
NUMERIC_KEYS = ['version', 'target_lufs', 'true_peak_db', 'tonal_tilt_db', 'crest_target_db',
                'width', 'reverb', 'compression', 'saturation', 'vocal_forward_db',
                'mono_below_hz', 'role_gains_db', 'guards']
# Frozen r2 numerical contract; descriptive/provenance metadata may be appended.
OLD_HASHES = {
    "acoustic": "d38798195da6b2dd2c1579975b74dae8892b05092e84506acbcabe5e3372027c",
    "cinematic": "8748b1161db3aca7df6cb691acd597d3e2a447bbfaac9fb6b43a383d52b23ab4",
    "lofi": "a4078224a41603eb348e1f9b7bb8d11fd8fe8f37487b7d0b2311096e37fb2338",
    "pop_2020s": "d142cd5b599c6f897d0588bc57eeba3ed011ea877cb45617e3444bab644f8e5f",
    "rock": "820c60e43947f71555114671821cef494f3e28f49b31655a751a17a2f92bebfc",
    "trap": "eb8edee6cf5e75ca9f01e140b3e5a81fb716305fd199de2ffdb14431a3ee1bcd"
}


def test_exact_fourteen_ids_shared_cli_server(capsys):
    assert set(cli._styles(cli._default_preset_dir())) == NEW | LEGACY
    rows = server._styles()
    assert len(rows) == 14 and {row['id'] for row in rows} == NEW | LEGACY
    assert cli.main(['--list-styles']) == 0
    output = capsys.readouterr().out
    for row in rows:
        assert row['id'] in output and row['description']
        if row['id'] in NEW:
            assert row['intent']['caveat'] in output
            assert row['intent']['numeric_basis'] in output


def test_existing_six_numerical_states_unchanged():
    for id, wanted in OLD_HASHES.items():
        p = plan.load_preset(id)
        numbers = {key: p[key] for key in NUMERIC_KEYS}
        got = hashlib.sha256(json.dumps(numbers, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert got == wanted, id


@pytest.fixture
def stems():
    sr = 24000; n = sr * 2
    t = np.arange(n) / sr
    rng = np.random.default_rng(23)
    noise = .035 * rng.standard_normal((n, 2))
    voice = .09 * np.sin(2 * np.pi * 730 * t)[:, None] * np.ones((1, 2))
    bass = .09 * np.sin(2 * np.pi * 110 * t)[:, None] * np.ones((1, 2))
    material = [analyze.Stem('other.wav', '', noise, sr, role='other'),
                analyze.Stem('vocal_lead.wav', '', voice, sr, role='lead_vocal'),
                analyze.Stem('bass.wav', '', bass, sr, role='bass')]
    for s in material: analyze.measure(s)
    return material


@pytest.mark.parametrize('id', sorted(NEW | LEGACY))
def test_every_preset_is_finite_guarded_and_builds_valid_plan(id, stems):
    p = plan.load_preset(id)
    assert set(p['role_gains_db']) == set(analyze.ROLES)
    assert set(p['compression']) == {'drum', 'bass', 'vocal', 'music', 'bus'}
    assert all(0 <= amount <= 1 for amount in p['compression'].values())
    assert all(0 <= amount <= p['guards']['max_saturation'] for amount in p['saturation'].values())
    assert all(-18 <= gain <= 18 and np.isfinite(gain) for gain in p['role_gains_db'].values())
    assert p['width'] <= p['guards']['max_width']
    assert p['true_peak_db'] <= -1
    assert all(abs(gain) <= p['guards']['max_eq_db'] for gain in p['tonal_tilt_db'].values())
    built = plan.build_plan(stems, p, source_balance_profile='balanced_stems', balance_preserve=.5)
    plan.validate(built)
    assert built['style']['balance_preserve'] == .5  # no style overrides user intent
    if id in NEW:
        assert p['intent']['numeric_basis'].startswith('UTUVO自己')
        assert built['style']['intent'] == p['intent']
        assert built['style']['intent'] is not p['intent']
        assert all(p['intent'][key] for key in ('applicability', 'caveat', 'numeric_basis'))
        assert -8 <= p['role_gains_db']['other'] <= -3


def test_heavy_dense_sources_keep_the_existing_crest_guard(stems):
    p = plan.load_preset('heavy_rock_metal')
    assert p['guards']['min_crest_db'] == 8
    built = plan.build_plan(stems, p)
    # Pure sustained tones have low crest and must not receive the declared
    # compressor simply because the preset's label suggests aggression.
    dense = [t for t in built['tracks'] if t['role'] in {'lead_vocal', 'bass'}]
    assert dense and all('comp' not in t for t in dense)
    assert any('compression skipped' in note and '8.0' in note for note in built['warnings'])


def test_new_intents_execute_differences_at_one_common_loudness(stems):
    outcomes = {}
    for id in ('jazz_natural', 'house_dance', 'ambient_space'):
        collect = {}
        p, audio, report = pipeline.mix(stems, plan.load_preset(id), target_lufs=-18,
                                        source_balance_profile='balanced_stems', collect=collect)
        assert abs(dsp.loudness_lufs(audio, stems[0].sr) + 18) < .05
        assert dsp.true_peak_db(audio, stems[0].sr) <= -1 + .02
        assert np.isfinite(audio).all()
        outcomes[id] = (audio, report)
    jazz = outcomes['jazz_natural'][0]
    for id in ('house_dance', 'ambient_space'):
        other = outcomes[id][0]
        fitted = np.sum(jazz * other) / np.sum(other * other)
        residual = np.sqrt(np.sum((jazz - fitted * other) ** 2) / np.sum(jazz ** 2))
        assert residual > .05  # a scalar loudness difference cannot explain it
    assert outcomes['ambient_space'][1]['wet_ratio_db'] > outcomes['jazz_natural'][1]['wet_ratio_db']


def test_bad_provenance_is_rejected_without_optional_schema(stems, schema_free):
    import copy
    import jsonschema
    p = plan.build_plan(stems, plan.load_preset('jazz_natural'))
    broken = copy.deepcopy(p)
    broken['style']['intent']['source_refs'] = [{'title': 'fake recipe'}]
    with pytest.raises(jsonschema.ValidationError):
        plan.validate(broken)
    with pytest.raises(plan.PlanError, match='source reference'):
        plan.validate(broken, schema_free)


def test_style_intent_metadata_never_changes_saved_plan_audio(stems):
    import copy
    from automix import render
    p = plan.build_plan(stems, plan.load_preset('indie_pop'), source_balance_profile='balanced_stems')
    without = copy.deepcopy(p)
    del without['style']['intent']
    a, _ = render.render(stems, p)
    b, _ = render.render(stems, without)
    assert np.array_equal(a, b)


def test_no_style_names_or_links_to_real_people():
    """The styles are UTUVO's own starting points; no preset cites anyone."""
    for id in cli._styles(cli._default_preset_dir()):
        preset = plan.load_preset(id)
        assert (preset.get('intent') or {}).get('source_refs', []) == []
        assert 'http' not in json.dumps(preset)


def test_a_restored_old_result_no_longer_carries_names():
    old = {'style': {'id': 'rnb_soul', 'intent': {'applicability': 'a', 'caveat': 'c', 'numeric_basis': 'n',
           'source_refs': [{'title': 'Someone', 'url': 'https://x', 'principle': 'p', 'access': 'a'}]}}, 'lufs': -12}
    restored = server._without_source_refs(old)
    assert restored['style']['intent']['source_refs'] == [] and restored['lufs'] == -12
    assert old['style']['intent']['source_refs']  # the saved copy is not rewritten
    assert server._without_source_refs({'style': {'id': 'x'}}) == {'style': {'id': 'x'}}
    assert server._without_source_refs({'style': 'pop_2020s'}) == {'style': 'pop_2020s'}
    assert server._without_source_refs([1]) == [1]
