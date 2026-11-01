"""Physical provenance tests: a spectral resemblance is never source intent."""
import copy
import json
import wave

import numpy as np
import pytest

from automix import analyze, cli, dsp, pipeline, plan, reference, render, source_balance

SR = 8000


def write(path, x, sr=SR):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(x.shape[1]); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(x, -.99, .99) * 32767).astype('<i2').tobytes())
    return str(path)


@pytest.fixture
def material(tmp_path):
    folder = tmp_path / 'stems'; folder.mkdir()
    rng = np.random.default_rng(410)
    t = np.arange(9 * SR) / SR
    # Nonstationary stereo sources: an offset must not pass a periodic oracle.
    vocal = .08 * np.sin(2 * np.pi * 730 * t) * (.2 + .8 * (np.sin(t * 3) ** 2))
    backing = .05 * rng.standard_normal((len(t), 2))
    write(folder / 'vocal_lead.wav', vocal)  # genuine mono loader path
    write(folder / 'other.wav', backing)
    write(folder / 'bass.wav', np.zeros((len(t), 2)))  # dropped role must not zip-mismatch
    stems = analyze.load_stems(str(folder)); analyze.detect_roles(stems); analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    summed = sum(s.audio for s in stems)
    path = str(tmp_path / 'reference.wav')
    render.write_wav24(path, summed * .8, SR)
    return stems, {'path': path}, summed, folder


def test_aligned_positive_and_original_inputs_unchanged(material):
    stems, ref, _, _ = material
    before = [s.audio.copy() for s in stems]; ref_before = dict(ref)
    p = source_balance.proof(stems, ref)
    assert p['passed'] and p['active_coverage_s'] == 9
    assert p['relative_rms_residual'] < .001 and p['correlation'] > .99999
    assert p['gain'] == pytest.approx(.8, abs=.0002)
    assert ref == ref_before
    for s, old in zip(stems, before):
        assert np.array_equal(s.audio, old)


@pytest.mark.parametrize('kind', ['unrelated', 'offset', 'short', 'silent', 'inverted', 'raw_gain_accident', 'cancelling'])
def test_negatives_never_auto_preserve(material, tmp_path, kind):
    stems, ref, summed, _ = material
    stems = copy.deepcopy(stems)
    x = summed.copy()
    if kind == 'unrelated':
        x = np.random.default_rng(2).normal(0, .1, x.shape)
    elif kind == 'offset':
        x = np.concatenate([np.zeros((137, 2)), x[:-137]])
    elif kind == 'short':
        x = x[:7 * SR]
    elif kind == 'silent':
        x[:] = 0
    elif kind == 'inverted':
        x *= -1
    elif kind == 'raw_gain_accident':
        for s in stems:
            if s.role == 'other': s.audio *= dsp.db_to_lin(12)
            elif s.role == 'lead_vocal': s.audio *= dsp.db_to_lin(-12)
    elif kind == 'cancelling':
        stems = [analyze.Stem('a', '', summed, SR), analyze.Stem('b', '', -summed, SR)]
    ref = {'path': write(tmp_path / 'negative.wav', x)}
    b = source_balance.resolve(stems, ref)
    assert not b['proof']['passed'], b
    assert b['resolved'] == 'unknown' and b['balance_preserve'] == .5
    assert b['proof']['reason']


def test_short_active_coverage_is_not_long_silence(material, tmp_path):
    stems, _, summed, _ = material
    for s in stems:
        s.audio[:2 * SR] = 0
    summed = sum(s.audio for s in stems)
    b = source_balance.resolve(stems, {'path': write(tmp_path / 'sparse.wav', summed)})
    assert b['proof']['reason'] == 'insufficient_active_coverage'
    assert b['proof']['active_coverage_s'] == 7
    assert b['balance_preserve'] == .5


@pytest.mark.parametrize('profile', source_balance.PROFILES)
@pytest.mark.parametrize('k', [0, .5, 1])
def test_explicit_always_wins_even_zero(material, profile, k, monkeypatch):
    def forbidden(*args):
        raise AssertionError('explicit k must not need a proof')
    monkeypatch.setattr(source_balance, 'proof', forbidden)
    b = source_balance.resolve(material[0], material[1], profile, k)
    assert b['balance_preserve'] == k and b['origin'] == 'explicit'
    assert b['requested_balance_preserve'] == k


@pytest.mark.parametrize('k', [float('nan'), float('inf'), -1, 2, True, '0.5'])
def test_malformed_k_rejected(k):
    with pytest.raises(ValueError, match='finite number'):
        source_balance.resolve([], balance_preserve=k)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='source_balance_profile'):
        source_balance.resolve([], profile='four_stems')


def test_rate_treatment_and_mono_power(material, tmp_path):
    stems, _, summed, _ = material
    rate = 16000
    up = dsp.resample_fft(summed, SR, rate)
    p = source_balance.proof(stems, {'path': write(tmp_path / 'upsampled.wav', up, rate)})
    assert p['passed'] and p['sample_rate_treatment'] == 'fft_resample_reference'
    mono = summed.mean(axis=1)
    one = [analyze.Stem('mono', '', mono[:, None], SR)]
    assert source_balance.proof(one, {'path': write(tmp_path / 'mono.wav', mono)})['passed']


def test_no_stale_reference_filename_cache(material):
    stems, ref, summed, _ = material
    assert source_balance.resolve(stems, ref)['balance_preserve'] == 1
    write(ref['path'], np.roll(summed, 193, axis=0))
    assert source_balance.resolve(stems, ref)['balance_preserve'] == .5


def test_two_pass_decision_once_and_prebus_relative_levels(material, monkeypatch):
    stems, ref, _, _ = material
    ref = reference.analyse(ref['path'])
    original = source_balance.proof; calls = []
    def counted(*a):
        calls.append(1)
        return original(*a)
    monkeypatch.setattr(source_balance, 'proof', counted)
    collected = {}
    p, audio, report = pipeline.mix(stems, plan.load_preset('pop_2020s'), reference=ref,
                                    blend=.3, collect=collected)
    assert len(calls) == 1 and report['reference_passes'] == 2
    b = p['style']['source_balance']
    assert b['balance_preserve'] == 1 and b['origin'] == 'proof'
    assert report['source_balance'] == b
    assert len(p['tracks']) == 2  # silence drop preserved proof alignment
    errors = []
    for tr in p['tracks']:
        m = tr['level_match']; x = collected['stems'][tr['id']]
        # Collector holds post-track/pre-bus audio on the SAME planned window.
        got = dsp.loudness_lufs(x[m['start_frame']:m['end_frame']], SR)
        errors.append(got - m['source_lufs'])
    assert max(errors) - min(errors) < .05
    plan.validate(p)
    # Metadata never influences replay DSP. This is also the legacy-plan path.
    legacy = copy.deepcopy(p); del legacy['style']['source_balance']
    replay, legacy_report = render.render(stems, legacy)
    assert 'source_balance' not in legacy_report
    assert np.array_equal(audio, replay)


def test_declared_profile_raw_default_and_manual_controls_once(material):
    stems, _, _, _ = material; preset = plan.load_preset('pop_2020s')
    raw = plan.build_plan(stems, preset, source_balance_profile='raw_tracks')
    explicit = plan.build_plan(stems, preset, balance_preserve=.5)
    assert raw['tracks'] == explicit['tracks']
    balanced = plan.build_plan(stems, preset, source_balance_profile='balanced_stems')
    assert balanced['style']['balance_preserve'] == 1
    target = balanced['tracks'][0]['id']
    trimmed = plan.build_plan(stems, preset, source_balance_profile='balanced_stems',
                              trims={target: 3}, pans={target: -.3})
    assert trimmed['tracks'][0]['gain_db'] - balanced['tracks'][0]['gain_db'] == pytest.approx(3)
    assert trimmed['tracks'][0]['pan'] == -.3
    broken = copy.deepcopy(balanced); broken['style']['source_balance']['balance_preserve'] = float('nan')
    with pytest.raises(plan.PlanError): plan.validate(broken)


def test_cli_plan_only_resolves_identically(material, tmp_path):
    stems, ref, _, folder = material
    out = tmp_path / 'out'
    assert cli.main(['--stems', str(folder), '--reference', ref['path'], '--blend', '0',
                     '--plan-only', '--out', str(out), '--quiet']) == 0
    p = json.loads((out / 'mixplan.json').read_text())
    assert p['style']['source_balance']['origin'] == 'proof'
    assert p['style']['balance_preserve'] == 1
    assert not (out / 'mix.wav').exists()
    override = tmp_path / 'explicit'
    assert cli.main(['--stems', str(folder), '--source-balance-profile', 'balanced_stems',
                     '--balance-preserve', '0', '--plan-only', '--out', str(override), '--quiet']) == 0
    assert json.loads((override / 'mixplan.json').read_text())['style']['balance_preserve'] == 0
    with pytest.raises(SystemExit):
        cli.main(['--balance-preserve', 'nan'])


def test_small_lossy_residual_passes_but_duration_cannot_be_cropped(material, tmp_path):
    stems, _, summed, _ = material
    rng = np.random.default_rng(20)
    noise = rng.standard_normal(summed.shape)
    noise *= .03 * np.sqrt(np.mean(summed ** 2) / np.mean(noise ** 2))
    p = source_balance.proof(stems, {'path': write(tmp_path / 'lossy.wav', summed + noise)})
    assert p['passed'] and .025 < p['relative_rms_residual'] < .035
    short = summed[:-int(.06 * SR)]
    p = source_balance.proof(stems, {'path': write(tmp_path / 'cropped.wav', short)})
    assert not p['passed'] and p['reason'] == 'duration_mismatch'


def test_cli_run_and_saved_plan_never_reinfer(material, tmp_path, monkeypatch):
    _, ref, _, folder = material
    out = tmp_path / 'run'
    assert cli.main(['--stems', str(folder), '--reference', ref['path'], '--blend', '0',
                     '--out', str(out), '--quiet']) == 0
    p = json.loads((out / 'mixplan.json').read_text())
    assert p['style']['source_balance']['origin'] == 'proof'
    assert 'coherent_same_content_reconstruction' in (out / 'report.md').read_text()
    def forbidden(*a, **kw):
        raise AssertionError('saved-plan replay must not resolve input intent')
    monkeypatch.setattr(source_balance, 'resolve', forbidden)
    replay = tmp_path / 'replay'
    assert cli.main(['--render-plan', str(out / 'mixplan.json'), '--out', str(replay), '--quiet']) == 0
    assert (out / 'mix.wav').read_bytes() == (replay / 'mix.wav').read_bytes()


@pytest.mark.parametrize('requested,effective', [(5, 1), (-3, 0), (0, 0), (.5, .5), (1, 1)])
def test_legacy_python_apis_clamp_finite_values_without_changing_valid_audio(material, requested, effective):
    stems, _, _, _ = material
    preset = plan.load_preset('pop_2020s')
    built = plan.build_plan(stems, preset, balance_preserve=requested)
    assert built['style']['balance_preserve'] == effective
    assert built['style']['source_balance']['requested_balance_preserve'] == effective
    p, audio, report = pipeline.mix(stems, preset, balance_preserve=requested)
    endpoint, expected, _ = pipeline.mix(stems, preset, balance_preserve=effective)
    assert p['tracks'] == endpoint['tracks']
    assert np.array_equal(audio, expected)
    assert p['style']['balance_preserve'] == effective
    assert report['source_balance']['balance_preserve'] == effective
    assert report['source_balance']['origin'] == 'explicit'
    plan.validate(p)


@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), -float('inf'), '0.5'])
def test_legacy_python_api_compatibility_never_clamps_invalid_types(material, value):
    stems = material[0]; preset = plan.load_preset('pop_2020s')
    for api in (plan.build_plan, pipeline.mix):
        with pytest.raises(plan.PlanError, match='finite number'):
            api(stems, preset, balance_preserve=value)


@pytest.mark.parametrize('value', ['5', '-3', 'nan', 'inf'])
def test_cli_retains_strict_explicit_k_despite_python_api_compatibility(value):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--balance-preserve', value])
    assert exc.value.code == 2
