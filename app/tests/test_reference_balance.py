"""Physical controls for gain identification; no docs/runtime fixture imports."""
import numpy as np
import pytest

from automix import analyze, reference_balance as rb

SR = 2000


@pytest.fixture
def material():
    rng = np.random.default_rng(43102)
    n = SR * 24
    x = rng.standard_normal((n, 2, 4)) * [.07, .04, .06, .03]
    return x, rng


def evaluate(x, y, sr=SR):
    stems = [analyze.Stem(f'track{i}.wav', '', x[:, :, i], sr) for i in range(x.shape[2])]
    before = [s.audio.copy() for s in stems]
    result = rb.evaluate(stems, y, sr)
    assert all(np.array_equal(s.audio, old) for s,old in zip(stems,before))
    return result


def test_identifies_known_gain_errors_and_ignores_common_reference_loudness(material):
    base, _ = material
    errors = np.array([12, -12, 6, 0])
    x = base * 10 ** (errors/20)
    a = evaluate(x, base.sum(axis=2)*.7)
    b = evaluate(x, base.sum(axis=2)*7)
    assert a['passed'] and b['passed']
    expected = -errors - ((-errors).min()+(-errors).max())/2
    np.testing.assert_allclose(list(a['relative_gain_db'].values()), expected, atol=1e-8)
    np.testing.assert_allclose(list(a['relative_gain_db'].values()), list(b['relative_gain_db'].values()), atol=1e-8)
    assert b['common_gain_db'] - a['common_gain_db'] == pytest.approx(20)
    assert a['common_gain_applied'] is False
    assert a['residual'] < 1e-7


@pytest.mark.parametrize('case', ['duplicate','near_duplicate','polarity','asymmetry','unrelated','offset','time_varying','local_hidden','outside_bound','weak_biased','weak_coherent'])
def test_rejects_unidentifiable_or_non_gain_models(material, case):
    base,rng = material; x=base.copy(); y=x.sum(axis=2)
    held = (np.arange(len(x)) // (2*SR)) % 2 == 1
    if case == 'duplicate':
        x[:,:,1]=x[:,:,0];y=x.sum(axis=2)
    elif case == 'near_duplicate':
        x[:,:,1]=x[:,:,0]+base[:,:,1]*1e-4;y=x.sum(axis=2)+rng.standard_normal(y.shape)*.0002
    elif case == 'polarity': y=np.einsum('nci,i->nc',x,[1,-1,1,1])
    elif case == 'asymmetry': y[:,0]+=.5*x[:,0,0];y[:,1]-=.5*x[:,1,0]
    elif case == 'unrelated': y=rng.standard_normal(y.shape)*.1
    elif case == 'offset': y=np.roll(y,137,axis=0)
    elif case == 'time_varying': y[held]+=2*x[held,:,0]
    elif case == 'local_hidden':
        quiet=(np.arange(len(x))//(2*SR))==7;x[quiet]*=.05;y=x.sum(axis=2);y[quiet]+=2*x[quiet,:,0]
    elif case == 'outside_bound': x[:,:,3]*=.001;y=x.sum(axis=2)+99*x[:,:,3]
    elif case in ('weak_biased','weak_coherent'):
        x[:,:,3]*=.01;y=x.sum(axis=2)+.5*x[:,:,3]+rng.standard_normal(y.shape)*(.0002 if case=='weak_biased' else .00001)
    result=evaluate(x,y)
    assert not result['passed'],result
    assert 'relative_gain_db' not in result  # no partial/clipped correction


def test_digital_zero_excluded_but_a_quiet_source_is_not_silenced(material):
    x,_=material;x[:,:,3]=0
    result=evaluate(x,x.sum(axis=2))
    assert result['passed'] and result['excluded_ids']==['track3']
    x[:,:,:3]*=1e-5
    result=evaluate(x,x.sum(axis=2))
    assert result['passed'] and len(result['active_ids'])==3


def test_mono_16k_short_duration_and_track_cap():
    sr=16000;t=np.arange(sr*8)/sr
    x=np.sin(2*np.pi*430*t)*.05
    stem=analyze.Stem('mono.wav','',x[:,None],sr)
    r=rb.evaluate([stem],x,sr)
    assert r['passed'] and r['relative_gain_db']=={'mono':0}
    assert rb.evaluate([stem],x[:-sr],sr)['reason']=='duration_mismatch'
    assert rb.evaluate([analyze.Stem('x','',x[:sr],sr)],x[:sr],sr)['reason']=='shorter_than_minimum_coverage'
    stems=[analyze.Stem(f's{i}.wav','',x[:,None],sr) for i in range(33)]
    assert rb.evaluate(stems,x,sr)['reason']=='more_than_32_nonzero_stems'


@pytest.fixture
def session(tmp_path):
    from automix import render
    sr=8000;n=sr*24;rng=np.random.default_rng(6029)
    base=rng.standard_normal((n,2,4))*[.015,.01,.012,.008]
    base[:,1,2]=base[:,0,2]  # actual mono lead loader path
    errors=np.array([12,-12,6,0])
    source=base*10**(errors/20)
    folder=tmp_path/'stems';folder.mkdir()
    names=['drums.wav','bass.wav','vocal_lead.wav','other.wav']
    for i,name in enumerate(names):
        x=source[:,:,i]
        render.write_wav24(str(folder/name),x[:,:1] if i==2 else x,sr)
    render.write_wav24(str(folder/'silent.wav'),np.zeros((n,2)),sr)
    stems=analyze.load_stems(str(folder));analyze.detect_roles(stems);analyze.align(stems)
    for s in stems: analyze.measure(s)
    ref=tmp_path/'rough.wav';render.write_wav24(str(ref),base.sum(axis=2)*.7,sr)
    return stems,{'path':str(ref)},folder,dict(zip([analyze.track_id(n) for n in names],-errors))


def test_pipeline_corrected_prebus_manual_trim_and_metadata_replay(session, monkeypatch):
    import copy
    from automix import dsp,pipeline,plan,render
    stems,ref,_,expected=session;preset=plan.load_preset('pop_2020s')
    collect={}
    p,a,report=pipeline.mix(stems,preset,source_balance_profile='reference_balance',reference=ref,blend=0,collect=collect)
    decision=p['style']['source_balance'];assert decision['origin']=='reference_fit'
    assert decision['proof']['excluded_ids']==['silent']
    plan.validate(p)
    for tr in p['tracks']:
        assert tr['gain_db']==pytest.approx(expected[tr['id']],abs=.01)
        m=tr['level_match'];raw=next(s for s in stems if analyze.track_id(s.name)==tr['id'])
        assert m['source_lufs']==raw.metrics['lufs']
        got=render._window_lufs(collect['stems'][tr['id']],raw.sr,m['start_frame'],m['end_frame'])
        assert got==pytest.approx(m['source_lufs']+expected[tr['id']],abs=.05)
    more={}
    p2,_,r2=pipeline.mix(stems,preset,source_balance_profile='reference_balance',reference=ref,blend=0,
                          trims={'vocal_lead':3},pans={'vocal_lead':.4},collect=more)
    v=next(t for t in p2['tracks'] if t['id']=='vocal_lead')
    assert v['gain_db']-expected['vocal_lead']==pytest.approx(3,abs=.01)
    m=v['level_match'];out=more['stems']['vocal_lead']
    assert render._window_lufs(out,stems[0].sr,m['start_frame'],m['end_frame'])==pytest.approx(m['source_lufs']+expected['vocal_lead']+3,abs=.05)
    assert np.sum(out[:,1]**2)>np.sum(out[:,0]**2)
    stripped=copy.deepcopy(p);del stripped['style']['source_balance']
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('renderer cannot fit')))
    replay,_=render.render(stems,stripped)
    assert np.array_equal(a,replay)


def test_two_pass_fits_once_and_uses_stable_file_ids(session, monkeypatch):
    from automix import pipeline,plan,reference
    stems,ref,_,_=session;ref=reference.analyse(ref['path'])
    original=rb.evaluate;calls=[]
    def counted(*args): calls.append(1);return original(*args)
    monkeypatch.setattr(rb,'evaluate',counted)
    p,_,r=pipeline.mix(stems,plan.load_preset('pop_2020s'),reference=ref,blend=.2,source_balance_profile='reference_balance')
    assert calls==[1] and r['reference_passes']==2
    assert p['style']['source_balance']['origin']=='reference_fit'
    assert set(p['style']['source_balance']['proof']['relative_gain_db'])=={t['id'] for t in p['tracks']}


@pytest.mark.parametrize('k',[0,.5,1])
def test_explicit_k_skips_reference_fit_and_matches_legacy_tracks(session,monkeypatch,k):
    from automix import plan
    stems,ref,_,_=session;preset=plan.load_preset('pop_2020s')
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('explicit k cannot fit')))
    p=plan.build_plan(stems,preset,reference=ref,blend=0,source_balance_profile='reference_balance',balance_preserve=k)
    old=plan.build_plan(stems,preset,balance_preserve=k)
    assert p['tracks']==old['tracks']
    assert p['style']['source_balance']['origin']=='explicit'
    assert 'relative_gain_db' not in p['style']['source_balance']['proof']


def test_scalar_noaction_missing_reference_and_no_stale_filename(session,tmp_path,monkeypatch):
    from automix import render,source_balance
    stems,ref,_,_=session
    missing=source_balance.resolve(stems,profile='reference_balance')
    assert missing['balance_preserve']==.5 and missing['proof']['reason']=='no_reference_audio'
    path=tmp_path/'same.wav';sr=stems[0].sr
    render.write_wav24(str(path),sum(s.audio for s in stems)*.1,sr)
    original=rb.evaluate
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('scalar proof must skip richer fit')))
    scalar=source_balance.resolve(stems,{'path':str(path)},'reference_balance')
    assert scalar['origin']=='proof' and scalar['balance_preserve']==1
    monkeypatch.setattr(rb,'evaluate',original)
    good=source_balance.resolve(stems,ref,'reference_balance')
    assert good['origin']=='reference_fit'
    y=analyze.load_stem_file(ref['path'])[0]
    render.write_wav24(ref['path'],np.roll(y,117,axis=0),sr)
    bad=source_balance.resolve(stems,ref,'reference_balance')
    assert bad['balance_preserve']==.5 and bad['resolved']=='unknown'
    assert 'relative_gain_db' not in bad['proof']


def test_fit_obstruction_by_manual_stage_limit_is_explicit(session):
    from automix import plan
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,
                       source_balance_profile='reference_balance',trims={'bass':12})
    assert next(t for t in p['tracks'] if t['id']=='bass')['gain_db']==18
    assert any('reference balance target' in s and 'not reached' in s for s in p['warnings'])


@pytest.mark.parametrize('change',['unknown_id','nan','wrong_limits','negative_coef','wrong_relative','common_applied','missing_fold','weak_check','rejected_corrections','sample_rate','duration'])
def test_invalid_fit_metadata_rejected_without_optional_schema(session,change,schema_free):
    import copy,jsonschema
    from automix import plan
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    proof=p['style']['source_balance']['proof'];first=next(iter(proof['relative_gain_db']))
    if change=='unknown_id': proof['active_ids'][0]='missing'
    elif change=='nan': proof['folds'][0]['aggregate_residual']=float('nan')
    elif change=='wrong_limits': proof['limits']['max_aggregate_residual']=.9
    elif change=='negative_coef': proof['raw_coefficients'][first]=-1
    elif change=='wrong_relative': proof['relative_gain_db'][first]+=1
    elif change=='common_applied': proof['common_gain_applied']=True
    elif change=='missing_fold': proof['folds'].pop()
    elif change=='weak_check': proof['folds'][0]['columns'][0]['unique_over_reference']=.001
    elif change=='rejected_corrections': proof['passed']=False
    elif change=='sample_rate': proof['source_sample_rate']+=1
    elif change=='duration': proof['source_duration_s']+=1
    with pytest.raises((plan.PlanError,jsonschema.ValidationError)): plan.validate(p)
    with pytest.raises(plan.PlanError): plan.validate(p,schema_free)


def test_cli_plan_only_run_and_saved_replay(session,tmp_path,monkeypatch):
    import json
    from automix import cli
    _,ref,folder,_=session
    args=['--stems',str(folder),'--reference',ref['path'],'--blend','0',
          '--source-balance-profile','reference_balance','--quiet']
    out=tmp_path/'plan'
    assert cli.main(args+['--plan-only','--out',str(out)])==0
    p=json.loads((out/'mixplan.json').read_text())
    assert p['style']['source_balance']['origin']=='reference_fit'
    actual=tmp_path/'actual'
    assert cli.main(args+['--out',str(actual)])==0
    assert 'common fit gain' in (actual/'report.md').read_text()
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('saved plan must not refit')))
    replay=tmp_path/'replay'
    assert cli.main(['--render-plan',str(actual/'mixplan.json'),'--out',str(replay),'--quiet'])==0
    assert (actual/'mix.wav').read_bytes()==(replay/'mix.wav').read_bytes()


def test_existing_bv_guard_and_render_match_cap_never_claim_fit_target(session):
    from automix import plan,render
    stems,ref,_,_=session
    next(s for s in stems if analyze.track_id(s.name)=='other').role='backing_vocal'
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    assert p['style']['source_balance']['origin']=='reference_fit'
    assert any('reference balance target' in w and 'not reached' in w for w in p['warnings'])
    track=next(t for t in p['tracks'] if t['id']=='drums')
    track['eq']=[{'type':'low_pass','freq':100,'gain_db':0,'q':.707,'enabled':True}]
    _,report=render.render(stems,p)
    assert any('post-track loudness correction capped' in w for w in report['warnings'])


def test_reference_resample_treatment_is_recorded(material):
    from automix import dsp
    x,_=material;errors=10**(np.array([6,-6,0,0])/20)
    stems=[analyze.Stem(f'track{i}.wav','',x[:,:,i]*errors[i],SR) for i in range(4)]
    y=dsp.resample_fft(x.sum(axis=2),SR,SR*2)
    p=rb.evaluate(stems,y,SR*2)
    assert p['passed'] and p['sample_rate_treatment']=='fft_resample_reference'
    assert p['source_sample_rate']==SR and p['reference_sample_rate']==SR*2


def test_resampling_metadata_allows_one_destination_frame_rounding(material):
    from automix import dsp
    x,_=material
    stems=[analyze.Stem(f'track{i}.wav','',x[:,:,i],SR) for i in range(4)]
    y=dsp.resample_fft(x.sum(axis=2),SR,SR*8)[:-1]
    p=rb.evaluate(stems,y,SR*8)
    assert p['passed']
    rb.validate_receipt(p,{s.name[:-4] for s in stems},SR,len(x))


def test_actual_pcm_pooled_fit_must_agree_with_each_temporal_fold(tmp_path):
    from automix import render,source_balance
    sr=2000;n=sr*24;rng=np.random.default_rng(9173);rho=.999
    original=np.zeros((n,2,3));reference=np.zeros((n,2))
    for block,start in enumerate(range(0,n,2*sr)):
        count=min(2*sr,n-start)
        q,_=np.linalg.qr(rng.standard_normal((count*2,3)))
        q*=np.sqrt(count*2)*.05
        amplitudes=[1,1.04] if block%2==0 else [1.04,1]
        x=np.column_stack([amplitudes[0]*q[:,0],amplitudes[1]*(rho*q[:,0]+np.sqrt(1-rho*rho)*q[:,1]),q[:,2]]).reshape(count,2,3)
        original[start:start+count]=x
        reference[start:start+count]=x.sum(axis=2)*(.998 if block%2==0 else 1.002)
    folder=tmp_path/'pooled_stems';folder.mkdir()
    raw=original*np.array([1,1,4])
    for index in range(3): render.write_wav24(str(folder/f'mic_{index}.wav'),raw[:,:,index],sr)
    ref=tmp_path/'rough.wav';render.write_wav24(str(ref),reference,sr)
    stems=analyze.load_stems(str(folder));analyze.align(stems)
    decision=source_balance.resolve(stems,{'path':str(ref)},profile='reference_balance')
    assert decision['origin']=='default'
    assert decision['balance_preserve']==.5
    assert decision['proof']['reason']=='pooled_coefficients_unstable'
    assert decision['proof']['full_vs_fold_gain_spread_db']>.25
    assert 'relative_gain_db' not in decision['proof']


def test_v3_pooled_summary_is_required_and_bounded(session,schema_free):
    import copy,jsonschema
    from automix import plan
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    proof=p['style']['source_balance']['proof']
    assert proof['method']=='positive_stereo_time_split_v3'
    assert proof['limits']['max_full_vs_fold_relative_db']==.25
    assert proof['full_vs_fold_gain_spread_db']<=.25
    for value in (None,.251,float('nan')):
        bad=copy.deepcopy(p)
        if value is None: bad['style']['source_balance']['proof'].pop('full_vs_fold_gain_spread_db')
        else: bad['style']['source_balance']['proof']['full_vs_fold_gain_spread_db']=value
        with pytest.raises((plan.PlanError,jsonschema.ValidationError)): plan.validate(bad)
        with pytest.raises(plan.PlanError): plan.validate(bad,schema_free)


def test_legal_v2_saved_recipe_validates_and_replays_wav_exact(session,tmp_path,monkeypatch):
    """This stable fixture has the same coefficients under v2 and v3.

    Preserve the historical v2 receipt shape and limits, without requesting a
    v2 analysis path. Neither receipt version may influence numeric replay.
    """
    import copy,json
    from automix import cli,plan,render,source_balance
    stems,ref,folder,_=session
    current=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    historical=copy.deepcopy(current)
    proof=historical['style']['source_balance']['proof']
    proof['method']='positive_stereo_time_split_v2'
    proof['limits']=dict(rb.V2_LIMITS)
    proof.pop('full_vs_fold_gain_spread_db')
    historical['source']['stems_dir']=str(folder)
    plan.validate(historical)
    loaded=json.loads(json.dumps(historical))
    plan.validate(loaded)
    recipe=tmp_path/'v2.json';recipe.write_text(json.dumps(loaded))
    def forbidden(*a,**kw): raise AssertionError('historical numeric replay must not infer')
    monkeypatch.setattr(rb,'evaluate',forbidden)
    monkeypatch.setattr(source_balance,'resolve',forbidden)
    a,_=render.render(stems,current)
    b,_=render.render(stems,loaded)
    assert np.array_equal(a,b)
    expected=tmp_path/'expected.wav';render.write_wav24(str(expected),a,stems[0].sr)
    out=tmp_path/'old_replay'
    assert cli.main(['--render-plan',str(recipe),'--out',str(out),'--quiet'])==0
    assert expected.read_bytes()==(out/'mix.wav').read_bytes()
    saved=json.loads((out/'mixplan.json').read_text())
    assert saved['style']['source_balance']['proof']['method']=='positive_stereo_time_split_v2'


@pytest.mark.parametrize('passed',[True,False])
def test_receipt_context_rejects_fabricated_excluded_input(material,passed):
    import copy
    x,rng=material
    y=x.sum(axis=2) if passed else rng.standard_normal(x.shape[:2])*.1
    proof=evaluate(x,y)
    assert proof['passed'] is passed
    known=set(proof['input_ids'])
    rb.validate_receipt(proof,known)
    fake=copy.deepcopy(proof)
    fake['input_ids'].append('fabricated_input')
    fake.setdefault('excluded_ids',[]).append('fabricated_input')
    with pytest.raises(ValueError,match='known source graph'):
        rb.validate_receipt(fake,known)
    # No external source graph means structural-only validation by contract.
    rb.validate_receipt(fake,None)


@pytest.mark.parametrize('version',['v2','v3'])
def test_standalone_schema_limits_are_exact_like_portable(material,version):
    import copy,json
    from pathlib import Path
    jsonschema=pytest.importorskip('jsonschema')
    x,_=material
    proof=evaluate(x*10**(np.array([6,-6,0,0])/20),x.sum(axis=2))
    assert proof['passed']
    if version=='v2':
        proof['method']=rb.V2_LIMITS['method'];proof['limits']=dict(rb.V2_LIMITS)
        proof.pop('full_vs_fold_gain_spread_db')
    schema=json.loads((Path(__file__).resolve().parents[2]/'schemas/mix_plan.schema.json').read_text())
    definition='reference_balance_fit' if version=='v2' else 'reference_balance_fit_v3'
    fragment={'$ref':'#/$defs/'+definition,'$defs':schema['$defs']}
    jsonschema.validate(proof,fragment)
    rb.validate_receipt(proof,set(proof['input_ids']))
    bad=copy.deepcopy(proof);bad['limits']['max_aggregate_residual']=.9
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(bad,fragment)
    with pytest.raises(ValueError,match='settings'): rb.validate_receipt(bad)


@pytest.mark.parametrize('version',['v2','v3'])
@pytest.mark.parametrize('variant',['known_gains','common_scaled','zero_excluded','quiet_nonzero','mono','resampled'])
def test_twelve_valid_versioned_receipts_schema_and_portable(material,version,variant):
    import json
    from pathlib import Path
    from automix import dsp
    jsonschema=pytest.importorskip('jsonschema')
    x,_=material;x=x.copy()
    if variant=='zero_excluded': x[:,:,3]=0
    if variant=='quiet_nonzero': x*=1e-5
    errors=10**(np.array([6,-6,0,0])/20)
    stems=[analyze.Stem(f'track{i}.wav','',x[:,:,i]*errors[i],SR) for i in range(4)]
    y=x.sum(axis=2)*.7;ref_sr=SR
    if variant=='common_scaled': y*=10
    elif variant=='mono':
        mono=x[:,0,0]
        stems=[analyze.Stem('mono.wav','',mono[:,None],SR)]
        y=mono*.7
    elif variant=='resampled':
        ref_sr=SR*2;y=dsp.resample_fft(y,SR,ref_sr)
    proof=rb.evaluate(stems,y,ref_sr)
    assert proof['passed'],proof
    if version=='v2':
        proof['method']=rb.V2_LIMITS['method'];proof['limits']=dict(rb.V2_LIMITS)
        proof.pop('full_vs_fold_gain_spread_db')
    restored=json.loads(json.dumps(proof,allow_nan=False))
    schema=json.loads((Path(__file__).resolve().parents[2]/'schemas/mix_plan.schema.json').read_text())
    definition='reference_balance_fit' if version=='v2' else 'reference_balance_fit_v3'
    jsonschema.validate(restored,{'$ref':'#/$defs/'+definition,'$defs':schema['$defs']})
    rb.validate_receipt(restored,{analyze.track_id(s.name) for s in stems},SR,len(x))


def test_plan_source_graph_comes_from_actual_silent_inputs(session):
    import copy
    from automix import plan
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    assert p['source']['excluded_stems']==[{'id':'silent','file':'silent.wav','reason':'silent'}]
    assert 'silent' not in {s['id'] for s in p['source']['stems']}
    plan.validate(p)
    for mutation in ('extra','omitted'):
        bad=copy.deepcopy(p);proof=bad['style']['source_balance']['proof']
        if mutation=='extra':
            proof['input_ids'].append('bogus');proof['excluded_ids'].append('bogus')
        else:
            proof['input_ids'].remove('silent');proof['excluded_ids'].remove('silent')
        with pytest.raises(plan.PlanError,match='known source graph'): plan.validate(bad)


def test_failed_fit_records_quiet_dropped_source_independently(session):
    from automix import plan,render
    stems,ref,_,_=session;sr=stems[0].sr
    quiet=analyze.Stem('quiet.wav','',np.random.default_rng(204).standard_normal(stems[0].audio.shape)*1e-8,sr)
    analyze.measure(quiet);assert quiet.metrics['silent']
    stems=stems+[quiet]
    y=analyze.load_stem_file(ref['path'])[0]
    render.write_wav24(ref['path'],np.roll(y,113,axis=0),sr)
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    proof=p['style']['source_balance']['proof']
    assert not proof['passed'] and p['style']['source_balance']['balance_preserve']==.5
    assert 'quiet' in proof['active_ids'] and 'quiet' not in proof['excluded_ids']
    assert 'quiet' in {s['id'] for s in p['source']['excluded_stems']}
    assert 'quiet' not in {s['id'] for s in p['source']['stems']}
    plan.validate(p)


@pytest.mark.parametrize('version',['v2','v3'])
def test_legacy_silent_warning_bridge_and_missing_evidence_replay(session,version,monkeypatch):
    import copy
    from automix import plan,render,source_balance
    stems,ref,_,_=session
    current=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    legacy=copy.deepcopy(current);legacy['source'].pop('excluded_stems')
    if version=='v2':
        proof=legacy['style']['source_balance']['proof'];proof['method']=rb.V2_LIMITS['method'];proof['limits']=dict(rb.V2_LIMITS);proof.pop('full_vs_fold_gain_spread_db')
    plan.validate(legacy)
    missing=copy.deepcopy(legacy);missing['warnings']=[w for w in missing['warnings'] if not w.startswith('dropped ')]
    with pytest.raises(plan.PlanError,match='remove style.source_balance'): plan.validate(missing)
    # Numeric recipe survives even when the optional legacy evidence cannot be verified.
    missing['style'].pop('source_balance');plan.validate(missing)
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('replay cannot infer')))
    monkeypatch.setattr(source_balance,'resolve',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('replay cannot resolve')))
    a,_=render.render(stems,current);b,_=render.render(stems,missing)
    assert np.array_equal(a,b)


@pytest.mark.parametrize('many,comma',[(False,True),(True,False),(True,True)])
def test_legacy_warning_filename_split_is_unambiguous(session,many,comma):
    import copy
    from automix import plan
    stems,ref,folder,_=session
    if comma:
        silent=next(s for s in stems if s.name=='silent.wav')
        renamed=folder/'silent, ending.wav'
        from pathlib import Path
        Path(silent.path).rename(renamed);silent.name=renamed.name;silent.path=str(renamed)
    if many:
        extra=analyze.Stem('tail.wav','',np.zeros_like(stems[0].audio),stems[0].sr);analyze.measure(extra);stems=stems+[extra]
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    plan.validate(p)  # explicit graph always has real, unambiguous input records
    legacy=copy.deepcopy(p);legacy['source'].pop('excluded_stems')
    if many and comma:
        with pytest.raises(plan.PlanError,match='ambiguous'): plan.validate(legacy)
        legacy['style'].pop('source_balance');plan.validate(legacy)
    else:
        plan.validate(legacy)  # count1 treats the entire comma-containing filename as one


@pytest.mark.parametrize('mutation',['id_file','extra_field','missing_reason','wrong_reason','duplicate','overlap'])
def test_excluded_source_graph_rejects_inconsistent_records(session,mutation,schema_free):
    import jsonschema
    from automix import plan
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset('pop_2020s'),reference=ref,blend=0,source_balance_profile='reference_balance')
    records=p['source']['excluded_stems']
    if mutation=='id_file': records[0]['id']='invented'
    elif mutation=='extra_field': records[0]['untrusted']=True
    elif mutation=='missing_reason': records[0].pop('reason')
    elif mutation=='wrong_reason': records[0]['reason']='maybe quiet'
    elif mutation=='duplicate': records.append(dict(records[0]))
    elif mutation=='overlap': records[0]={'id':p['source']['stems'][0]['id'],'file':p['source']['stems'][0]['file'],'reason':'silent'}
    with pytest.raises((plan.PlanError,jsonschema.ValidationError)): plan.validate(p)
    with pytest.raises(plan.PlanError): plan.validate(p,schema_free)


@pytest.mark.parametrize('version',['v2','v3'])
@pytest.mark.parametrize('style',['acoustic','cinematic','lofi','pop_2020s','rock','trap'])
def test_twelve_legal_plans_schema_and_portable_source_graph(session,version,style,monkeypatch,schema_free):
    import json,sys
    from pathlib import Path
    from automix import plan
    jsonschema=pytest.importorskip('jsonschema')
    stems,ref,_,_=session
    p=plan.build_plan(stems,plan.load_preset(style),reference=ref,blend=0,source_balance_profile='reference_balance')
    if version=='v2':
        proof=p['style']['source_balance']['proof'];proof['method']=rb.V2_LIMITS['method'];proof['limits']=dict(rb.V2_LIMITS);proof.pop('full_vs_fold_gain_spread_db')
        p['source'].pop('excluded_stems')  # original builder shape; use actual dropped warning
    schema=json.loads((Path(__file__).resolve().parents[2]/'schemas/mix_plan.schema.json').read_text())
    jsonschema.validate(p,schema)
    plan.validate(p)
    monkeypatch.setattr(rb,'evaluate',lambda *a: (_ for _ in ()).throw(AssertionError('validation cannot infer')))
    plan.validate(p)  # the receipt is checked from the plan alone, never by re-analysis
    plan.validate(p,schema_free)  # the engine's portable source-graph check accepts it by itself
    monkeypatch.setitem(sys.modules,'jsonschema',None)
    with pytest.raises(plan.PlanError,match='validator is not installed'): plan.validate(p)  # a missing validator fails closed
