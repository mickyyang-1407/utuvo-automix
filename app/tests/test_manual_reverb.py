"""Manual amount changes only the added pure return after automatic decisions."""
import copy
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from automix import analyze, cli, dsp, pipeline, plan as planner, reference, render, server

SR = 8000


@pytest.fixture
def stems():
    t = np.arange(SR) / SR
    result = []
    for name, role, frequency in [('voice.wav','lead_vocal',440),('music.wav','other',1300),('fx.wav','fx',280)]:
        a = .1*np.sin(2*np.pi*frequency*t)*(1+.2*np.sin(2*np.pi*3*t))
        s = analyze.Stem(name,name,np.column_stack([a,a*.93]).astype(np.float32),SR,role=role)
        analyze.measure(s)
        result.append(s)
    return result


def planned(stems, amount=None):
    return planner.build_plan(stems,planner.load_preset('pop_2020s'),reverb_amount=amount)


def test_legacy_and_explicit_one_have_exact_audio_collector_and_report(stems,tmp_path):
    p = planned(stems)
    assert 'manual_amount' not in p['reverb']
    expected_collect = {};expected,expected_report = render.render(stems,p,collect=expected_collect)
    render.write_report(str(tmp_path/'old.md'),p,expected_report,stems)
    for amount in (None,1):
        q=copy.deepcopy(p)
        if amount is not None:q['reverb']['manual_amount']=amount
        collected={};actual,report=render.render(stems,q,collect=collected)
        np.testing.assert_array_equal(actual,expected)
        assert report==expected_report
        for tid in expected_collect['stems']:
            np.testing.assert_array_equal(collected['stems'][tid],expected_collect['stems'][tid])
        np.testing.assert_array_equal(collected['reverb'],expected_collect['reverb'])
        render.write_report(str(tmp_path/'new.md'),q,report,stems)
        assert (tmp_path/'new.md').read_bytes()==(tmp_path/'old.md').read_bytes()


def test_zero_and_fraction_only_scale_pure_return_once(stems,tmp_path):
    original = [s.audio.copy() for s in stems]
    p=planned(stems);wet={};render.render(stems,p,collect=wet)
    for amount in (0,.25,.6):
        q=copy.deepcopy(p);q['reverb']['manual_amount']=amount
        c={'stems':{'stale':np.ones(2)},'reverb':wet['reverb']};audio,r=render.render(stems,q,collect=c)
        assert set(c['stems'])==set(wet['stems']) and 'fx' in c['stems']
        for tid in wet['stems']:np.testing.assert_array_equal(c['stems'][tid],wet['stems'][tid])
        if amount:
            np.testing.assert_array_equal(c['reverb'],wet['reverb']*amount)
        else:
            assert 'reverb' not in c and r['wet_ratio_db'] is None
        render.write_stem_export(str(tmp_path/str(amount)),q,c,r,audio)
        assert (tmp_path/str(amount)/'_reverb.wav').exists()==bool(amount)
    for s,a in zip(stems,original):np.testing.assert_array_equal(s.audio,a)


def test_zero_matches_zero_convolution_oracle_with_dry_fx_intact(stems,monkeypatch):
    p=planned(stems,0);collected={};actual,_=render.render(stems,p,collect=collected)
    p['reverb'].pop('manual_amount')
    monkeypatch.setattr(dsp,'reverb_send',lambda dry,*args,**kwargs:np.zeros_like(dry))
    expected,_=render.render(stems,p)
    np.testing.assert_array_equal(actual,expected)
    assert np.any(collected['stems']['fx']) and 'reverb' not in collected


def test_manual_follows_reference_and_auto_scale_without_changing_dry(stems,tmp_path,monkeypatch):
    path=tmp_path/'ref.wav';render.write_wav24(str(path),sum(s.audio for s in stems)*.5,SR)
    ref=reference.analyse(str(path));ref['tail_ratio']=1.0
    original_tail=dsp.tail_ratio
    monkeypatch.setattr(dsp,'tail_ratio',lambda *args,**kwargs:.1)
    renders=[];original_render=render.render
    def observed(s,p,*args,**kwargs):
        renders.append(copy.deepcopy(p));return original_render(s,p,*args,**kwargs)
    monkeypatch.setattr(render,'render',observed)
    outputs=[]
    for amount in (None,1,0,.4):
        c={};p,a,r=pipeline.mix(stems,planner.load_preset('pop_2020s'),reference=ref,reverb_match=True,
                              reverb_amount=amount,collect=c,source_balance_profile='balanced_stems')
        assert 'manual_amount' not in renders[-2]['reverb']
        assert p['reverb']['scale']!=1.0
        outputs.append((p,a,r,c))
    base=outputs[0]
    for p,a,r,c in outputs[1:]:
        assert p['tracks']==base[0]['tracks'] and p['buses']==base[0]['buses']
        assert p['reverb']['scale']==base[0]['reverb']['scale']
        for tid in c['stems']:np.testing.assert_array_equal(c['stems'][tid],base[3]['stems'][tid])
        amount=p['reverb']['manual_amount']
        if amount==1:
            np.testing.assert_array_equal(a,base[1]);assert r==base[2]
        elif amount:
            np.testing.assert_array_equal(c['reverb'],base[3]['reverb']*amount)
        else:assert 'reverb' not in c
        assert r['reference_tone']['after_tone']==reference.spectrum_db(a,SR)
    monkeypatch.setattr(dsp,'tail_ratio',original_tail)


@pytest.mark.parametrize('value',[True,False,float('nan'),float('inf'),float('-inf'),-.01,1.01,'0.5',{},[]])
def test_invalid_values_fail_before_planning_or_rendering(stems,monkeypatch,value,schema_free):
    p=planned(stems);p['reverb']['manual_amount']=value
    def forbidden(*args,**kwargs):raise AssertionError('invalid input reached expensive work')
    monkeypatch.setattr(planner.balancemod,'resolve',forbidden)
    monkeypatch.setattr(render,'_apply_comp',forbidden)
    with pytest.raises(planner.PlanError):planner.build_plan(stems,planner.load_preset('pop_2020s'),reverb_amount=value)
    with pytest.raises(planner.PlanError):pipeline.mix(stems,planner.load_preset('pop_2020s'),reverb_amount=value)
    with pytest.raises((planner.PlanError,jsonschema.ValidationError)):planner.validate(p)
    with pytest.raises(planner.PlanError):planner.validate(p,schema_free)
    with pytest.raises(render.RenderError):render.render(stems,p)


def test_null_plan_amount_is_invalid(stems,schema_free):
    p=planned(stems);p['reverb']['manual_amount']=None
    with pytest.raises(jsonschema.ValidationError):planner.validate(p)
    with pytest.raises(planner.PlanError):planner.validate(p,schema_free)
    with pytest.raises(render.RenderError):render.render(stems,p)


@pytest.mark.parametrize('value',['nan','inf','-inf','1.1','-.1','true','garbage'])
def test_cli_rejects_invalid_before_loading(value,monkeypatch):
    monkeypatch.setattr(analyze,'load_stems',lambda *args:pytest.fail('must reject before loading'))
    with pytest.raises(SystemExit) as exc:cli.main(['--stems','unused','--reverb-amount',value])
    assert exc.value.code==2


def test_cli_plan_only_and_saved_plan_replay(stems,tmp_path):
    source=tmp_path/'stems';source.mkdir()
    for s in stems:render.write_wav24(str(source/s.name),s.audio,SR)
    out=tmp_path/'plan'
    assert cli.main(['--stems',str(source),'--out',str(out),'--plan-only','--reverb-amount','0','--quiet'])==0
    p=json.loads((out/'mixplan.json').read_text());assert p['reverb']['manual_amount']==0
    assert cli.main(['--render-plan',str(out/'mixplan.json'),'--out',str(tmp_path/'render'),'--quiet'])==0
    assert (tmp_path/'render/mix.wav').exists()


@pytest.mark.parametrize('stored',[None,True,'0.4',float('nan'),float('inf'),-.1,1.1])
def test_bad_stored_amount_does_not_break_session_load(tmp_path,stored):
    s=server.Session(str(tmp_path));s.save();path=Path(s.dir)/s.MANIFEST
    data=json.loads(path.read_text());data['reverb_amount']=stored;path.write_text(json.dumps(data))
    restored=server.Session.load(str(tmp_path),s.id)
    assert restored is not None and restored.reverb_amount==1


def test_session_amount_roundtrip_and_legacy_default(tmp_path):
    s=server.Session(str(tmp_path));s.reverb_amount=.37;s.save()
    assert server.Session.load(str(tmp_path),s.id).reverb_amount==.37
    path=Path(s.dir)/s.MANIFEST;data=json.loads(path.read_text());data.pop('reverb_amount');path.write_text(json.dumps(data))
    assert server.Session.load(str(tmp_path),s.id).reverb_amount==1


def test_standalone_schema_accepts_optional_amount_and_rejects_bad_values(stems):
    jsonschema=pytest.importorskip('jsonschema')
    schema=json.loads((Path(__file__).resolve().parents[2]/'schemas/mix_plan.schema.json').read_text())
    for amount in (None,0,.37,1):
        jsonschema.validate(planned(stems,amount),schema)
    for amount in (True,False,None,'0.4',-.1,1.1):
        p=planned(stems);p['reverb']['manual_amount']=amount
        with pytest.raises(jsonschema.ValidationError):jsonschema.validate(p,schema)


def test_cli_export_directory_keeps_old_files_but_zero_manifest_omits_return(stems,tmp_path):
    p=planned(stems);collected={};audio,r=render.render(stems,p,collect=collected)
    render.write_stem_export(str(tmp_path),p,collected,r,audio)
    previous=(tmp_path/'_reverb.wav').read_bytes()
    (tmp_path/'user.wav').write_bytes(b'user-owned file')
    p['reverb']['manual_amount']=0
    audio,r=render.render(stems,p,collect=collected)
    render.write_stem_export(str(tmp_path),p,collected,r,audio)
    assert 'reverb' not in collected
    assert all(f['filename']!='_reverb.wav' for f in r['export_summary']['files'])
    assert (tmp_path/'_reverb.wav').read_bytes()==previous
    assert (tmp_path/'user.wav').read_bytes()==b'user-owned file'


@pytest.mark.parametrize('container', ['missing',None,{},[],False,0,0.0,''])
def test_legacy_empty_reverb_containers_keep_normalized_audio_and_receipt(stems,tmp_path,container):
    p=planned(stems);p['reverb']={}
    expected,expected_report=render.render(stems,p)
    render.write_report(str(tmp_path/'expected.md'),p,expected_report,stems)
    if container=='missing':p.pop('reverb')
    else:p['reverb']=container
    planner.validate(p)
    actual,report=render.render(stems,p)
    render.write_report(str(tmp_path/'actual.md'),p,report,stems)
    np.testing.assert_array_equal(actual,expected)
    assert report==expected_report
    assert (tmp_path/'actual.md').read_bytes()==(tmp_path/'expected.md').read_bytes()


@pytest.mark.parametrize('container',[True,42,'not a mapping',['not a pair']])
def test_legacy_schema_container_range_is_not_narrowed(stems,container):
    jsonschema=pytest.importorskip('jsonschema')
    schema=json.loads((Path(__file__).resolve().parents[2]/'schemas/mix_plan.schema.json').read_text())
    p=planned(stems);p['reverb']=container
    planner.validate(p);jsonschema.validate(p,schema)
    # These containers were schema-accepted but not renderable in the old engine.
    with pytest.raises((TypeError,ValueError)):render.render(stems,p)


def test_cli_legacy_container_replay_and_explicit_override(stems,tmp_path):
    source=tmp_path/'stems';source.mkdir()
    for s in stems:render.write_wav24(str(source/s.name),s.audio,SR)
    p=planned(stems);p['source']['stems_dir']=str(source)
    for i,container in enumerate((None,{},[],False,0,'')):
        p['reverb']=container;path=tmp_path/f'legacy-{i}.json'
        path.write_text(json.dumps(p))
        base=tmp_path/f'base-{i}';override=tmp_path/f'override-{i}'
        assert cli.main(['--render-plan',str(path),'--out',str(base),'--quiet'])==0
        assert json.loads((base/'mixplan.json').read_text())['reverb']==container
        assert cli.main(['--render-plan',str(path),'--out',str(override),'--reverb-amount','1','--quiet'])==0
        assert json.loads((override/'mixplan.json').read_text())['reverb']=={'manual_amount':1}
        assert (base/'mix.wav').read_bytes()==(override/'mix.wav').read_bytes()
