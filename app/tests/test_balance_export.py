"""Regression oracles for preserved channel balance and actual DAW exports."""
import copy
import json
import jsonschema
import numpy as np
import pytest
from automix import analyze, dsp, plan as planner, render, server

SR = 24000


def tone(freq=440, amp=.2, seconds=1.0):
    t=np.arange(int(SR*seconds))/SR
    return np.column_stack([amp*np.sin(2*np.pi*freq*t)]*2)


@pytest.fixture
def sources():
    a=analyze.Stem('vocal.wav','vocal.wav',tone(440,.3),SR,role='lead_vocal')
    b=analyze.Stem('drums.wav','drums.wav',tone(800,.1)*(0.6+0.4*np.sin(2*np.pi*4*np.arange(SR)/SR))[:,None],SR,role='drums_full')
    for s in (a,b):analyze.measure(s)
    return [a,b]


def render_tracks(sources,p):
    c={};_,r=render.render(sources,p,collect=c)
    return c['stems'],r


def test_old_plan_without_block_never_enters_new_stage(sources, monkeypatch):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'),balance_preserve=0)
    assert all(t['level_match']['amount']==1 for t in p['tracks'])
    for t in p['tracks']:t.pop('level_match')
    def forbidden(*args):raise AssertionError('old plans must not use level matching')
    monkeypatch.setattr(render,'_level_match',forbidden)
    a,_=render_tracks(sources,p);b,_=render_tracks(sources,copy.deepcopy(p))
    for tid in a:np.testing.assert_array_equal(a[tid],b[tid])


def test_balance_k_interpolates_target_once(sources):
    targets=[]
    for k in (0,.5,1):
        p=planner.build_plan(sources,planner.load_preset('pop_2020s'),balance_preserve=k)
        assert all(t['level_match']['amount']==1 for t in p['tracks'])
        vals=[t['level_match']['source_lufs']+t['gain_db'] for t in p['tracks']]
        targets.append(vals[0]-vals[1])
        out,r=render_tracks(sources,p)
        for t,rep in zip(p['tracks'],r['tracks']):
            assert not rep['level_match']['capped']
            assert rep['level_match']['after_lufs']==pytest.approx(rep['level_match']['target_lufs'],abs=.03)
    assert targets[1]==pytest.approx((targets[0]+targets[2])/2,abs=.015)


def test_k1_matches_source_lufs_after_dynamics_and_eq(sources):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'),balance_preserve=1)
    # Force distinct tonal/dynamic changes that used to undo the fader balance.
    p['tracks'][0]['eq']=[{'type':'bell','freq':440,'gain_db':-4,'q':1,'enabled':True}]
    out,report=render_tracks(sources,p)
    for t,rep in zip(p['tracks'],report['tracks']):
        m=t['level_match'];got=render._window_lufs(out[t['id']],SR,m['start_frame'],m['end_frame'])
        assert abs(got-(m['source_lufs']+t['gain_db']))<.03
        assert not rep['level_match']['capped']
        assert any(x['step']=='loudness match' for x in rep['chain'])


def test_midpoint_rule_is_continuous_and_uses_frozen_window():
    x=tone()
    target=render._window_lufs(x,SR,0,len(x))+4
    m={'mode':'active_lufs_blend','source_lufs':target,'max_gain_db':6,'start_frame':0,'end_frame':len(x)}
    gains=[]
    for k in (0,.25,.5,.75,1):
        _,r=render._level_match(x,SR,dict(m,amount=k),len(x));gains.append(r['applied_db'])
    np.testing.assert_allclose(gains,[0,1,2,3,4],atol=.001)


def test_manual_trim_and_pan_survive_matching(sources):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'),balance_preserve=1,
                         trims={'vocal':2},pans={'vocal':.7})
    t=next(t for t in p['tracks'] if t['id']=='vocal')
    assert t['level_match']['source_lufs']+t['gain_db']==pytest.approx(sources[0].metrics['lufs']+2,abs=.001)
    out,_=render_tracks(sources,p)
    left=np.sqrt(np.mean(out['vocal'][:,0]**2));right=np.sqrt(np.mean(out['vocal'][:,1]**2))
    assert right>left*3
    assert abs(render._window_lufs(out['vocal'],SR,t['level_match']['start_frame'],t['level_match']['end_frame'])-(t['level_match']['source_lufs']+t['gain_db']))<.03


def test_matching_cap_silence_and_short_signal():
    x=tone();before=render._window_lufs(x,SR,0,len(x))
    m={'mode':'active_lufs_blend','source_lufs':before+20,'amount':1,'max_gain_db':6,'start_frame':0,'end_frame':len(x)}
    _,r=render._level_match(x,SR,m,len(x));assert r['capped'] and r['applied_db']==6
    z=np.zeros_like(x);got,r=render._level_match(z,SR,m,len(z))
    assert r['skipped'] and np.isfinite(got).all() and not got.any()
    short=tone(seconds=.15);m.update(start_frame=0,end_frame=len(short),source_lufs=render._window_lufs(short,SR,0,len(short))+2)
    _,r=render._level_match(short,SR,m,len(short));assert abs(r['after_lufs']-m['source_lufs'])<.01
    with pytest.raises(render.RenderError):render._level_match(short,SR,dict(m,end_frame=len(short)+1),len(short))


@pytest.mark.parametrize('key,value',[('mode','magic'),('amount',float('nan')),('amount',1.1),('source_lufs',float('inf')),('max_gain_db',7),('start_frame',-1),('end_frame',0),('end_frame',1000000),('start_frame',True)])
def test_invalid_level_match_rejected_without_jsonschema(sources,key,value,schema_free):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'))
    p['tracks'][0]['level_match'][key]=value
    with pytest.raises((planner.PlanError,jsonschema.ValidationError)):planner.validate(p)
    with pytest.raises(planner.PlanError):planner.validate(p,schema_free)


def export_plan():
    return {'source':{'sample_rate':SR},'buses':{'master':{'target_lufs':-14,'true_peak_db':-1,'limiter':{'ceiling_db':-1}}}}


def read_export(path):
    return analyze.load_stem_file(str(path))[0]


def test_actual_export_includes_return_and_one_shared_headroom_gain(tmp_path):
    a=tone(220,1.7);b=tone(440,.4);wet=tone(660,.2)
    report={};col={'stems':{'_reverb':a,'../../mix':b},'reverb':wet}
    written=render.write_stem_export(str(tmp_path),export_plan(),col,report,(a+b+wet)*.1)
    meta=json.loads((tmp_path/'_export_report.json').read_text())
    assert meta['common_gain_db']<0
    assert len({x['filename'].casefold() for x in meta['files']})==3
    assert all('/' not in x['filename'] and '\\' not in x['filename'] for x in meta['files'])
    assert {x['kind'] for x in meta['files']}=={'track','return'}
    total=np.zeros_like(a);gain=dsp.db_to_lin(meta['common_gain_db'])
    originals={'_reverb':a,'../../mix':b}
    for entry in meta['files']:
        x=read_export(tmp_path/entry['filename']);total+=x
        assert np.max(np.abs(x))<.892
        orig=wet if entry['kind']=='return' else originals[entry['source_id']]
        np.testing.assert_allclose(x,orig*gain,atol=3e-7)
    assert np.max(np.abs(total))<dsp.db_to_lin(-1)
    actual=10*np.log10(np.mean(total**2))
    assert meta['comparison']['sum_rms_dbfs']==pytest.approx(actual,abs=1e-8)
    assert '_export_report.json' in [x['name'] for x in written]
    assert 'does NOT reproduce mix.wav' in (tmp_path/'README.txt').read_text()


def test_export_metrics_separate_gain_from_tone_and_keep_stereo_energy():
    x=tone(500);x[:,1]*=-1
    m=render._export_comparison(x,x*.5,SR)
    assert m['reference_minus_sum_rms_dbfs']==pytest.approx(-6.0206,abs=.001)
    assert m['gain_match_linear']==pytest.approx(.5,abs=1e-6)
    assert m['gain_matched_residual_ratio']<1e-6
    mid=next(b for b in m['bands'] if b['name']=='mid')
    assert mid['sum_dbfs'] is not None # anti-phase stereo must not disappear
    assert mid['gain_matched_gap_db']==pytest.approx(0,abs=.001)
    mixed=tone(500)+tone(4000)
    coloured=tone(500)+tone(4000)*.3
    m=render._export_comparison(mixed,coloured,SR)
    assert m['gain_matched_residual_ratio']>.3
    bands={b['name']:b for b in m['bands']}
    assert bands['presence']['gain_matched_gap_db']<bands['mid']['gain_matched_gap_db']-9


def test_silent_export_metrics_are_explicit_and_invalid_alignment_fails(tmp_path):
    z=np.zeros((SR,2));r={}
    render.write_stem_export(str(tmp_path),export_plan(),{'stems':{'silent':z}},r,z)
    assert r['export_summary']['comparison']['correlation'] is None
    assert r['export_summary']['comparison']['sum_rms_dbfs'] is None
    with pytest.raises(render.RenderError):
        render.write_stem_export(str(tmp_path),export_plan(),{'stems':{'a':tone(),'b':tone(seconds=.5)}})


def test_capacity_counts_both_channels_exact_frames_and_return():
    p={'source':{'sample_rate':SR,'frames':SR,'duration_s':1}}
    assert server._export_size_bytes(p,4,True)==5*(SR*6+44)
    assert server._export_size_bytes(p,4,False)==4*(SR*6+44)


def test_editing_only_plan_gain_still_moves_the_rendered_track(sources):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'),balance_preserve=1)
    before,_=render_tracks(sources,p)
    t=p['tracks'][0];m=t['level_match'];t['gain_db']+=3
    after,_=render_tracks(sources,p)
    old=render._window_lufs(before[t['id']],SR,m['start_frame'],m['end_frame'])
    new=render._window_lufs(after[t['id']],SR,m['start_frame'],m['end_frame'])
    assert new-old==pytest.approx(3,abs=.03)


def test_frozen_window_excludes_later_energy_after_processing():
    quiet=tone(440,.05);loud=tone(440,.7)
    processed=np.vstack([quiet,loud])
    before=render._window_lufs(quiet,SR,0,len(quiet))
    rule={'mode':'active_lufs_blend','source_lufs':before+3,'amount':1,'max_gain_db':6,'start_frame':0,'end_frame':len(quiet)}
    _,r=render._level_match(processed,SR,rule,len(processed))
    assert r['applied_db']==pytest.approx(3,abs=.001)
    tiny=tone(seconds=.01)
    _,r=render._level_match(tiny,SR,dict(rule,end_frame=len(tiny)),len(tiny))
    assert r['skipped']


def test_export_refuses_symlink_destination_without_touching_target(tmp_path):
    original=tmp_path/'precious.bin';original.write_bytes(b'keep me')
    out=tmp_path/'export';out.mkdir();(out/'a.wav').symlink_to(original)
    with pytest.raises(render.RenderError):
        render.write_stem_export(str(out),export_plan(),{'stems':{'a':tone()}})
    assert original.read_bytes()==b'keep me'


def test_hierarchy_stage_clamp_reports_unmet_target():
    stems=[analyze.Stem('lead.wav','lead.wav',tone(440,.003),SR,role='lead_vocal'),
           analyze.Stem('bv.wav','bv.wav',tone(440,.9),SR,role='backing_vocal')]
    for s in stems:analyze.measure(s)
    p=planner.build_plan(stems,planner.load_preset('pop_2020s'),balance_preserve=1)
    lead,bv=p['tracks']
    wanted=stems[0].metrics['lufs']+lead['gain_db']-4-stems[1].metrics['lufs']
    gap=stems[0].metrics['lufs']+lead['gain_db']-stems[1].metrics['lufs']-bv['gain_db']
    assert wanted < -18 and bv['gain_db']==-18 and gap < 4
    text=' '.join(bv['evidence'])+' '.join(p['warnings'])
    assert 'hierarchy target NOT reached: stage clamp' in text
    assert f'gain wanted {wanted:+.2f} dB, applied -18.00 dB' in text
    assert f'gap {gap:+.2f} dB' in text
    assert 'below the lead vocal after adjustment' not in text


def test_band_absolute_dbfs_uses_mean_channel_energy():
    x=tone(500,.2);x[:,1]*=-1
    got=render._export_comparison(x,x*.5,SR)
    band=next(b for b in got['bands'] if b['name']=='mid')
    time_dbfs=10*np.log10(np.mean(x**2))
    assert band['sum_dbfs']==pytest.approx(time_dbfs,abs=1e-8)
    assert got['sum_rms_dbfs']==pytest.approx(time_dbfs,abs=1e-8)


def test_return_is_pure_convolution_before_fx_bus(sources,tmp_path):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'))
    p['buses']['fx']={'comp':{'threshold_db':-35,'ratio':2,'attack_ms':15,'release_ms':120},
                       'saturation':.2,'width':.7}
    collected={};mix,report=render.render(sources,p,collect=collected)
    send=sum((collected['stems'][t['id']]*t['sends']['plate'] for t in p['tracks']),np.zeros_like(mix))
    rev=p['reverb']
    pure=dsp.reverb_send(send,SR,1,rev['decay_s'],rev['pre_delay_ms'],rev['high_cut_hz'])
    np.testing.assert_allclose(collected['reverb'],pure,atol=1e-12)
    processed=render._set_width(dsp.saturate(render._apply_comp(pure,SR,p['buses']['fx']['comp']),.2),.7)
    assert not np.allclose(collected['reverb'],processed,atol=1e-7)
    dry=sum(collected['stems'].values(),np.zeros_like(mix))
    expected_ratio=20*np.log10(np.sqrt(np.mean(pure**2))/np.sqrt(np.mean(dry**2)))
    assert report['wet_ratio_db']==pytest.approx(expected_ratio,abs=.006)
    assert 'pre-bus' in report['wet_ratio_measurement']
    assert any(name=='reverb' and 'pre-bus' in value for name,value,_ in render._verdicts(p,report))
    render.write_stem_export(str(tmp_path),p,collected,report,mix)
    meta=json.loads((tmp_path/'_export_report.json').read_text())
    assert '(L squared + R squared) / 2' in meta['definitions']['bands']
    assert 'pure convolution return' in meta['definitions']['return']
    assert 'dry fx tracks' in meta['definitions']['return']
    assert 'pure convolution return before the fx bus' in (tmp_path/'README.txt').read_text()


@pytest.mark.parametrize('send',[0,.2])
def test_dry_fx_track_is_never_duplicated_as_a_return(tmp_path,send):
    source=analyze.Stem('fx.wav','fx.wav',tone(500,.2),SR,role='fx')
    analyze.measure(source)
    p=planner.build_plan([source],planner.load_preset('pop_2020s'))
    p['tracks'][0]['sends']['plate']=send
    collected={};mix,report=render.render([source],p,collect=collected)
    dry=collected['stems']['fx']
    if send==0:
        assert 'reverb' not in collected
        assert report['wet_ratio_db'] is None
        pure=np.zeros_like(dry)
    else:
        rev=p['reverb']
        pure=dsp.reverb_send(dry*send,SR,1,rev['decay_s'],rev['pre_delay_ms'],rev['high_cut_hz'])
        np.testing.assert_allclose(collected['reverb'],pure,atol=1e-12)
        assert not np.allclose(collected['reverb'],dry+pure,atol=1e-6)
    render.write_stem_export(str(tmp_path),p,collected,report,mix)
    meta=report['export_summary']
    assert len(meta['files'])==(1 if send==0 else 2)
    assert (tmp_path/'_reverb.wav').exists()==bool(send)
    total=sum((read_export(tmp_path/e['filename']) for e in meta['files']),np.zeros_like(dry))
    np.testing.assert_allclose(total,(dry+pure)*dsp.db_to_lin(meta['common_gain_db']),atol=5e-7)
    # Reusing a collection for a later zero-send render cannot retain the old return.
    p['tracks'][0]['sends']['plate']=0
    render.render([source],p,collect=collected)
    assert 'reverb' not in collected


def test_reused_collection_removes_stems_excluded_from_the_new_plan(sources,tmp_path):
    p=planner.build_plan(sources,planner.load_preset('pop_2020s'))
    collected={'unrelated_metadata':{'keep':True}}
    render.render(sources,p,collect=collected)
    assert set(collected['stems'])=={'vocal','drums'}
    p['tracks']=[t for t in p['tracks'] if t['id']=='vocal']
    p['tracks'][0]['sends']['plate']=0
    mix,report=render.render(sources,p,collect=collected)
    assert set(collected['stems'])=={'vocal'}
    assert 'reverb' not in collected
    assert collected['unrelated_metadata']=={'keep':True}
    render.write_stem_export(str(tmp_path),p,collected,report,mix)
    assert [f['source_id'] for f in report['export_summary']['files']]==['vocal']
    assert not (tmp_path/'drums.wav').exists()


@pytest.mark.parametrize('kind',['signal','silent','all_skipped'])
@pytest.mark.parametrize('collect_enabled',[False,True])
def test_completed_buffers_without_sends_or_sections(kind,collect_enabled):
    """Empty branches still render and preserve post-fx-pop bus metadata."""
    source=analyze.Stem('vocal.wav','vocal.wav',tone(),SR,role='lead_vocal')
    analyze.measure(source)
    p=planner.build_plan([source],planner.load_preset('pop_2020s'))
    p['sections']=[]
    p['tracks'][0]['sends']['plate']=0
    if kind=='silent':
        source.audio=np.zeros_like(source.audio)
    elif kind=='all_skipped':
        p['tracks']=[]
    collected={'unrelated_metadata':{'keep':True}} if collect_enabled else None
    mixed,report=render.render([source],p,collect=collected)
    assert np.isfinite(mixed).all()
    assert report['wet_ratio_db'] is None
    assert 'fx' not in report['buses']
    if kind=='all_skipped':
        assert report['buses']=={} and report['skipped_stems']==['vocal.wav']
    else:
        assert report['buses']=={'vocal':kind=='signal'}
    if kind!='signal':
        assert not np.any(mixed)
    if collect_enabled:
        assert collected['unrelated_metadata']=={'keep':True}
        assert 'reverb' not in collected
        assert set(collected['stems'])==({'vocal'} if kind!='all_skipped' else set())
