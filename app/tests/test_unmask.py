"""Independent energy, timing and physical DSP oracles for frozen unmask."""
import copy
import jsonschema
import numpy as np
import pytest
from automix import analyze,dsp,plan as planner,render,unmask
SR=24000
DURATION=8


def sources(vocal=True,overlap=True,role='guitar_electric',dark=False):
    t=np.arange(SR*DURATION)/SR
    music=.2*np.sin(2*np.pi*110*t)+.08*np.sin(2*np.pi*3200*t)
    if not overlap:music*=t<1
    voice=.15*np.sin(2*np.pi*660*t)
    if not dark:voice+=.06*np.sin(2*np.pi*2640*t)
    voice*=((t>=2)&(t<4))
    items=[analyze.Stem('music.wav','music.wav',np.column_stack([music]*2),SR,role=role)]
    if vocal:items.append(analyze.Stem('lead.wav','lead.wav',np.column_stack([voice]*2),SR,role='lead_vocal'))
    for s in items:analyze.measure(s)
    return items


def build(items,**kwargs):
    return planner.build_plan(items,planner.load_preset('pop_2020s'),balance_preserve=1,**kwargs)


def target(p):
    return next(t for t in p['tracks'] if t['id']=='music')


def test_frame_energy_has_physical_units_duration_and_polarity_invariance():
    for seconds in (1,6):
        t=np.arange(SR*seconds)/SR;x=.2*np.sin(2*np.pi*3200*t)
        a=unmask.frame_power(np.column_stack([x,-x]),SR,len(x))
        b=unmask.frame_power(np.column_stack([x,x]),SR,len(x))
        assert 10*np.log10(np.median(a['presence'][:-2]))==pytest.approx(-16.9897,abs=.02)
        np.testing.assert_allclose(a['presence'],b['presence'],atol=1e-12)


def test_plan_temporal_competition_and_intensity_scales_once():
    items=sources();p=build(items);block=target(p)['dynamic_eq']
    assert block['source_ids']==['lead'] and block['band']['gain_db']==-1.5
    curve=np.asarray(block['curve']);assert curve[:,1].min()==0 and curve[:,1].max()==1
    assert np.all(np.diff(curve[:,0])>0)
    slopes=np.diff(curve[:,1])/np.diff(curve[:,0])
    assert slopes.max()<=20.001 and slopes.min()>=-6.668
    half=target(build(items,intensity=.5))['dynamic_eq']
    assert half['band']['gain_db']==-.75 and half['curve']==block['curve']
    off=build(items,intensity=0)
    assert 'dynamic_eq' not in target(off)
    assert not any('vocal masking' in str(e) for e in target(off)['evidence'])


@pytest.mark.parametrize('case',['no_vocal','no_overlap','dark','silent_vocal','quiet_after_trim'])
def test_negative_controls_do_not_carve(case):
    items=sources(vocal=case!='no_vocal',overlap=case!='no_overlap',dark=case=='dark')
    if case=='silent_vocal':items[1].audio[:]=0;analyze.measure(items[1])
    options={'trims':{'music':-18}} if case=='quiet_after_trim' else {}
    p=build(items,**options)
    assert all('dynamic_eq' not in t for t in p['tracks']),p['unmask']


def test_generic_accompaniment_cap_and_maximum_two_targets():
    items=sources(role='other');p=build(items)
    assert target(p)['dynamic_eq']['band']['gain_db']==-.8
    for i in range(4):
        item=copy.deepcopy(items[0]);item.name=f'music{i}.wav';item.path=item.name;items.append(item)
    p=build(items)
    assert sum('dynamic_eq' in t for t in p['tracks'])==2
    assert all(t.get('dynamic_eq',{}).get('band',{}).get('gain_db',0)>=-.8 for t in p['tracks'])


def test_multiple_leads_use_order_independent_power_group_and_union():
    items=sources();second=copy.deepcopy(items[1]);second.name='lead2.wav';second.path=second.name
    second.audio=np.roll(second.audio,3*SR,axis=0);analyze.measure(second);items.append(second)
    a=target(build(items))['dynamic_eq'];b=target(build(list(reversed(items))))['dynamic_eq']
    assert a['source_ids']==['lead','lead2'] and a['curve']==b['curve']
    points=np.asarray(a['curve'])
    assert np.interp(3,points[:,0],points[:,1])==1
    assert np.interp(6,points[:,0],points[:,1])==1


def test_dynamic_stage_identity_bounded_cut_and_stereo_direction():
    block=target(build(sources()))['dynamic_eq']
    t=np.arange(SR*DURATION)/SR;tone=.2*np.sin(2*np.pi*3200*t)
    x=np.column_stack([tone,tone*.5]);y,r=render._dynamic_eq(x,SR,block)
    points=np.asarray(block['curve']);env=np.interp(t,points[:,0],points[:,1])
    np.testing.assert_array_equal(y[env==0],x[env==0])
    np.testing.assert_allclose(y[:,0],y[:,1]*2,atol=1e-12)
    w=slice(int(2.5*SR),int(3.5*SR))
    attenuation=10*np.log10(np.mean(y[w]**2)/np.mean(x[w]**2))
    assert attenuation==pytest.approx(-1.5,abs=.03)
    assert r['inactive_max_abs_error']==0 and r['max_nominal_cut_db']==pytest.approx(1.5)
    assert r['probes'] and all(abs(v['measured_db']-v['expected_center_db'])<.1 for v in r['probes'])


def test_level_match_target_survives_dynamic_stage():
    items=sources();p=build(items);collected={};_,r=render.render(items,p,collect=collected)
    t=target(p);m=t['level_match']
    got=render._window_lufs(collected['stems']['music'],SR,m['start_frame'],m['end_frame'])
    assert got==pytest.approx(m['source_lufs']+t['gain_db'],abs=.03)
    rep=next(t for t in r['tracks'] if t['id']=='music')
    assert any(step['step']=='dynamic vocal EQ' for step in rep['chain'])
    assert rep['dynamic_eq']['duty_fraction']<.4
    assert rep['dynamic_eq']['measurement_stage']=='dynamic EQ before compressor and loudness match'


@pytest.mark.parametrize('change',['mode','nan','unordered','duplicate','range','abrupt','depth','frequency','priority','extra'])
def test_malformed_curves_fail_without_optional_validator(change,schema_free):
    p=build(sources());block=target(p)['dynamic_eq']
    if change=='mode':block['mode']='guess'
    if change=='nan':block['curve'][1][1]=float('nan')
    if change=='unordered':block['curve']=[[0,0],[1,0],[.5,1]]
    if change=='duplicate':block['curve']=[[0,0],[0,1]]
    if change=='range':block['curve'][1][1]=1.1
    if change=='abrupt':block['curve']=[[0,0],[.001,1]]
    if change=='depth':block['band']['gain_db']=-2
    if change=='frequency':block['band']['freq_hz']=14000
    if change=='priority':block['source_ids']=['missing']
    if change=='extra':block['unused_knob']=.5
    with pytest.raises((planner.PlanError,jsonschema.ValidationError)):planner.validate(p)
    with pytest.raises(planner.PlanError):planner.validate(p,schema_free)


def test_old_plan_without_block_keeps_static_eq_replay():
    items=sources();p=build(items)
    for t in p['tracks']:t.pop('dynamic_eq',None)
    target(p)['eq'].append({'type':'bell','freq':3200,'gain_db':-1.5,'q':1.2,'enabled':True})
    a,_=render.render(items,p);b,_=render.render(items,copy.deepcopy(p))
    np.testing.assert_array_equal(a,b)


def test_silent_drop_and_input_order_cannot_mismatch_source_ids():
    items=sources()
    silent=analyze.Stem('00silent.wav','00silent.wav',np.zeros_like(items[0].audio),SR,role='keys')
    analyze.measure(silent)
    normal=build(items);dropped=build([silent]+items)
    assert target(normal)['dynamic_eq']==target(dropped)['dynamic_eq']
    # The helper matches IDs directly even when its input lists differ in order.
    tracks=copy.deepcopy(normal['tracks'])
    for t in tracks:t.pop('dynamic_eq',None)
    unmask.build(list(reversed(items)),tracks,1,{'max_eq_db':6})
    assert target({'tracks':tracks})['dynamic_eq']==target(normal)['dynamic_eq']


@pytest.mark.parametrize('frames,peak_time',[(51022,1.05),(39991,.8)])
def test_partial_tail_curve_retains_precision_and_satisfies_its_own_slope_contract(frames,peak_time):
    sr=44100;size=round(.1*sr);hop=round(.05*sr)
    times=np.minimum((np.arange(0,frames,hop)+size/2)/sr,frames/sr)
    gate=np.isclose(times,peak_time)|(times>peak_time+.100001)
    curve=unmask._curve(gate,times,frames/sr)
    block={'mode':unmask.MODE,'source_ids':['lead'],
           'band':{'freq_hz':3200,'gain_db':-.8,'q':1.2},'curve':curve,
           'analysis':{'method':unmask.METHOD}}
    unmask.validate(block,'other',sr,frames/sr,{'lead'})
    assert curve[-1][0]==frames/sr
    points=np.asarray(curve);slope=np.diff(points[:,1])/np.diff(points[:,0])
    assert slope.max()<=20.001 and slope.min()>=-6.668


def test_frame_power_explicit_mono_shapes_match_stereo_energy():
    t=np.arange(SR)/SR
    audio=.2*np.sin(2*np.pi*3200*t)
    stereo=unmask.frame_power(np.column_stack([audio,audio]),SR,len(audio))
    for shape in (audio,audio[:,None]):
        measured=unmask.frame_power(shape,SR,len(audio))
        for key in ('rms','mid','presence','times'):
            np.testing.assert_array_equal(measured[key],stereo[key])
    for unsupported in (np.zeros((10,3)),np.zeros((2,10,2)),np.zeros((10,0))):
        with pytest.raises(ValueError,match='mono or stereo'):
            unmask.frame_power(unsupported,SR,10)
