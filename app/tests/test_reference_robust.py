"""Reference analysis invariants and physical, fixed-support result evidence."""
import copy
import numpy as np
import pytest
from automix import analyze,dsp,reference as ref,plan as planner,pipeline,render

SR=24000


def pair(seconds=4,sr=SR):
    t=np.arange(int(seconds*sr))/sr
    return .1*np.sin(2*np.pi*250*t),.1*np.sin(2*np.pi*4000*t)


def difference(tone,a=4000.,b=250.):
    return tone[ref.THIRD_OCTAVE_CENTERS.index(a)]-tone[ref.THIRD_OCTAVE_CENTERS.index(b)]


def test_energy_spectrum_preserves_center_side_polarity_and_channel_swap():
    low,high=pair()
    variants=[np.column_stack([low+high]*2),np.column_stack([low+high,low-high]),
              np.column_stack([low+high,-low-high])]
    tones=[ref.spectrum_db(a,SR) for a in variants]
    # Whole-channel polarity is an exact energy invariant. Moving only one
    # component to side can change finite-window leakage in silent bins;
    # compare its physical occupied-band contrast, not the noise-floor mean.
    np.testing.assert_allclose(tones[0],tones[2],atol=.001)
    assert difference(tones[0])==pytest.approx(difference(tones[1]),abs=.03)
    np.testing.assert_allclose(tones[1],ref.spectrum_db(variants[1][:,::-1],SR),atol=.001)
    assert abs(difference(tones[1]))<.03


def test_whole_duration_has_equal_time_weight_and_gain_invariance():
    low,high=pair(12);t=np.arange(len(low))/SR
    ratios=[]
    for start in (.5,5.5,10.5):
        x=np.column_stack([low+high*((t>=start)&(t<start+1))]*2)
        tone=ref.spectrum_db(x,SR);ratios.append(difference(tone))
        np.testing.assert_allclose(tone,ref.spectrum_db(x*.2,SR),atol=.01)
    assert max(ratios)-min(ratios)<.03
    assert np.mean(ratios)==pytest.approx(10*np.log10(1/12),abs=.03)


def test_empty_short_silent_and_low_rate_spectra_are_finite(tmp_path):
    for n,sr in [(0,8000),(1,8000),(10,8000),(127,48000)]:
        x=np.ones((n,2))*.01
        got=ref.spectrum_db(x,sr)
        assert len(got)==31 and np.isfinite(got).all()
    assert ref.spectrum_db(np.zeros((400,2)),8000)==[0]*31
    path=tmp_path/'silence.wav';render.write_wav24(str(path),np.zeros((8000,2)),8000)
    with pytest.raises(ref.ReferenceError,match='silent'):ref.analyse(str(path))
    with pytest.raises(ref.ReferenceError):ref.spectrum_db(np.array([np.nan]),8000)


@pytest.mark.parametrize("seconds", [.5, 1, 3, 20])
def test_common_support_removes_missing_nyquist_band_offset(seconds):
    def tone(sr,frequencies):
        t=np.arange(int(seconds*sr))/sr
        return ref.spectrum_db(sum(.1*np.sin(2*np.pi*f*t) for f in frequencies),sr)
    a=tone(8000,[1000,2500]);b=tone(44100,[1000,2500,8000])
    indices=ref.common_support(a,b,reference_sr=8000,base_sr=44100)
    assert [ref.THIRD_OCTAVE_CENTERS[i] for i in indices]==[1000,2500]
    assert all(abs(v)<.03 for v in ref.corrective_tilt(a,b,indices=indices).values())
    assert ref.tone_distance(a,b,indices)<.03
    # One comparable band cannot carry a relative shape contrast.
    assert all(v==0 for v in ref.corrective_tilt(a,b,indices=[17]).values())
    assert ref.tone_distance(a,b,[17]) is None


def test_fixed_comparison_never_drops_a_band_that_gets_worse():
    reference=[-100.]*31;base=reference.copy();after=reference.copy()
    reference[17]=reference[22]=0;base[17]=0;base[22]=-4;after[17]=0;after[22]=-100
    indices=ref.common_support(reference,base)
    assert indices==[17,22]
    assert ref.tone_distance(reference,after,indices)>40
    assert ref.common_support(reference,after)==[17]


def test_known_reference_eq_moves_toward_target_and_reports_fixed_mask(tmp_path):
    sr=24000;t=np.arange(sr*4)/sr
    audio=.12*np.sin(2*np.pi*110*t)+.09*np.sin(2*np.pi*440*t)+.06*np.sin(2*np.pi*3200*t)
    s=analyze.Stem('music.wav','music.wav',np.column_stack([audio]*2),sr,role='keys')
    analyze.measure(s);preset=planner.load_preset('acoustic')
    p0,a0,r0=pipeline.mix([s],preset)
    target=dsp.apply_eq(a0,sr,[('lowshelf',120,3,.7)])
    path=tmp_path/'coloured.wav';render.write_wav24(str(path),target,sr)
    reference=ref.analyse(str(path))
    p,a,r=pipeline.mix([s],preset,reference=reference,blend=1)
    evidence=r['reference_tone']
    assert evidence['after_db']<evidence['before_db']-.05,evidence
    assert evidence['indices']==p['style']['reference']['tone_comparison']['indices']
    assert evidence['band_count']>=2
    assert p['style']['reference']['tone_method']['id']==ref.TONE_METHOD['id']
    report=tmp_path/'report.md';render.write_report(str(report),p,r,[s])
    assert 'relative-tone RMS distance' in report.read_text()
    # Reporting a saved plan must not change its rendered signal.
    replay,rr=render.render([s],copy.deepcopy(p))
    pipeline.note_reference_tone(p,replay,rr)
    np.testing.assert_array_equal(a,replay)
    assert rr['reference_tone']['indices']==evidence['indices']


def test_insufficient_common_support_receipt_has_no_null_distance_text(tmp_path):
    sr=24000;t=np.arange(sr)/sr
    s=analyze.Stem('keys.wav','keys.wav',np.column_stack([.1*np.sin(2*np.pi*1000*t)]*2),sr,role='keys')
    analyze.measure(s);p,a,r=pipeline.mix([s],planner.load_preset('acoustic'))
    tone=[-120.]*31;tone[17]=0
    indices=ref.common_support(tone,tone)
    assert indices==[17]
    p['style']['reference']={'file':'single-band.wav','tone':tone,'lufs':r['lufs'],
        'crest_db':r['crest_db'],'width':r['width'],
        'tone_comparison':{'method':'rms_centered_common_third_octave_db','indices':indices,
            'centers_hz':[1000.],'baseline_tone':tone,'baseline_kind':'post_style_render',
            'before_db':ref.tone_distance(tone,tone,indices)}}
    pipeline.note_reference_tone(p,a,r)
    assert r['reference_tone']['before_db'] is None and r['reference_tone']['after_db'] is None
    path=tmp_path/'receipt.md';render.write_report(str(path),p,r,[s]);text=path.read_text()
    assert 'not enough comparable bands (no relative-tone conclusion)' in text
    assert 'None -> None' not in text
