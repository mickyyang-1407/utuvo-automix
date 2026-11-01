"""Bounded same-song gain identification; frozen v2 math plus strict v3 pooled-consistency guard, no renderer inference.

Independent product implementation; no runtime docs/fixture imports.
"""
import numpy as np
from . import analyze, dsp
V2_LIMITS={'method':'positive_stereo_time_split_v2','block_s':2.,'max_design_condition':100.,'max_aggregate_residual':.08,'max_local_residual':.12,'min_reference_active_s':8.,'max_split_relative_db':.25,'max_gain_span_db':36.,'unique_over_reference':.01,'unique_over_residual':4.,'min_column_active_blocks_per_fold':2,'relative_gain_no_action_db':.05,'active_floor_relative_power':1e-4}
LIMITS={**V2_LIMITS,'method':'positive_stereo_time_split_v3','max_full_vs_fold_relative_db':.25}
SUPPORTED_METHODS=(V2_LIMITS['method'],LIMITS['method'])
def _total(stats,m):
    return sum((s[0] for s in stats),np.zeros((m,m))),sum((s[1] for s in stats),np.zeros(m)),sum(s[2] for s in stats),sum(s[3] for s in stats)
def _error(stats,w):
    G,b,yy,n=_total(stats,len(w));sse=max(0.,float(yy-2*w@b+w@G@w));return sse,float(yy),float(np.sqrt(sse/max(yy,1e-30)))
def _scaled(stats,scales):return [(G*np.outer(scales,scales),b*scales,yy,n) for G,b,yy,n in stats]
def _fit(stats,m):
    G,b,yy,n=_total(stats,m);power=np.diag(G);active=np.flatnonzero(power>max(float(power.max())*1e-24,1e-30));out={'active':active.tolist(),'inactive':[i for i in range(m) if i not in active]}
    if not len(active):return {**out,'rejected':'digital_silence'}
    norms=np.sqrt(power[active]);K=G[np.ix_(active,active)]/np.outer(norms,norms);ev=np.linalg.eigvalsh(K);cond=float(np.sqrt(ev[-1]/ev[0])) if ev[0]>0 else None
    out.update(design_condition=cond)
    if cond is None or cond>LIMITS['max_design_condition']:return {**out,'rejected':'rank_or_condition'}
    w=np.zeros(m);w[active]=np.linalg.solve(K,b[active]/norms)/norms
    if np.any(w[active]<=0):return {**out,'rejected':'nonpositive_coefficient'}
    raw=20*np.log10(w[active]);common=float((raw.min()+raw.max())/2);relative=raw-common
    _,_,res=_error(stats,w);corr=float(w@b/np.sqrt(max((w@G@w)*yy,1e-30)))
    return {**out,'weights':w.tolist(),'common_gain_db':common,'relative_gain_db':{str(i):float(v) for i,v in zip(active,relative)},'gain_span_db':float(raw.max()-raw.min()),'residual':res,'correlation':corr}
def _validate_stats(stats,scales,sr):
    m=len(scales);stats=_scaled(stats,scales);parts=[stats[::2],stats[1::2]];fits=[_fit(p,m) for p in parts];full=_fit(stats,m);result={'accepted':False,'train':fits[0],'heldout':fits[1],'all':full,'limits':dict(LIMITS)}
    def fail(reason):result['reason']=reason;return result
    for f in [*fits,full]:
        if f.get('rejected'):return fail(f['rejected'])
    active=full['active']
    if any(f['active']!=active for f in fits):return fail('inconsistent_active_columns')
    spread=max(abs(fits[0]['relative_gain_db'][str(i)]-fits[1]['relative_gain_db'][str(i)]) for i in active);result['split_gain_spread_db']=spread
    if spread>LIMITS['max_split_relative_db']:return fail('split_coefficients_unstable')
    if full['gain_span_db']>LIMITS['max_gain_span_db']:return fail('relative_gain_span_out_of_bounds')
    peak_ref=max(s[2]/s[3] for s in stats);ref_floor=peak_ref*LIMITS['active_floor_relative_power'];active_ref_s=sum(s[3]/(2*sr) for s in stats if s[2]/s[3]>=ref_floor)
    result['reference_active_seconds']=active_ref_s
    if active_ref_s<LIMITS['min_reference_active_s']:return fail('insufficient_reference_coverage')
    checks=[]
    for direction in range(2):
        w=np.asarray(fits[direction]['weights']);held=parts[1-direction];sse,yy,res=_error(held,w);local=[_error([s],w)[2] for s in held if s[2]/s[3]>=ref_floor];check={'direction':direction,'aggregate_residual':res,'max_active_local_residual':max(local,default=None),'columns':[]};checks.append(check);result['validation']=checks
        if res>LIMITS['max_aggregate_residual'] or not local or max(local)>LIMITS['max_local_residual']:return fail('local_or_global_validation_mismatch')
        for i in active:
            if abs(full['relative_gain_db'][str(i)])<=LIMITS['relative_gain_no_action_db']:continue
            peak=max(s[0][i,i]/s[3] for s in stats);windows=[s for s in held if s[0][i,i]/s[3]>=peak*LIMITS['active_floor_relative_power'] and s[2]/s[3]>=ref_floor]
            detail={'column':i,'active_blocks':len(windows)};check['columns'].append(detail)
            if len(windows)<LIMITS['min_column_active_blocks_per_fold']:return fail('insufficient_column_coverage')
            G,b,sy,ns=_total(windows,m);others=[j for j in active if j!=i and G[j,j]>0];unique=float(G[i,i])
            if others:
                norm=np.sqrt(np.diag(G)[others]);K=G[np.ix_(others,others)]/np.outer(norm,norm);cross=G[i,others]/norm
                unique-=float(cross@np.linalg.lstsq(K,cross,rcond=1e-10)[0])
            contribution=max(0.,unique)*w[i]**2;se,_,_=_error(windows,w);u=float(np.sqrt(contribution/max(sy,1e-30)));q=float(np.sqrt(contribution/max(se,sy*1e-24,1e-30)));detail.update(unique_over_reference=u,unique_over_residual=q)
            if u<LIMITS['unique_over_reference'] or q<LIMITS['unique_over_residual']:return fail('weak_or_unresolved_column')
    if full['residual']>LIMITS['max_aggregate_residual'] or full['correlation']<.99:return fail('not_coherent_linear_same_content')
    pooled_spread=max(abs(full['relative_gain_db'][str(i)]-fold['relative_gain_db'][str(i)]) for i in active for fold in fits)
    result['full_vs_fold_gain_spread_db']=pooled_spread
    if pooled_spread>LIMITS['max_full_vs_fold_relative_db']:return fail('pooled_coefficients_unstable')
    result.update(accepted=True,reason='stable_identifiable_same_content_fit');return result


def _stereo(audio):
    x = np.asarray(audio)
    if x.ndim == 1:
        return np.repeat(x[:, None], 2, axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('reference balance requires mono or stereo audio')
    return x


def _accumulate(stems, reference, sr, n):
    """Only one 2s×stems design block; retain small float64 sufficient stats."""
    stats = []
    for start in range(0, n, 2 * sr):
        end = min(n, start + 2 * sr)
        a = np.zeros(((end - start) * 2, len(stems)), dtype=np.float64)
        for index, stem in enumerate(stems):
            available = min(end, len(stem.audio)) - start
            if available > 0:
                a[:available * 2, index] = _stereo(stem.audio[start:start + available]).reshape(-1)
        y = np.asarray(reference[start:end], dtype=np.float64).reshape(-1)
        if not np.isfinite(a).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite_audio')
        with np.errstate(over='ignore', invalid='ignore'):
            gram, xy, yy = a.T @ a, a.T @ y, float(y @ y)
        if not np.isfinite(gram).all() or not np.isfinite(xy).all() or not np.isfinite(yy):
            raise ValueError('nonfinite_audio_energy')
        stats.append((gram, xy, yy, len(y)))
    return stats


def evaluate(stems, reference_audio, reference_sr):
    """Return a bounded auditable proof; no input mutation or signal processing.

    Common fit gain is receipt-only. Only an accepted relative_gain_db mapping
    may be consumed by the planner; failures never expose usable corrections.
    """
    result = {'method': LIMITS['method'], 'passed': False, 'limits': dict(LIMITS),
              'input_ids': [analyze.track_id(s.name) for s in stems]}
    def fail(reason):
        result['reason'] = reason
        return result
    if not stems:
        return fail('no_source_audio')
    sr = int(stems[0].sr)
    if any(s.sr != sr for s in stems):
        return fail('source_sample_rates_not_aligned')
    ids = [analyze.track_id(s.name) for s in stems]
    if len(set(ids)) != len(ids):
        return fail('ambiguous_track_ids')
    try:
        ref = _stereo(reference_audio)
        duration = max(len(s.audio) for s in stems) / sr
        reference_duration = len(ref) / reference_sr
        result.update(source_duration_s=duration, reference_duration_s=reference_duration,
                      source_sample_rate=sr, reference_sample_rate=int(reference_sr),
                      duration_difference_s=abs(duration-reference_duration),
                      sample_rate_treatment='native' if sr == reference_sr else 'fft_resample_reference')
        if abs(duration-reference_duration) > .05:
            return fail('duration_mismatch')
        if min(duration, reference_duration) < 8:
            return fail('shorter_than_minimum_coverage')
        selected = [i for i,s in enumerate(stems) if np.any(s.audio)]
        if len(selected) > 32:
            return fail('more_than_32_nonzero_stems')
        result['excluded_ids'] = [id for i,id in enumerate(ids) if i not in selected]
        if not selected:
            return fail('digital_silence')
        if reference_sr != sr:
            ref = dsp.resample_fft(ref, reference_sr, sr)
        n = min(max(len(s.audio) for s in stems), len(ref))
        result['comparable_duration_s'] = n / sr
        usable = [stems[i] for i in selected]
        stats = _accumulate(usable, ref, sr, n)
        checked = _validate_stats(stats, np.ones(len(usable)), sr)
        def finite(value):
            if isinstance(value, (int,float,np.integer,np.floating)): return bool(np.isfinite(value))
            if isinstance(value,dict): return all(finite(v) for v in value.values())
            if isinstance(value,list): return all(finite(v) for v in value)
            return True
        if not finite(checked):
            return fail('nonfinite_fit_statistics')

        full = checked['all']
        active = full['active']
        result['active_ids'] = [ids[selected[i]] for i in active]
        result['excluded_ids'] += [ids[selected[i]] for i in full['inactive']]
        # Small bounded summaries only: no Gram, audio, or per-sample evidence.
        for source, dest in (('split_gain_spread_db','split_gain_spread_db'),
                             ('reference_active_seconds','reference_active_seconds'),
                             ('full_vs_fold_gain_spread_db','full_vs_fold_gain_spread_db')):
            if source in checked:
                result[dest] = checked[source]
        for source in ('common_gain_db', 'gain_span_db', 'design_condition', 'residual', 'correlation'):
            if source in full:
                result[source] = full[source]
        folds = []
        for index, fit in enumerate((checked['train'], checked['heldout'])):
            folds.append({'direction': index, 'design_condition': fit.get('design_condition'),
                          'rejected': fit.get('rejected')})
        for check in checked.get('validation', []):
            fold = folds[check['direction']]
            fold.update(aggregate_residual=check['aggregate_residual'],
                        max_active_local_residual=check['max_active_local_residual'],
                        columns=[{**{k:v for k,v in column.items() if k != 'column'},
                                  'id': ids[selected[column['column']]]} for column in check['columns']])
        result['folds'] = folds
        if not checked['accepted']:
            return fail(checked['reason'])
        result.update(passed=True, reason=checked['reason'],
                      common_gain_applied=False,
                      raw_coefficients={ids[selected[i]]: float(full['weights'][i]) for i in active},
                      relative_gain_db={ids[selected[int(i)]]: value for i,value in full['relative_gain_db'].items()})
        return result
    except (ValueError, np.linalg.LinAlgError) as exc:
        return fail(str(exc))


def validate_receipt(proof, known_ids=None, source_sr=None, source_frames=None):
    """Structural and physical consistency checks, also without jsonschema."""
    allowed = {'method','passed','reason','limits','input_ids','scalar_proof',
               'source_duration_s','reference_duration_s','source_sample_rate','reference_sample_rate',
               'duration_difference_s','sample_rate_treatment','comparable_duration_s',
               'active_ids','excluded_ids','split_gain_spread_db','full_vs_fold_gain_spread_db','reference_active_seconds',
               'common_gain_db','gain_span_db','design_condition','residual','correlation','folds',
               'common_gain_applied','raw_coefficients','relative_gain_db'}
    if (not isinstance(proof, dict) or set(proof)-allowed or proof.get('method') not in SUPPORTED_METHODS
            or type(proof.get('passed')) is not bool or not isinstance(proof.get('reason'),str)
            or proof.get('limits') != (V2_LIMITS if proof.get('method')==V2_LIMITS['method'] else LIMITS)):
        raise ValueError('invalid reference balance proof settings')
    def number(value, low=-float('inf'), high=float('inf')):
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)
                or not low <= value <= high):
            raise ValueError('reference balance proof requires finite bounded numbers')
    def ids(key):
        values = proof.get(key)
        if (not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('reference balance proof requires unique track IDs')
        return set(values)
    inputs = ids('input_ids')
    active = ids('active_ids') if 'active_ids' in proof else set()
    excluded = ids('excluded_ids') if 'excluded_ids' in proof else set()
    if not active <= inputs or not excluded <= inputs or active & excluded or len(active)>32:
        raise ValueError('reference balance active/excluded IDs disagree')
    if known_ids is not None and inputs != set(known_ids):
        raise ValueError('reference balance input IDs disagree with the known source graph')
    scalar = proof.get('scalar_proof')
    if scalar is not None and (not isinstance(scalar,dict) or scalar.get('method') != 'aligned_stereo_gain_fit_v1'
                               or scalar.get('passed') is not False or not isinstance(scalar.get('reason'),str)):
        raise ValueError('reference balance scalar fallback proof is invalid')
    folds = proof.get('folds', [])
    if (not isinstance(folds,list) or len(folds)>2 or any(not isinstance(f,dict) for f in folds)
            or len({f.get('direction') for f in folds}) != len(folds)):
        raise ValueError('invalid reference fit temporal folds')
    for fold in folds:
        if (set(fold)-{'direction','design_condition','rejected','aggregate_residual','max_active_local_residual','columns'}
                or type(fold.get('direction')) is not int or fold['direction'] not in (0,1)
                or (fold.get('rejected') is not None and not isinstance(fold['rejected'],str))):
            raise ValueError('invalid reference fit temporal fold fields')
        for key in ('design_condition','aggregate_residual','max_active_local_residual'):
            if fold.get(key) is not None: number(fold[key],0)
        columns=fold.get('columns',[])
        if not isinstance(columns,list) or len(columns)>32:
            raise ValueError('invalid reference fit column checks')
        for column in columns:
            if (not isinstance(column,dict) or set(column)-{'id','active_blocks','unique_over_reference','unique_over_residual'}
                    or column.get('id') not in active or type(column.get('active_blocks')) is not int or column['active_blocks']<0):
                raise ValueError('invalid reference fit column fields')
            for key in ('unique_over_reference','unique_over_residual'):
                if key in column: number(column[key],0)
    if proof['method']==V2_LIMITS['method']:
        if 'full_vs_fold_gain_spread_db' in proof:
            raise ValueError('historical v2 proof cannot declare v3 pooled validation')
    elif 'full_vs_fold_gain_spread_db' in proof:
        number(proof['full_vs_fold_gain_spread_db'],0)
    if proof.get('reason')=='pooled_coefficients_unstable':
        if (proof['method']!=LIMITS['method'] or proof.get('full_vs_fold_gain_spread_db',0)<=LIMITS['max_full_vs_fold_relative_db']):
            raise ValueError('pooled reference fit rejection lacks its measured mismatch')
    if not proof['passed']:
        if 'relative_gain_db' in proof or 'raw_coefficients' in proof or 'common_gain_applied' in proof:
            raise ValueError('rejected reference fit cannot carry usable corrections')
        return
    if not active or active | excluded != inputs or proof.get('common_gain_applied') is not False:
        raise ValueError('accepted reference fit must exclude common gain and account for inputs')
    coefficients, relative = proof.get('raw_coefficients'), proof.get('relative_gain_db')
    if (not isinstance(coefficients,dict) or not isinstance(relative,dict)
            or set(coefficients)!=active or set(relative)!=active):
        raise ValueError('reference fit coefficients must match active track IDs')
    for value in coefficients.values(): number(value, np.finfo(float).tiny)
    for value in relative.values(): number(value,-18,18)
    logs = np.array([20*np.log10(coefficients[id]) for id in sorted(active)])
    common = proof.get('common_gain_db'); number(common)
    if (abs(common-(float(logs.min())+float(logs.max()))/2)>1e-7
            or any(abs(relative[id]-(20*np.log10(coefficients[id])-common))>1e-7 for id in active)):
        raise ValueError('reference fit canonical relative gains disagree with raw coefficients')
    number(proof.get('gain_span_db'),0,LIMITS['max_gain_span_db'])
    if abs(proof['gain_span_db']-float(logs.max()-logs.min()))>1e-7:
        raise ValueError('reference fit gain span disagrees')
    number(proof.get('design_condition'),1e-12,LIMITS['max_design_condition'])
    number(proof.get('split_gain_spread_db'),0,LIMITS['max_split_relative_db'])
    if proof['method']==LIMITS['method']:
        number(proof.get('full_vs_fold_gain_spread_db'),0,LIMITS['max_full_vs_fold_relative_db'])
    number(proof.get('reference_active_seconds'),LIMITS['min_reference_active_s'])
    number(proof.get('residual'),0,LIMITS['max_aggregate_residual'])
    number(proof.get('correlation'),.99,1.0000001)
    for key in ('source_duration_s','reference_duration_s','comparable_duration_s'): number(proof.get(key),8)
    number(proof.get('duration_difference_s'),0,.05)
    for key in ('source_sample_rate','reference_sample_rate'):
        if type(proof.get(key)) is not int or proof[key]<=0:
            raise ValueError('reference fit sample rates must be positive integers')
    if source_sr is not None and proof['source_sample_rate'] != source_sr:
        raise ValueError('reference fit source sample rate disagrees with plan')
    if source_frames is not None and source_sr is not None and abs(proof['source_duration_s']-source_frames/source_sr)>1e-9:
        raise ValueError('reference fit source duration disagrees with plan')
    if (abs(proof['duration_difference_s']-abs(proof['source_duration_s']-proof['reference_duration_s']))>1e-9
            or proof['comparable_duration_s']>min(proof['source_duration_s'],proof['reference_duration_s'])+1/proof['source_sample_rate']):
        raise ValueError('reference fit duration metadata disagrees')
    expected_treatment = 'native' if proof['source_sample_rate']==proof['reference_sample_rate'] else 'fft_resample_reference'
    if proof.get('sample_rate_treatment') != expected_treatment:
        raise ValueError('reference fit sample rate treatment disagrees')
    if proof.get('sample_rate_treatment') not in ('native','fft_resample_reference'):
        raise ValueError('reference fit sample rate treatment is invalid')
    folds = proof.get('folds')
    if len(folds)!=2 or {f['direction'] for f in folds} != {0,1}:
        raise ValueError('reference fit requires both temporal validation directions')
    required_columns = {id for id in active if abs(relative[id])>LIMITS['relative_gain_no_action_db']}
    for fold in folds:
        if fold.get('rejected') is not None:
            raise ValueError('accepted reference fit has a rejected temporal fold')
        number(fold.get('design_condition'),1e-12,LIMITS['max_design_condition'])
        number(fold.get('aggregate_residual'),0,LIMITS['max_aggregate_residual'])
        number(fold.get('max_active_local_residual'),0,LIMITS['max_local_residual'])
        columns=fold.get('columns')
        if (not isinstance(columns,list) or len(columns)!=len(required_columns)
                or {c.get('id') for c in columns}!=required_columns):
            raise ValueError('reference fit lacks per-column heldout checks')
        for column in columns:
            if type(column.get('active_blocks')) is not int or column['active_blocks']<2:
                raise ValueError('reference fit column has insufficient activity')
            number(column.get('unique_over_reference'),LIMITS['unique_over_reference'])
            number(column.get('unique_over_residual'),LIMITS['unique_over_residual'])
