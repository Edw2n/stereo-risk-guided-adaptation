"""Prediction-only DeltaAbs on a declared full-native finite domain; no GT."""
import hashlib
import io
import json
from pathlib import Path
import h5py
import numpy as np

SHAPE=(1080,1920)


def load_validated(path, manifest_row, expected_metadata, shape=SHAPE, throttle=None):
    path=Path(path)
    raw=path.read_bytes()
    if throttle is not None:throttle(len(raw))
    if hashlib.sha256(raw).hexdigest()!=manifest_row['file_sha256']:
        raise ValueError('Prediction file SHA256 mismatch: '+str(path))
    with h5py.File(io.BytesIO(raw),'r') as f:
        if set(f.keys())!={'disparity'}:raise ValueError('Unexpected disparity datasets')
        d=f['disparity'][:]
        metadata=json.loads(f.attrs['provenance_json'])
    if d.shape!=tuple(shape) or d.dtype!=np.float32:raise ValueError('Native disparity shape/dtype mismatch')
    if not np.isfinite(d).all():raise ValueError('NaN/Inf anywhere; no pixel dropping or filling')
    digest=hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()
    if digest!=metadata.get('array_sha256') or digest!=manifest_row['array_sha256']:
        raise ValueError('Native disparity content SHA256 mismatch')
    for k,v in expected_metadata.items():
        if metadata.get(k)!=v:raise ValueError('Prediction metadata mismatch: '+k)
    return d,metadata,len(raw)


def disagreement(clean,corrupt):
    if clean.shape!=corrupt.shape or clean.ndim!=2:raise ValueError('Paired geometry mismatch')
    if clean.dtype!=np.float32 or corrupt.dtype!=np.float32:raise ValueError('Expected native float32 disparity')
    if not np.isfinite(clean).all() or not np.isfinite(corrupt).all():raise ValueError('Nonfinite disparity')
    delta=np.abs(corrupt.astype(np.float64)-clean.astype(np.float64))
    return {'delta_abs_sum':float(delta.sum(dtype=np.float64)),'pixels':int(delta.size),'delta_abs':float(delta.mean()),'invalid_total':0}


def pooled(rows):
    if not rows:raise ValueError('Cannot aggregate an empty scope')
    n=sum(r['pixels'] for r in rows)
    total=sum(r['delta_abs_sum'] for r in rows)
    return {'images':len(rows),'pixels':n,'delta_abs_sum':total,'delta_abs':total/n,'invalid_total':sum(r['invalid_total'] for r in rows)}


def summarize_condition(rows,condition):
    if any(r['condition']!=condition or r['eye'] not in ['left','right'] for r in rows):raise ValueError('Wrong condition/eye')
    keys=[(r['pair_id'],r['eye']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate pair/eye')
    pairs={r['pair_id'] for r in rows}
    if set(keys)!={(p,e) for p in pairs for e in ['left','right']}:raise ValueError('Incomplete left/right pairing')
    seq=[];total=[]
    for eye in ['left','right','both']:
        rr=[r for r in rows if eye=='both' or r['eye']==eye]
        total.append({'condition':condition,'eye':eye,**pooled(rr)})
        for scene in sorted({r['scene'] for r in rr}):
            seq.append({'condition':condition,'scene':scene,'eye':eye,**pooled([r for r in rr if r['scene']==scene])})
    return seq,total


def summarize_conditions(condition_rows,expected_conditions,clean_abs=None):
    if clean_abs is not None and (not np.isfinite(clean_abs) or clean_abs<0):raise ValueError('Invalid externally supplied clean Abs')
    completed=[c for c in expected_conditions if any(r['condition']==c for r in condition_rows)]
    if not completed:raise ValueError('No completed condition')
    if set(r['condition'] for r in condition_rows)-set(expected_conditions):raise ValueError('Unknown condition')
    macro=[]
    for eye in ['left','right','both']:
        rr=[r for r in condition_rows if r['eye']==eye]
        if len(rr)!=len(completed) or {r['condition'] for r in rr}!=set(completed):raise ValueError('Incomplete/duplicate condition-eye summary')
        delta=float(np.mean([r['delta_abs'] for r in rr]))
        proxy=None if clean_abs is None else .5*clean_abs/3.4545+.5*(clean_abs+delta)/18.4505
        macro.append({'scope':'twenty_condition_equal_macro' if len(completed)==len(expected_conditions) else 'completed_condition_equal_macro_partial',
                      'eye':eye,'completed_conditions':len(completed),'expected_conditions':len(expected_conditions),'delta_abs':delta,
                      'clean_abs_input':clean_abs,'E0_plus_native_delta_proxy':None if clean_abs is None else clean_abs+delta,
                      'RbS_from_native_delta_proxy_not_official':proxy,'official_server_score':None})
    return {'complete_all_conditions':len(completed)==len(expected_conditions),'completed_conditions':completed,
            'missing_conditions':[c for c in expected_conditions if c not in completed],'macro':macro}
