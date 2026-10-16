import os, json, uuid
from pathlib import Path
import numpy as np, h5py
from screening.official_foundation_io import native_output, reference_inputs, arr_sha, sha
from screening.croco_proxy_metrics import census, geometry_baselines
RAW_SIGNALS=['head_logscale_weighted','tile_disparity_std_unweighted','decoder_final_relative_update','left_right_disparity_residual','reference_census_residual']
INTERNAL_INDICES=[1,14,12]

def assemble_native(disparity,internal,left,right):
    """Both disparity arrays start in model reference coordinates, not native right."""
    cl,cr=census(left),census(right)
    native_d=np.stack([disparity[0],native_output(disparity[1],'right')])
    maps={s:np.empty_like(native_d) for s in RAW_SIGNALS}
    for eye in range(2):
        # Opposite-eye map in the same horizontal frame as this reference.
        opposite=native_d[1] if eye==0 else native_d[0][:,::-1]
        lc,rc=(cl,cr) if eye==0 else (cr[:,::-1],cl[:,::-1])
        lr,ham,_=geometry_baselines(disparity[eye],opposite,lc,rc)
        values=[*internal[eye],lr,ham]
        for signal,v in zip(RAW_SIGNALS,values):
            maps[signal][eye]=v if eye==0 else v[:,::-1]
    return {'disparity':native_d,**maps}


def verify_cache(path,expected=None):
    with h5py.File(path,'r') as f:
        meta=json.loads(f.attrs['provenance_json'])
        for k,v in (expected or {}).items():
            if meta.get(k)!=v:raise ValueError('cache provenance mismatch: '+k)
        if set(f.keys())!={'disparity',*RAW_SIGNALS}:raise ValueError('cache dataset inventory')
        for name in f:
            a=f[name][:]
            if a.dtype!=np.float32 or list(a.shape)!=meta['array_shape'] or not np.isfinite(a).all():
                raise ValueError('cache shape/dtype/nonfinite: '+name)
            if arr_sha(a)!=f[name].attrs['array_sha256']:raise ValueError('cache content hash: '+name)
    return dict(path=str(path),bytes=Path(path).stat().st_size,file_sha256=sha(path),
                pair_id=meta['pair_id'],candidate_id=meta['candidate_id'],invalid=0)


def save_cache(path,arrays,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    meta={**meta,'array_shape':list(arrays['disparity'].shape)}
    with h5py.File(tmp,'x') as f:
        f.attrs['provenance_json']=json.dumps(meta,sort_keys=True)
        for name,a in arrays.items():
            a=np.ascontiguousarray(a,dtype=np.float32)
            if a.shape!=arrays['disparity'].shape or not np.isfinite(a).all():raise ValueError('invalid map')
            ds=f.create_dataset(name,data=a,chunks=(1,min(128,a.shape[1]),min(256,a.shape[2])),compression='lzf',shuffle=True)
            ds.attrs['array_sha256']=arr_sha(a)
    # Verify every map before making this paired, candidate-atomic cache visible.
    receipt=verify_cache(tmp,meta)
    os.link(tmp,path) # exclusive final publication; never overwrite existing cache
    tmp.unlink();receipt['path']=str(path)
    return receipt


def pack_reference_inputs(items,reference_batch_size):
    """Pad a partial final group to its fixed batch; discard duplicate outputs."""
    left=[];right=[]
    for _,_,_,a,b,_ in items:
        for eye in ('left','right'):
            l,r=reference_inputs(a,b,eye);left.append(l);right.append(r)
    actual=len(left)
    if not actual or actual>reference_batch_size or actual%2 or reference_batch_size%2:
        raise ValueError('paired batch contract')
    while len(left)<reference_batch_size:
        left.extend(left[actual-2:actual]);right.extend(right[actual-2:actual])
    return left,right,reference_batch_size-actual

