"""Shared coordinates and scene-sharded sampled analysis cache; no dense maps.

Coordinates depend only on GT validity, fixed geometry, pair ID and pool seed.
Default nested subsets are prefixes of the saved pool. A new subset seed can
permute the saved pool, but cannot create independent unseen native pixels.
"""
import json
from pathlib import Path
import threading
import h5py
import numpy as np
from sensor_v4_common import seed, digest

_LOCK=threading.RLock()


def pool_coordinates(valid, pair_id, max_pixels, pool_seed):
    domain=np.asarray(valid,dtype=bool).copy();domain[:,:4]=False
    eligible=np.flatnonzero(domain)
    # One permutation, independent of requested pool length, gives nested pools.
    ids=np.random.default_rng(seed('table2-max-pool',pool_seed,pair_id)).permutation(eligible)
    return ids[:min(max_pixels,len(ids))].astype(np.uint32)


def subset_indices(pool_size, pixels, pair_id, subset_seed=0):
    if pixels>pool_size:
        raise ValueError(f'Requested {pixels} pixels but cache contains only {pool_size}; no inference is launched')
    if pixels<1:raise ValueError('Sample size must be positive')
    if subset_seed==0:return np.arange(pixels,dtype=np.int64)
    return np.random.default_rng(seed('table2-nested-subset',subset_seed,pair_id)).permutation(pool_size)[:pixels]


def get_shared_coordinates(root, record, valid, config):
    cache=config['metrics']['analysis_cache'];path=Path(root)/'coordinates.h5'
    key=record['pair_id'];domain=np.asarray(valid,dtype=bool).copy();domain[:,:4]=False
    attrs={'pool_seed':cache['pool_seed'],'max_pool_pixels':cache['max_pool_pixels'],
           'domain_sha256':digest(domain.tobytes()),'shape':json.dumps(list(valid.shape))}
    with _LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(path,'a') as f:
            if key in f:
                d=f[key]
                if any(d.attrs.get(k)!=v for k,v in attrs.items()):raise ValueError('Coordinate identity/mask mismatch')
                return d[:]
            ids=pool_coordinates(valid,key,cache['max_pool_pixels'],cache['pool_seed'])
            d=f.create_dataset(key,data=ids,dtype='u4')
            for k,v in attrs.items():d.attrs[k]=v
            return ids


def write_pool(root, record, model, condition, coords, error, d1, scores, config):
    root=Path(root);cache=config['metrics']['analysis_cache']
    names=list(scores);n=len(coords)
    if not n or not names:raise ValueError('Empty pool or signal list')
    values=np.stack([np.asarray(scores[k]) for k in names],axis=-1).astype(np.float32)
    err=np.asarray(error,dtype=np.float32)
    if not np.array_equal(err.astype(np.float64),np.asarray(error,dtype=np.float64)):
        raise ValueError('Float32 cache would lose absolute-error precision')
    if not np.isfinite(err).all() or not np.isfinite(values).all():raise ValueError('Nonfinite analysis sample')
    path=root/model/f'{record["scene"]}.h5';key=f'{record["frame"]}/{condition}'
    with _LOCK:
        with h5py.File(root/'coordinates.h5','r') as f:
            if not np.array_equal(f[record['pair_id']][:],coords):raise ValueError('Writer coordinates disagree with shared pool')
        path.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(path,'a') as f:
            if key in f:raise FileExistsError('Sample pool already exists; never overwrite')
            g=f.create_group(key);g.attrs['complete']=False
            g.attrs['pair_id']=record['pair_id'];g.attrs['split']=record['split'];g.attrs['model']=model
            g.attrs['condition']=condition;g.attrs['signal_names']=json.dumps(names)
            g.attrs['scene']=record['scene'];g.attrs['frame']=record['frame']
            g.attrs['config_sha256']=digest(config)
            g.attrs['checkpoint_sha256']=config['models'][model]['checkpoint_sha256']
            g.attrs['coordinate_sha256']=digest(np.asarray(coords,dtype=np.uint32).tobytes())
            chunk=min(cache['chunk_pixels'],n)
            compression=cache['compression'] if cache['compression']!='none' else None
            opts={'compression':compression,'shuffle':compression is not None}
            g.create_dataset('scores',data=values,chunks=(chunk,len(names)),**opts)
            g.create_dataset('error',data=err,chunks=(chunk,),**opts)
            g.create_dataset('d1',data=np.asarray(d1,dtype=np.uint8),chunks=(chunk,),**opts)
            g.attrs['complete']=True
    return {'format':'scene_hdf5_pool_v1','shard':str(path),'key':key,
            'coordinates':str(root/'coordinates.h5'),'coordinate_key':record['pair_id'],
            'pool_pixels':n,'signal_names':names}


def save_common_support(root,record,coords,common):
    """One GT-only mask per pair, not one per model/condition; no resampling."""
    values=np.asarray(common,dtype=bool).ravel()[coords].astype(np.uint8)
    with _LOCK:
        with h5py.File(Path(root)/'coordinates.h5','a') as f:
            key='__common_support__/'+record['pair_id']
            if key in f:
                if not np.array_equal(f[key][:],values):raise ValueError('Shared common-support mismatch')
            else:f.create_dataset(key,data=values,dtype='u1')


def load_samples(row, pixels=None, subset_seed=0):
    if 'analysis_pool' not in row:
        with np.load(row['sample_path'],allow_pickle=False) as d:data={k:d[k] for k in d.files}
        n=len(data['error']);idx=subset_indices(n,pixels or n,row['pair_id'],subset_seed)
        return {k:v[idx] for k,v in data.items()}
    ref=row['analysis_pool'];n=ref['pool_pixels'];pixels=pixels or n
    idx=subset_indices(n,pixels,row['pair_id'],subset_seed)
    with h5py.File(ref['shard'],'r') as f:
        g=f[ref['key']]
        if not g.attrs['complete']:raise ValueError('Incomplete cache observation')
        names=json.loads(g.attrs['signal_names'])
        # Prefix reads touch only relevant chunks. Non-prefix permutations load
        # the small max pool and gather in RAM, avoiding costly HDF5 point I/O.
        cut=slice(0,pixels) if subset_seed==0 else slice(None)
        score=g['scores'][cut];err=g['error'][cut];d1=g['d1'][cut].astype(bool)
        coordinate_hash=g.attrs['coordinate_sha256']
    with h5py.File(ref['coordinates'],'r') as f:
        coords=f[ref['coordinate_key']][:]
        key='__common_support__/'+ref['coordinate_key']
        common=f[key][:].astype(bool) if key in f else None
    if digest(coords.astype(np.uint32).tobytes())!=coordinate_hash:raise ValueError('Wrong shared coordinates')
    if subset_seed!=0:score,err,d1=score[idx],err[idx],d1[idx]
    data={'error':err,'d1':d1,'pixel_index':coords[idx]}
    if common is not None:data['common_support']=common[idx]
    data.update({name:score[:,i] for i,name in enumerate(names)})
    return data
