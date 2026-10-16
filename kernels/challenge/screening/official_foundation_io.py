"""CPU-only geometry, immutable .dsp5 IO and exact official test inventory."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import h5py
SEQUENCES={'0003':131,'0019':111,'0028':39,'0029':135,'0031':73,'0034':47,'0035':120,'0040':111,'0042':116,'0046':117}
CONDITIONS=['clean','brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_blur','gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','rain','saturate','shot_noise','snow','spatter','speckle_noise','zoom_blur']
SHAPE=(1080,1920)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def arr_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)
def observations(conditions=None):
 for condition in conditions or CONDITIONS:
  if condition not in CONDITIONS:raise ValueError(condition)
  for scene,n in SEQUENCES.items():
   for frame in range(1,n+1):yield {'condition':condition,'scene':scene,'frame':f'{frame:04d}','pair_id':f'{scene}/{frame:04d}'}
def rgb_path(root,row,eye):return Path(root)/row['condition']/'test'/row['scene']/('frame_'+eye)/f'frame_{eye}_{row["frame"]}.png'
def disp_path(root,row,eye):return Path(root)/row['condition']/'test'/row['scene']/('disp1_'+eye)/f'disp1_{eye}_{row["frame"]}.dsp5'
def reference_inputs(left,right,eye):
 if eye=='left':return left,right
 if eye!='right':raise ValueError(eye)
 # x'_R-x'_L = (W-1-x_R)-(W-1-x_L) = x_L-x_R >0.
 return np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1])
def native_output(disparity,eye):
 if eye=='left':return disparity
 if eye!='right':raise ValueError(eye)
 return np.ascontiguousarray(disparity[:,::-1])
def validate_disparity(d,shape=SHAPE):
 if d.shape!=shape or d.dtype!=np.float32:raise ValueError(f'disparity shape/dtype {d.shape}/{d.dtype}')
 if not np.isfinite(d).all():raise ValueError('NaN/Inf disparity; no drop/fill permitted')
 return {'shape':list(d.shape),'dtype':str(d.dtype),'invalid':0,'min':float(d.min()),'max':float(d.max()),'array_sha256':arr_sha(d)}
def write_dsp5(path,d,metadata,shape=SHAPE):
 p=Path(path);stats=validate_disparity(d,shape);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 tmp=p.with_suffix('.partial')
 with h5py.File(tmp,'w') as f:
  f.create_dataset('disparity',data=d,compression='gzip',compression_opts=5)
  # Official reader selects dataset; private reproducibility attrs do not alter it.
  f.attrs['provenance_json']=json.dumps(dict(metadata,array_sha256=stats['array_sha256']),sort_keys=True)
 os.replace(tmp,p)
 return dict(stats,file_sha256=sha(p),bytes=p.stat().st_size)
def read_dsp5(path,expected_metadata=None,shape=SHAPE):
 with h5py.File(path,'r') as f:
  if set(f.keys())!={'disparity'}:raise ValueError('unexpected .dsp5 datasets')
  d=f['disparity'][:];meta=json.loads(f.attrs['provenance_json'])
 stats=validate_disparity(d,shape)
 if meta.get('array_sha256')!=stats['array_sha256']:raise ValueError('cached disparity content hash mismatch')
 for key,value in (expected_metadata or {}).items():
  if meta.get(key)!=value:raise ValueError(f'stale output metadata: {key}')
 return d,meta

def validate_outputs(root,conditions=None,expected_metadata=None):
 root=Path(root);conditions=conditions or CONDITIONS;seen=set();rows=[]
 for row in observations(conditions):
  for eye in ['left','right']:
   p=disp_path(root,row,eye);d,meta=read_dsp5(p,dict(expected_metadata or {},condition=row['condition'],pair_id=row['pair_id'],reference_eye=eye))
   rows.append(dict(condition=row['condition'],pair_id=row['pair_id'],eye=eye,path=str(p),**validate_disparity(d),file_sha256=sha(p),bytes=p.stat().st_size));seen.add(p)
 actual={p for cond in conditions for p in (root/cond).rglob('*.dsp5')}
 if actual!=seen:raise ValueError(f'unexpected files: {actual-seen}')
 return rows
