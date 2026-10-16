import json, time, threading
import numpy as np, h5py
from sensor_v4_common import ROOT, digest, storage
from sensor_v4_metrics import native_maps, sufficient, native_values
from sensor_v4_rank_selection import _pixel_signal_fusion, selection_support
from sensor_v4_signals import select_without_gt, oracle_diagnostic
from broad_sweep_v4 import exact_counts, select_scores
from candidate_adequacy_v4 import rank_scores_from_counts, geometry_pool
LOCK=threading.Lock()
RUN='reproduction'
METHODS=['Clean','Identity','Random','Signal-Global','Signal-Pixel Fusion','Global Oracle','Pixel Oracle']

def pixel_oracle_heldout(disparities,gt):
 """Evaluation-only absolute-error argmin, lexical ID ties, identical native rule."""
 ids=sorted(disparities);shape=disparities['identity'].shape;best=np.full(shape,np.inf,np.float64);out=disparities['identity'].copy();chosen=np.full(shape,ids.index('identity'),np.int16)
 for i,c in enumerate(ids):
  e,_,valid,_=native_maps(disparities[c],gt);take=valid&(e<best);best[take]=e[take];out[take]=disparities[c][take];chosen[take]=i
 return out,chosen


def evaluate(bundle,cfg,bank,frozen,approval,root,so):
 if approval.get('heldout_pixel_fusion_approved') is not True or bundle['record']['split']!='test':raise PermissionError('Held-out fusion is explicitly gated')
 start=time.monotonic();r=bundle['record'];model=bundle['model'];condition=bundle['condition'];signal=frozen['primary_signals'][model];ids=sorted(bundle['risks'])
 assert set(ids)=={c['id'] for c in bank['candidates']}
 rec={**r,'incoming_condition':condition,'model':model,'primary_signal':signal}
 counts=exact_counts(bundle['risks'],cfg,so);decision=select_scores(rank_scores_from_counts(counts,ids),rec,cfg)
 random=select_without_gt(ids,{},'Random',None,rec,cfg)
 fusion,assignment,fusion_info=_pixel_signal_fusion(bundle['disparities'],bundle['risks'],rec,cfg)
 fusion_info['scope']='explicitly approved held-out evaluation'
 # Actual selectors above have no GT access.
 with h5py.File(ROOT/r['assets']['disp1_left']['path'],'r') as f:gt=f['disparity'][:]
 indices=geometry_pool(r['pair_id'],fusion.shape,cfg);support=selection_support(fusion.shape,cfg)
 def stats(pred):
  e,d,v,common=native_maps(pred,gt)
  return {'full':sufficient(e,d,v),'common_support':sufficient(e,d,common),'shared_signal_support':sufficient(e,d,v&support)},e,d,v,common
 candidate_stats={};sample_error=[];sample_d1=[]
 for cid in ids:
  st,e,d,v,common=stats(bundle['disparities'][cid]);candidate_stats[cid]=st;sample_error.append(e.ravel()[indices].astype(np.float32));sample_d1.append(d.ravel()[indices])
 valid_sample=v.ravel()[indices];common_sample=common.ravel()[indices]
 oracle=oracle_diagnostic({c:native_values(st['full']) for c,st in candidate_stats.items()});best,oracle_assignment=pixel_oracle_heldout(bundle['disparities'],gt)
 picks={'Identity':'identity','Random':random,'Signal-Global':decision['candidate_id'],'Global Oracle':oracle}
 methods=[{'method':name,'candidate_id':cid,**candidate_stats[cid]} for name,cid in picks.items()]
 for name,pred in [('Clean',bundle['clean']),('Signal-Pixel Fusion',fusion),('Pixel Oracle',best)]:methods.append({'method':name,'candidate_id':{'Clean':'clean','Signal-Pixel Fusion':'spatial_fusion','Pixel Oracle':'spatial_oracle'}[name],**stats(pred)[0]})
 maps=storage(cfg,'intermediates',RUN,'analysis_pool');path=maps/f'{model}_{r["scene"]}.h5';key=r['frame']+'/'+condition;fingerprint=digest({'config':cfg,'inputs':bundle['input_sha256'],'model':model})
 with LOCK,h5py.File(path,'a') as f:
  if key in f:assert f[key].attrs['fingerprint']==fingerprint
  else:
   temp='_pending_'+str(time.time_ns());g=f.create_group(temp);g.attrs['fingerprint']=fingerprint;g.attrs['scope']='held-out geometry-only shared pool; not Table2 pool';g.attrs['candidate_ids']=json.dumps(ids)
   for name,a in [('coordinates',indices),('valid',valid_sample),('common_support',common_sample),('disparity',np.stack([bundle['disparities'][c].ravel()[indices] for c in ids])),('risk',np.stack([bundle['risks'][c].ravel()[indices] for c in ids])),('absolute_error',np.stack(sample_error)),('D1',np.stack(sample_d1))]:g.create_dataset(name,data=a,compression='lzf')
   f.require_group(r['frame']);f.move(temp,key);f.flush()
 dense=None
 if r['pair_id'] in cfg['heldout_run']['qualitative_pair_ids']:
  folder=storage(cfg,'intermediates',RUN,'qualitative',model);dense=folder/f'{r["scene"]}_{r["frame"]}.npz'
  if not dense.exists():
   temp=dense.with_name(dense.stem+'.partial.npz')
   with temp.open('wb') as f:np.savez(f,candidate_ids=ids,disparity=np.stack([bundle['disparities'][c] for c in ids]),risk=np.stack([bundle['risks'][c] for c in ids]),clean=bundle['clean'],signal_fusion=fusion,pixel_oracle=best,signal_assignment=assignment,oracle_assignment=oracle_assignment)
   temp.rename(dense)
 shared=v&support
 return {'model':model,'pair_id':r['pair_id'],'scene':r['scene'],'split':'test','condition':condition,'assignment_seed':101,'config_sha256':digest(cfg),'freeze_sha256':digest(frozen),'checkpoint_sha256':cfg['models'][model]['checkpoint_sha256'],'signal':signal,'input_sha256':bundle['input_sha256'],'RGB_cache_path':bundle['RGB_cache_path'],'global_rank_decision':decision,'pairwise_rank_counts':counts,'candidate_statistics':candidate_stats,'picks':picks,'methods':methods,'pixel_fusion':fusion_info,'assignment_agreement':float((assignment[shared]==oracle_assignment[shared]).mean()),'pixel_oracle_objective':'minimum native absolute error, lexical candidate ID ties','pool':{'path':str(path),'group':key,'pixels':len(indices)},'dense_qualitative_path':str(dense) if dense else None,'GT_used_by_signal':False,'timing':{**bundle['timing'],'CPU_evaluation_seconds':time.monotonic()-start}}

