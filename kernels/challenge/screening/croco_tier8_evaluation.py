import numpy as np
from screening.croco_proxy_metrics import rank_select
from screening.croco_candidate_pruning import ensemble_risk
SIGNALS=['tile_disparity_std_unweighted','rank_head_lr_census']
COMPONENTS=['head_logscale_weighted','left_right_disparity_residual','reference_census_residual']
RAW=[SIGNALS[0],*COMPONENTS]
METHODS=['Identity',*[s+'/'+kind for s in SIGNALS for kind in ('global','pixel')]]
METHOD_SCOPES={'all':METHODS,'tile-global':['Identity',SIGNALS[0]+'/global'],
    'global':['Identity',*[s+'/global' for s in SIGNALS]]}

def checked_methods(methods):
    methods=list(methods)
    if methods not in METHOD_SCOPES.values():raise ValueError('unsupported output method scope')
    return methods

POLICY=dict(version='croco_tier8_same_policy_v1',signals=SIGNALS,methods=METHODS,
    global_rule='mean within-pixel average candidate rank; minimum global score',
    ensemble='mean normalized average ranks of head_logscale, LR, incoming Census; float32 before final ranking',
    support='reference x>=4; native left x>=4, native right x<=W-5; outside Identity',
    tie_seed=9137,global_tie='uniform seeded among scores within absolute1e-10; existing pair/domain/signal key',
    pixel_tie='first canonical candidate; Identity first',
    eyes='independent selection; right maps flipped to reference before selection, back to native after',
    clean_corrupt_routing='identical selector and bank; domain label used only in existing deterministic tie key',
    GT_used=False,clean_reference_used_in_selection=False)

def select_arrays(disparity,raw,pair_id,condition,candidate_ids,chunk=32768):
    """Only predictions/risks. Cannot access any clean reference, GT, or error."""
    if disparity.dtype!=np.float32 or disparity.ndim!=4 or disparity.shape[1]!=2:
        raise ValueError('expected K x two native eyes x H x W float32')
    if len(candidate_ids)!=disparity.shape[0] or candidate_ids[0]!='identity' or len(set(candidate_ids))!=len(candidate_ids):
        raise ValueError('canonical candidate inventory')
    if set(raw)!=set(RAW) or any(v.shape!=disparity.shape or v.dtype!=np.float32 for v in raw.values()):
        raise ValueError('raw risk inventory/shape/dtype')
    if not all(np.isfinite(v).all() for v in [disparity,*raw.values()]):raise ValueError('nonfinite cache')
    if disparity.shape[-1]<=4:raise ValueError('empty selection support')
    out={'Identity':disparity[0].copy()};info={}
    for signal in SIGNALS:
        glob=[];pixel=[];details=[]
        for eye in range(2):
            disp=disparity[:,eye] if eye==0 else disparity[:,eye,:,::-1]
            if signal==SIGNALS[0]:risk=raw[signal][:,eye] if eye==0 else raw[signal][:,eye,:,::-1]
            else:
                parts=[(raw[s][:,eye] if eye==0 else raw[s][:,eye,:,::-1]).reshape(len(candidate_ids),-1) for s in COMPONENTS]
                flat=np.empty_like(parts[0])
                for off in range(0,flat.shape[1],chunk):
                    flat[:,off:off+chunk]=ensemble_risk([p[:,off:off+chunk] for p in parts])
                risk=flat.reshape(disp.shape)
            result=rank_select(risk,disp,pair_id,condition,signal,tie_seed=POLICY['tie_seed'],chunk=chunk)
            glob.append(result['global_disparity'] if eye==0 else result['global_disparity'][:,::-1])
            pixel.append(result['pixel_disparity'] if eye==0 else result['pixel_disparity'][:,::-1])
            details.append(dict(eye=['left','right'][eye],global_candidate=candidate_ids[result['global_index']],
                scores=result['scores'].tolist(),pixel_fraction=result['pixel_fraction'].tolist(),
                selection_pixels=result['selection_pixels']))
        out[signal+'/global']=np.stack(glob);out[signal+'/pixel']=np.stack(pixel);info[signal]=details
    return out,info
