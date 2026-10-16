import hashlib
import numpy as np
from scipy.stats import rankdata, spearmanr

def stable_rng(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8],'little'))


def census(rgb):
    gray=np.dot(rgb.astype(np.float32),np.array([.299,.587,.114],np.float32));h,w=gray.shape
    padded=np.pad(gray,2,mode='edge');out=np.zeros((h,w),np.uint32);k=0
    for dy in range(-2,3):
        for dx in range(-2,3):
            if dy==0 and dx==0:continue
            out|=((padded[2+dy:2+dy+h,2+dx:2+dx+w]<gray).astype(np.uint32)<<k);k+=1
    return out


def geometry_baselines(d,right,left_census,right_census):
    h,w=d.shape;x=np.arange(w)[None]-d;y=np.arange(h)[:,None]
    valid=(x>=0)&(x<=w-1)
    xc=np.clip(x,0,w-1);x0=np.floor(xc).astype(np.int32);x1=np.minimum(x0+1,w-1);a=xc-x0
    r=right[y,x0]*(1-a)+right[y,x1]*a
    lr=np.abs(d-r).astype(np.float32)
    assert np.isfinite(lr).all() and lr.max()<1e9
    lr[~valid]=1e9 # structural out-of-view, not a silently excluded error.
    codes=left_census^right_census[y,np.rint(xc).astype(np.int32)]
    ham=np.zeros((h,w),np.float32)
    for b in range(24):ham+=((codes>>b)&1)
    ham/=24;ham[~valid]=1
    return lr,ham,int((~valid).sum())


def rank_select(risk,disparity,pair_id,domain,signal,tie_seed=9137,chunk=32768):
    """Full x>=4 geometry, no GT argument. Equal ranks; global ties uniform seeded.

    Pixel minimum-risk ties use canonical first ID (Identity first).
    """
    k,h,w=risk.shape;assert disparity.shape==(k,h,w)
    support=np.broadcast_to(np.arange(w)[None]>=4,(h,w)).ravel()
    positions=np.flatnonzero(support);scores=np.zeros(k,np.float64)
    pix=disparity[0].copy();assign=np.zeros(h*w,np.uint8)
    rf=risk.reshape(k,-1);df=disparity.reshape(k,-1)
    for off in range(0,len(positions),chunk):
        ii=positions[off:off+chunk];v=np.asarray(rf[:,ii]);assert np.isfinite(v).all()
        scores+=rankdata(v,axis=0,method='average').sum(axis=1)
        winner=np.argmin(v,axis=0);assign[ii]=winner;pix.ravel()[ii]=df[winner,ii]
    scores/=len(positions)
    tied=np.flatnonzero(np.isclose(scores,scores.min(),rtol=0,atol=1e-10))
    selected=int(stable_rng('global_score_tie',tie_seed,pair_id,domain,signal).choice(tied))
    glob=disparity[0].copy();glob.ravel()[positions]=df[selected,positions]
    return dict(global_disparity=glob,pixel_disparity=pix,global_index=selected,scores=scores,
                pixel_fraction=np.bincount(assign[positions],minlength=k)/len(positions),selection_pixels=len(positions))


def rank_metrics(risk,error,d1):
    from sklearn.metrics import roc_auc_score, average_precision_score
    risk=np.asarray(risk);error=np.asarray(error);d1=np.asarray(d1)
    assert np.isfinite(risk).all() and np.isfinite(error).all()
    both=np.unique(d1).size==2
    return dict(AUROC=float(roc_auc_score(d1,risk)) if both else None,
                AP=float(average_precision_score(d1,risk)) if d1.any() else None,
                Spearman=float(spearmanr(risk,error).statistic) if np.ptp(risk)>0 and np.ptp(error)>0 else None,
                prevalence=float(d1.mean()))


def candidate_ordering(risk,error):
    # [K,P], image-equal valid nonconstant candidate-axis correlations.
    a=rankdata(risk,axis=0);b=rankdata(error,axis=0);a-=a.mean(0);b-=b.mean(0)
    denom=np.sqrt((a*a).sum(0)*(b*b).sum(0));valid=denom>0
    rho=np.divide((a*b).sum(0),denom,out=np.zeros_like(denom),where=valid)
    best=np.argmin(risk,axis=0);picked=error[best,np.arange(error.shape[1])]
    concordant=discordant=ties=0
    for i in range(len(risk)):
        for j in range(i):
            sr=np.sign(risk[i]-risk[j]);se=np.sign(error[i]-error[j]);mask=se!=0
            concordant+=int(((sr==se)&mask).sum());discordant+=int(((sr==-se)&mask).sum());ties+=int(((sr==0)&mask).sum())
    den=concordant+discordant+ties
    return dict(candidate_Spearman=float(rho[valid].mean()) if valid.any() else None,
                nonconstant_pixel_fraction=float(valid.mean()),concordance=(concordant+.5*ties)/den if den else None,
                top1_abs_regret=float((picked-error.min(0)).mean()),pixels=error.shape[1])

