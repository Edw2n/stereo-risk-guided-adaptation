"""Side-channel CroCo signals; preserve upstream disparity arithmetic and ordering."""
import math,time
import numpy as np
import torch
import torch.nn.functional as F
from screening.batch_adapter import BatchAdapter
SIGNALS=['head_uncertainty_raw_weighted','head_logscale_weighted','tile_disparity_std','cross_attention_entropy','cross_attention_negative_margin']

class ProxyAdapter(BatchAdapter):
    def __init__(self,config,overlap=.7):
        super().__init__(config)
        self.overlap=overlap;self.original_tiled=self.base.tiled;self.active=False;self.attention=None;self.last_risks=None
        self.handle=self.base.model.dec_blocks[-1].cross_attn.attn_drop.register_forward_hook(self._attention_hook)
        self.base.tiled=self._tiled

    def _attention_hook(self,module,args,prob):
        if not self.active:return
        # Reduce one attention head at a time; don't retain B x heads x N x N copies.
        entropy=torch.zeros_like(prob[:,0,:,0]);margin=torch.zeros_like(entropy)
        for i in range(prob.shape[1]):
            p=prob[:,i]
            entropy+=-(p*torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum(-1)/math.log(p.shape[-1])
            top=torch.topk(p,2,dim=-1).values
            margin-=top[...,0]-top[...,1]
        entropy/=prob.shape[1];margin/=prob.shape[1]
        self.attention=torch.stack([entropy,margin],1)

    def _tiled(self,model,criterion,img1,img2,gt,**kwargs):
        kwargs['overlap']=self.overlap
        if not self.active:return self.original_tiled(model,criterion,img1,img2,gt,**kwargs)
        assert criterion is None and gt is None and kwargs['with_conf'] and kwargs['conf_mode']=='conf_expsigmoid_15_3'
        assert kwargs['crop']==[352,704] and img1.shape[-2:]==(1080,1920)
        from stereoflow.engine import _overlapping,split_prediction_conf
        B,_,H,W=img1.shape;th,tw=kwargs['crop']
        pred_sum=img1.new_zeros((B,1,H,W));weights=img1.new_zeros((B,H,W))+1e-16
        raw_sum=img1.new_zeros((B,H,W));log_sum=torch.zeros_like(raw_sum)
        att_sum=img1.new_zeros((B,2,H,W));variance_mean=torch.zeros_like(raw_sum);variance_m2=torch.zeros_like(raw_sum)
        coverage=torch.zeros((H,W),device=img1.device,dtype=torch.int16)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        for sy in _overlapping(H,th,self.overlap):
            for sx in _overlapping(W,tw,self.overlap):
                self.attention=None
                pred,u=split_prediction_conf(model(img1[...,sy,sx],img2[...,sy,sx]),with_conf=True)
                assert self.attention is not None and self.attention.shape==(B,2,22*44)
                weight=torch.exp(-15*2*(torch.sigmoid(u/3)-.5)).view(B,th,tw)
                # Upstream operations: do not change dtype, crop order, or summation order.
                pred_sum[...,sy,sx]+=pred*weight[:,None]
                weights[...,sy,sx]+=weight
                raw_sum[...,sy,sx]+=u.view(B,th,tw)*weight
                log_sum[...,sy,sx]+=6*(torch.sigmoid(u.view(B,th,tw)/3)-.5)*weight
                # Independent stable weighted population variance; no impact on pred_sum.
                d=pred[:,0];mu=variance_mean[...,sy,sx];delta=d-mu
                new_mu=mu+weight/weights[...,sy,sx]*delta
                variance_m2[...,sy,sx]+=weight*delta*(d-new_mu)
                variance_mean[...,sy,sx]=new_mu
                a=F.interpolate(self.attention.reshape(B,2,22,44),size=(th,tw),mode='bilinear',align_corners=False)
                att_sum[...,sy,sx]+=a*weight[:,None]
                coverage[sy,sx]+=1
        disparity=pred_sum/weights[:,None];raw=raw_sum/weights
        std=torch.sqrt(torch.clamp(variance_m2/weights,min=0))
        std[:,coverage<2]=0 # no disagreement information: all candidates tie; fusion retains Identity.
        risks=torch.stack([raw,log_sum/weights,std,att_sum[:,0]/weights,att_sum[:,1]/weights],1)
        assert torch.isfinite(disparity).all() and torch.isfinite(risks).all()
        end.record();torch.cuda.synchronize();seconds=start.elapsed_time(end)/1000
        self.last_risks=risks.float().cpu().numpy().copy();self.coverage=coverage.cpu().numpy().copy()
        self.attention=None
        # Upstream's no-GT loss is undefined; return explicit None, never use it as a signal.
        return disparity,None,raw,seconds

    def predict_with_risks(self,left,right):
        self.active=True;self.last_risks=None;start=time.monotonic()
        try:
            pred,timing=super().predict_batch(left,right)
            assert self.last_risks is not None
            risks=self.last_risks;self.last_risks=None
            timing['instrumented_end_to_end_seconds']=time.monotonic()-start
            return pred,risks,timing
        finally:self.active=False

    def predict_plain(self,left,right):
        self.active=False
        return super().predict_batch(left,right)
