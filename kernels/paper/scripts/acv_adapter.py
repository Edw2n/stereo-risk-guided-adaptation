"""Frozen official ACVNet, extracting attention and aggregated disparity logits."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/ACVNet'))
from models.acv import ACVNet


class ACVAdapter:
    def __init__(self,checkpoint,maxdisp=192):
        self.model=ACVNet(maxdisp,False,False)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict({k.removeprefix('module.'):v for k,v in state['model'].items()},strict=True)
        self.model.eval().cuda()
        self.native_signals={}
        self.collect_signals=True
        self.signal_events=[]
        self.model.classif_att_.register_forward_hook(self.capture('attention_'))
        self.model.classif2.register_forward_hook(self.capture(''))
        self.maxdisp=maxdisp

    def capture(self,prefix):
        def hook(module,inputs,output):
            if not self.collect_signals:
                return
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin.record()
            logits=output[:,0] # B,D,H,W, higher is better (official softmax convention).
            _,d,h,w=logits.shape
            ds=torch.arange(d,device=logits.device)[:,None,None]
            xs=torch.arange(w,device=logits.device)[None,None,:]
            valid=ds<=xs
            logits=logits.masked_fill(~valid,-torch.inf)
            count=valid.sum(0).float()
            top2=logits.topk(2,dim=1).values
            probabilities=torch.softmax(logits,dim=1)
            entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(1)/count.clamp_min(2).log()
            margin=top2[:,0]-top2[:,1]
            margin[:,:,0]=torch.nan
            entropy[:,:,0]=torch.nan
            self.native_signals.update({prefix+'negative_peak':-top2[:,0,None],
                prefix+'negative_margin':-margin[:,None],prefix+'entropy':entropy[:,None]})
            end.record()
            self.signal_events.append((begin,end))
        return hook

    @torch.inference_mode()
    def predict(self,left,right):
        disparity,signals,_=self.predict_batch([left],[right])
        return disparity[0],{k:v[0] for k,v in signals.items()}

    @torch.inference_mode()
    def predict_batch(self,left,right,collect_signals=True):
        if not len(left) or len(left)!=len(right):
            raise ValueError('Batch requires matching nonempty left/right lists')
        if not all(im.shape==left[0].shape and im.dtype==np.uint8
                   for im in list(left)+list(right)):
            raise ValueError('Batch requires equal-resolution uint8 RGB arrays')
        h,w,c=left[0].shape
        if c!=3:
            raise ValueError('Expected RGB inputs')
        self.native_signals.clear()
        self.signal_events.clear()
        self.collect_signals=collect_signals
        start=time.perf_counter()
        # Native model requires dimensions divisible by 16; official loader pads top/right with normalized zero.
        top=(-h)%16
        padright=(-w)%16
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        images=[]
        for ims in (left,right):
            tensor=torch.from_numpy(np.stack(ims)).permute(0,3,1,2).float().cuda()/255
            images.append(F.pad((tensor-mean)/std,(0,padright,top,0)))
        torch.cuda.synchronize()
        input_seconds=time.perf_counter()-start
        begin=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        model_start=time.perf_counter()
        begin.record()
        disparity=self.model(*images)[-1][:,top:top+h,:w]
        end.record()
        torch.cuda.synchronize()
        forward_wall=time.perf_counter()-model_start
        hook_seconds=sum(a.elapsed_time(b) for a,b in self.signal_events)/1000
        post_start=time.perf_counter()
        signals={k:F.interpolate(v,size=images[0].shape[-2:],mode='nearest')[:,0,top:top+h,:w]
                 for k,v in self.native_signals.items()}
        torch.cuda.synchronize()
        post_seconds=time.perf_counter()-post_start
        output_start=time.perf_counter()
        disparity_cpu=disparity.float().cpu().numpy()
        signals_cpu={k:v.float().cpu().numpy() for k,v in signals.items()}
        output_seconds=time.perf_counter()-output_start
        timing={'input_transfer_seconds':input_seconds,
                'model_inference_seconds':max(0,forward_wall-hook_seconds),
                'signal_extraction_seconds':hook_seconds+post_seconds,
                'output_transfer_seconds':output_seconds,
                'model_forward_gpu_seconds':begin.elapsed_time(end)/1000,
                'prediction_total_seconds':time.perf_counter()-start}
        self.native_signals.clear()
        self.signal_events.clear()
        return disparity_cpu,signals_cpu,timing
