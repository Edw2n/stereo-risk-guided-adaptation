"""Explicit AMP-forward FP16; all risk reductions/tile fusion/cache remain FP32.

Never globally half() the checkpoint or autocast the tiling/geometry/selector.
"""
import contextlib, time
import numpy as np
import torch
from screening.croco_proxy_extended_extract import ProxyAdapter as FP32ProxyAdapter, SIGNALS
from screening.official_foundation_io import write_json

PRECISION_PROTOCOL=dict(id='croco_amp_fp16_forward_fp32_risk_v1',weights='FP32 master weights',
    model_forward='torch.autocast(cuda, dtype=float16); tile model output immediately cast to float32',
    attention_risk='attention probabilities converted to float32; risk reductions with autocast disabled',
    decoder_risk='before/after features converted to float32; risk reductions with autocast disabled',
    tile_fusion='FP32 confidence weights, disparity accumulation, moments and risk-map accumulation',
    geometry='FP32 native disparity, LR and incoming Census; unchanged reference-eye conversion',
    storage='native float32 lossless; unchanged six cache planes',TF32=False)

class ForwardFP16:
    def __init__(self,model,enabled=True):self.model=model;self.head=model.head;self.enabled=enabled
    def __call__(self,*args):
        with torch.autocast('cuda',enabled=self.enabled,dtype=torch.float16):out=self.model(*args)
        return out.float()

class ProxyAdapter(FP32ProxyAdapter):
    def __init__(self,config,overlap=.7):
        super().__init__(config,overlap);self.fp16_enabled=True
    def _tiled(self,model,*args,**kwargs):
        # Same wrapper for plain and instrumented prediction: hooks cannot change disparity.
        return super()._tiled(ForwardFP16(model,self.fp16_enabled),*args,**kwargs)
    def _attention_hook(self,module,args,prob):
        if not self.active:return
        with torch.autocast('cuda',enabled=False):
            return super()._attention_hook(module,args,prob.float())
    def _decoder_hook(self,module,args,output):
        if not self.active:return
        with torch.autocast('cuda',enabled=False):
            # Parent reads only args[0]/output[0]; preserve other fields unchanged.
            return super()._decoder_hook(module,(args[0].float(),*args[1:]),(output[0].float(),*output[1:]))
