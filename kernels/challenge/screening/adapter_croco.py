"""Frozen official inference adapters. No GT argument; batch 1 native inference."""
import sys,argparse,time
from pathlib import Path
import numpy as np
import torch
class CroCo:
 def __init__(self,config):
  self.c=config;sys.path.insert(0,config['upstream']['path'])
  from models.croco_downstream import CroCoDownstreamBinocular
  from models.head_downstream import PixelwiseTaskWithDPT
  from stereoflow.engine import tiled_pred
  torch.serialization.add_safe_globals([argparse.Namespace])
  ck=torch.load(config['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
  a=ck['args'];assert a.task=='stereo' and list(a.crop)==[352,704] and a.tile_conf_mode=='conf_expsigmoid_15_3'
  head=PixelwiseTaskWithDPT();head.num_channels=2
  self.model=CroCoDownstreamBinocular(head,**a.croco_args)
  self.model.load_state_dict(ck['model'],strict=True);self.model.eval().cuda();self.tiled=tiled_pred
 def predict(self,left,right):
  def tensor(x):
   assert x.shape==(1080,1920,3) and x.dtype==np.uint8
   t=torch.from_numpy(x.copy()).permute(2,0,1).float().cuda()/255
   return ((t-t.new_tensor([.485,.456,.406])[:,None,None])/t.new_tensor([.229,.224,.225])[:,None,None])[None]
  a,b=tensor(left),tensor(right);torch.cuda.synchronize();start=time.monotonic()
  with torch.inference_mode():
   pred,_,_,gpu_seconds=self.tiled(self.model,None,a,b,None,overlap=.9,crop=[352,704],conf_mode='conf_expsigmoid_15_3',with_conf=True,return_time=True)
  torch.cuda.synchronize();out=pred[0,0].float().cpu().numpy().copy()
  return out,{'inference_seconds':time.monotonic()-start,'gpu_seconds':gpu_seconds,'tile_forwards':418}

Adapter=CroCo
