"""Batch wrappers preserve each approved model's forward, normalization and padding."""
import time
import numpy as np
import torch
from screening.candidate_adapters import load_adapter
class BatchAdapter:
 def __init__(self,c):self.c=c;self.base=load_adapter(c)
 def predict_batch(self,left,right):
  assert len(left)==len(right)>0
  for x in left+right:assert x.shape==(1080,1920,3) and x.dtype==np.uint8
  model=self.c['id'];base=self.base
  def tensor(xs):
   tensors=[]
   for x in xs:
    if model=='croco':
     # Preserve the approved adapter's per-image operation order AND B1 strides.
     t=torch.from_numpy(x.copy()).permute(2,0,1).float().cuda()/255
     t=((t-t.new_tensor([.485,.456,.406])[:,None,None])/t.new_tensor([.229,.224,.225])[:,None,None])[None]
    else:t=torch.from_numpy(x.copy()).cuda().float().permute(2,0,1)[None]
    tensors.append(t)
   return tensors[0] if len(tensors)==1 else torch.cat(tensors,dim=0)
  a,b=tensor(left),tensor(right)
  if model=='croco':pass
  else:
   kw={'divis_by':32}
   if model=='foundation':kw['force_square']=False
   pad=base.padder(a.shape,**kw);a,b=pad.pad(a,b)
  torch.cuda.synchronize();start=time.monotonic()
  with torch.inference_mode():
   if model=='croco':d,_,_,_=base.tiled(base.model,None,a,b,None,overlap=.9,crop=[352,704],conf_mode='conf_expsigmoid_15_3',with_conf=True,return_time=True)
   elif model=='foundation':
    with torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):d=base.model.run_hierachical(a,b,iters=32,test_mode=True,small_ratio=.5)
   elif model=='igev_plusplus':d=base.model(a,b,iters=16,test_mode=True)
   elif model=='defom':d=base.model(a,b,iters=32,scale_iters=8,test_mode=True)
   else:raise ValueError(model)
  if model!='croco':d=pad.unpad(d.float())
  torch.cuda.synchronize();seconds=time.monotonic()-start
  out=d[:,0].float().cpu().numpy().copy();assert out.shape==(len(left),1080,1920) and np.isfinite(out).all()
  return out,{'inference_seconds':seconds,'batch':len(left),'timing_scope':'padded model inference, excludes RGB stacking/H2D/D2H/I/O/metrics'}
