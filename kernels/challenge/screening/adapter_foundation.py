import sys,time,hashlib
from pathlib import Path
import torch,numpy as np
class Adapter:
 def __init__(self,c):
  sys.path.insert(0,c['upstream']['path']);self.c=c
  from omegaconf import OmegaConf
  from core.foundation_stereo import FoundationStereo
  from core.utils.utils import InputPadder
  cfgfile=c['inference']['cfg_yaml'];assert hashlib.sha256(Path(cfgfile['path']).read_bytes()).hexdigest()==cfgfile['sha256']
  args=OmegaConf.load(cfgfile['path'])
  args.update(dict(vit_size='vitl',valid_iters=32,hiera=1,scale=1,low_memory=0))
  assert args.max_disp==416
  # Avoid redundant classifier initialization download; every model key is replaced by strict checkpoint load.
  import timm
  create=timm.create_model
  def no_pretrain(*a,**kw):kw['pretrained']=False;return create(*a,**kw)
  timm.create_model=no_pretrain
  # Pin the exact historical torch-hub source; only lookup location changes.
  hub_load=torch.hub.load
  def local_hub(repo,model,*a,**kw):
   if repo=='facebookresearch/dinov2':
    repo=c['dinov2_source'];kw['source']='local'
   return hub_load(repo,model,*a,**kw)
  torch.hub.load=local_hub
  try:self.model=FoundationStereo(args)
  finally:
   timm.create_model=create;torch.hub.load=hub_load
  ck=torch.load(c['checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
  self.model.load_state_dict(ck['model'],strict=True);self.model.eval().cuda();self.padder=InputPadder
 def predict(self,left,right):
  tensors=[torch.from_numpy(x.copy()).cuda().float().permute(2,0,1)[None] for x in (left,right)]
  pad=self.padder(tensors[0].shape,divis_by=32,force_square=False);a,b=pad.pad(*tensors);torch.cuda.synchronize();t=time.monotonic()
  with torch.inference_mode(),torch.cuda.amp.autocast(enabled=True,dtype=torch.float16):d=self.model.run_hierachical(a,b,iters=32,test_mode=True,small_ratio=.5)
  d=pad.unpad(d.float());torch.cuda.synchronize()
  return d[0,0].cpu().numpy().copy(),{'inference_seconds':time.monotonic()-t}
