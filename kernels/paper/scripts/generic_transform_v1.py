"""Frozen joint RGB transformation; parameters are in configs/paper/bank.json."""
import cv2, numpy as np
from sensor_v4_images import srgb_to_linear, quantize
cv2.setNumThreads(1)
PARAMETERS=['exposure','contrast','gamma','saturation','detail']
ORDER=['linear_exposure','contrast','clip_before_gamma','gamma','saturation','detail','final_clip_quantize']

def transform(rgb,candidate,*,trace=False):
 """Float operations throughout; exposure highlights clipped only before gamma."""
 if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('uint8 HWC RGB required')
 params=candidate['parameters'];stages=[]
 if candidate['id']=='identity':
  if any(params[k]!=0 for k in PARAMETERS):raise ValueError('Identity parameters must be zero')
  return (rgb.copy(),['identity_exact_copy']) if trace else rgb.copy()
 x=rgb.astype(np.float32)/255
 linear=srgb_to_linear(x)*np.float32(2**params['exposure'])
 # Deliberately no clip in inverse transfer: preserve highlights until gamma preclip.
 x=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055).astype(np.float32)
 stages.append(ORDER[0])
 center=x.mean(axis=(0,1),dtype=np.float64,keepdims=True)
 x=(x-center)*2**params['contrast']+center;stages.append(ORDER[1])
 x=np.clip(x,0,1).astype(np.float32);stages.append(ORDER[2])
 x=np.power(x,np.float32(2**params['gamma']));stages.append(ORDER[3])
 hsv=cv2.cvtColor(x,cv2.COLOR_RGB2HSV);hsv[...,1]=np.clip(hsv[...,1]*np.float32(2**params['saturation']),0,1)
 x=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB);stages.append(ORDER[4])
 d=params['detail']
 if d<0:x=cv2.GaussianBlur(x,(0,0),2*abs(d),borderType=cv2.BORDER_REFLECT_101)
 elif d>0:x=x+np.float32(2*d)*(x-cv2.GaussianBlur(x,(0,0),1.,borderType=cv2.BORDER_REFLECT_101))
 stages.append(ORDER[5]);result=quantize(x);stages.append(ORDER[6])
 assert result.shape==rgb.shape and result.dtype==np.uint8
 return (result,stages) if trace else result

