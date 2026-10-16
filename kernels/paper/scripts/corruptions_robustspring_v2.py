"""Candidate paper-aligned recipes; NOT the official RobustSpring generator.

Legacy conditions/results remain unchanged. All implementation choices and gaps
are explicit in configs/robustspring_aligned_v2.json. GT is never an input.
"""
import hashlib
import io
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, features
import PIL

ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/'configs/robustspring_aligned_v2.json'
PROTOCOL=json.loads(CONFIG_PATH.read_text())
CORRUPTIONS=PROTOCOL['conditions']


def image_seed(scene,frame,name,eye,base_seed=2026):
    if eye not in (0,1):raise ValueError('eye must be 0 (left) or 1 (right)')
    key=CORRUPTIONS[name]['seed_key']
    text='|'.join(map(str,[base_seed,scene,frame,key,eye]))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')


def quantize(x):
    return np.rint(np.clip(x,0,255)).astype(np.uint8)


def corrupt(image,name,seed,parameters=None):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected HWC RGB uint8 input')
    p=CORRUPTIONS[name] if parameters is None else parameters
    if name=='clean':return image.copy()
    if name=='gaussian_blur':
        return cv2.GaussianBlur(image,(0,0),p['sigma_px'],borderType=cv2.BORDER_REFLECT_101)
    x=image.astype(np.float32)
    if name=='gaussian_noise':
        x+=np.random.default_rng(seed).normal(0,p['sigma_255'],x.shape).astype(np.float32)
    elif name=='shot_noise':
        x=np.random.default_rng(seed).poisson((x.astype(np.float64)/255)*p['poisson_scale'])/p['poisson_scale']*255
    elif name=='brightness':x+=p['offset_unit']*255
    elif name=='contrast':
        mean=x.mean(axis=(0,1),keepdims=True,dtype=np.float64)
        x=(x-mean)*p['multiplier']+mean
    elif name=='saturate':
        hsv=cv2.cvtColor(x/255,cv2.COLOR_RGB2HSV)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*p['alpha']+p['beta'],0,1)
        x=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)*255
    elif name=='jpeg_compression':
        buffer=io.BytesIO()
        Image.fromarray(image).save(buffer,format='JPEG',quality=p['quality'],
            subsampling=p['subsampling'],optimize=p['optimize'],progressive=p['progressive'])
        buffer.seek(0)
        with Image.open(buffer) as result:return np.array(result.convert('RGB'))
    elif name=='pixelate':
        h,w=image.shape[:2];c=p['size_fraction']
        small=(max(1,int(w*c)),max(1,int(h*c)))
        result=Image.fromarray(image).resize(small,Image.Resampling.BOX).resize((w,h),Image.Resampling.BOX)
        return np.array(result)
    else:raise ValueError(f'Unsupported condition: {name}')
    return quantize(x)


def corrupt_pair(left,right,name,scene,frame,base_seed=2026):
    return [corrupt(im,name,image_seed(scene,frame,name,eye,base_seed))
        for eye,im in enumerate((left,right))]


def provenance():
    return {'protocol':PROTOCOL,'protocol_sha256':hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy':np.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__,
        'jpeg_codec':features.version_codec('jpg')}
