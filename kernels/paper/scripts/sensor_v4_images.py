"""Declared RobustSpring recipes and synthetic sensor variants, without GT."""
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
from sensor_v4_common import digest, seed

cv2.setNumThreads(1)


def quantize(x):
    return np.rint(np.clip(x, 0, 1)*255).astype(np.uint8)


def corrupt(rgb, condition, record, eye, config):
    if condition in ('gaussian_blur', 'jpeg_compression', 'pixelate'):
        from corruptions_robustspring_v2 import corrupt as audited_operator
        return audited_operator(rgb, condition, seed=0,
            parameters=config['table3']['broader_corruptions'][condition])
    p = config['corruption']
    x = rgb.astype(np.float32)/255
    rng = np.random.default_rng(seed(config['protocol_id'], p['input_seed'], record['pair_id'], condition, eye))
    if condition == 'clean':
        return rgb.copy()
    if condition == 'brightness':
        x = x + p[condition]['offset']
    elif condition == 'contrast':
        mean = x.mean(axis=(0, 1), dtype=np.float64, keepdims=True)
        x = (x-mean)*p[condition]['factor']+mean
    elif condition == 'saturation':
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1]*p[condition]['scale']+p[condition]['offset'], 0, 1)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif condition == 'gaussian_noise':
        x = x + rng.normal(0, p[condition]['sigma'], x.shape)
    elif condition == 'shot_noise':
        x = rng.poisson(x*p[condition]['rate'])/p[condition]['rate']
    elif condition == 'speckle_noise':
        x = x + x*rng.normal(0, p[condition]['sigma'], x.shape)
    else:
        raise ValueError(f'Condition not in v4: {condition}')
    return quantize(x)


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x*12.92, 1.055*x**(1/2.4)-0.055)


def candidate_bank(config, ids=None, k=None):
    bank = config['table3']['candidates']
    if ids is not None:
        wanted = set(ids)
        if not wanted.issubset({c['id'] for c in bank}):
            raise ValueError('Unknown candidate ID')
        bank = [c for c in bank if c['id'] in wanted]
    elif k is not None:
        if not 0 <= k <= len(bank):
            raise ValueError('K outside configured bank; supply an expanded config')
        bank = bank[:k]
    else:
        bank = bank[:config['table3']['K']]
    return [{'id': 'identity', 'ev': 0.0, 'gain': 1.0}] + bank


def variant(incoming, candidate, record, condition, eye, config):
    if candidate['id'] == 'identity':
        return incoming.copy()
    x = srgb_to_linear(incoming.astype(np.float32)/255)
    # Exposure clips before subsequent gain; combined settings remain explicit.
    x = np.clip(x*2**candidate['ev'], 0, 1)
    gain = candidate['gain']
    if gain != 1:
        p = config['table3']['gain_noise']
        rng = np.random.default_rng(seed(config['protocol_id'], config['table3']['variant_seed'],
                                         record['pair_id'], condition, candidate['id'], eye))
        sigma = np.sqrt((p['read_sigma_linear']*gain)**2 + p['shot_scale_linear']**2*gain*x)
        x = x*gain + rng.standard_normal(x.shape)*sigma
    return quantize(linear_to_srgb(x))


def image_stats(rgb, reference=None):
    x = rgb.astype(np.float64)/255
    result = {'sha256': digest(rgb.tobytes()), 'mean_luma_srgb': float((x*[.2126,.7152,.0722]).sum(-1).mean()),
              'zero_channel_fraction': float((rgb == 0).mean()), 'saturated_channel_fraction': float((rgb == 255).mean()),
              'std_srgb': float(x.std())}
    if reference is not None:
        diff = x-reference.astype(np.float64)/255
        result.update({'delta_rms': float(np.sqrt((diff*diff).mean())), 'delta_std': float(diff.std())})
    return result


def cached_rgb(path, build):
    """Content/config-addressed paths supplied by caller; never overwrite."""
    path = Path(path)
    if path.exists():
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Invalid cached image')
        return image
    image = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent prefetch can request the same incoming observation. Publish a
    # complete immutable file atomically without ever replacing existing bytes.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as f:
        temp = Path(f.name)
        np.save(f, image, allow_pickle=False)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if not np.array_equal(np.load(path, allow_pickle=False), image):
                raise ValueError('Cache key produced different pixels')
    finally:
        temp.unlink()
    return image
