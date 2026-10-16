"""Instrument unmodified official RAFT-Stereo without using GT in inference."""
import sys
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/RAFT-Stereo'))
import core.raft_stereo as raft_module
from core.utils.utils import InputPadder
BASE_CORRELATION = raft_module.CorrBlock1D


class RAFTAdapter:
    def __init__(self, checkpoint, iters=32, last_n=4):
        if not 2 <= last_n <= iters:
            raise ValueError('last_n must be between 2 and the iteration count')
        self.last_n = last_n
        self.iters = iters
        self.args = Namespace(hidden_dims=[128]*3, context_norm='batch', n_downsample=2,
            n_gru_layers=3, shared_backbone=False, mixed_precision=False,
            corr_implementation='reg', corr_levels=4, corr_radius=4, slow_fast_gru=False)
        self.model = raft_module.RAFTStereo(self.args)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.eval().cuda()
        self.native_signals = {}
        original_class = BASE_CORRELATION
        adapter = self

        class InstrumentedCorrelation(original_class):
            def __init__(self, fmap1, fmap2, **kwargs):
                super().__init__(fmap1, fmap2, **kwargs)
                adapter.ambiguity_start = torch.cuda.Event(enable_timing=True)
                adapter.ambiguity_end = torch.cuda.Event(enable_timing=True)
                adapter.ambiguity_start.record()
                b, _, h, w = fmap1.shape
                scores = self.corr_pyramid[0].view(b, h, w, w)
                # Native correlation uses right-coordinate candidates (not disparity).
                xr = torch.arange(w, device=scores.device)
                xl = torch.arange(w, device=scores.device)[:, None]
                candidate_mask = (xr[None] <= xl) & (xr[None]*4 >= adapter.pad_left) & (xr[None]*4 < adapter.pad_left+adapter.width)
                count = candidate_mask.sum(-1).float()
                summaries = {k: [] for k in ['negative_peak', 'negative_margin', 'entropy']}
                for row in range(0, h, 16):
                    logits = scores[:, row:row+16].masked_fill(~candidate_mask, -torch.inf)
                    top2 = logits.topk(2, dim=-1).values
                    probabilities = torch.softmax(logits, dim=-1)
                    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                    entropy = entropy/count.clamp_min(2).log()[None, None]
                    margin = top2[..., 0]-top2[..., 1]
                    margin[..., count < 2] = torch.nan
                    entropy[..., count < 2] = torch.nan
                    summaries['negative_peak'].append(-top2[..., 0])
                    summaries['negative_margin'].append(-margin)
                    summaries['entropy'].append(entropy)
                adapter.native_signals = {k: torch.cat(v, dim=1)[:, None] for k,v in summaries.items()}
                adapter.ambiguity_end.record()

        self.correlation_class = InstrumentedCorrelation

    @torch.inference_mode()
    def predict(self, left, right):
        disparity, signals, _ = self.predict_batch([left], [right])
        return disparity[0], {k: v[0] for k,v in signals.items()}

    @torch.inference_mode()
    def predict_batch(self, left, right, precision='fp32'):
        assert len(left)==len(right) and len(left)>0
        assert precision in ('fp32','fp16','bf16')
        self.height, self.width = left[0].shape[:2]
        assert all(im.shape==left[0].shape for im in list(left)+list(right)), 'Batch requires equal resolution'
        start=time.perf_counter()
        images = [torch.from_numpy(np.stack(ims)).permute(0,3,1,2).float().cuda() for ims in (left,right)]
        padder = InputPadder(images[0].shape, divis_by=32)
        self.pad_left = padder._pad[0]
        images = padder.pad(*images)
        torch.cuda.synchronize()
        transfer_seconds=time.perf_counter()-start
        begin=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        self.args.mixed_precision=precision!='fp32'
        old_autocast=raft_module.autocast
        dtype=torch.bfloat16 if precision=='bf16' else torch.float16
        raft_module.autocast=lambda enabled: torch.autocast('cuda',dtype=dtype,enabled=enabled)
        raft_module.CorrBlock1D=self.correlation_class
        begin.record()
        model_start=time.perf_counter()
        try:
            predictions = self.model(*images, iters=self.iters, test_mode=False)
        finally:
            raft_module.autocast=old_autocast
            raft_module.CorrBlock1D=BASE_CORRELATION
        end.record()
        torch.cuda.synchronize()
        model_seconds=time.perf_counter()-model_start
        ambiguity_seconds=self.ambiguity_start.elapsed_time(self.ambiguity_end)/1000
        post_start=time.perf_counter()
        disparity = -padder.unpad(predictions[-1])[:,0]
        signals = {k: padder.unpad(F.interpolate(v, size=images[0].shape[-2:], mode='nearest'))[:,0]
                   for k,v in self.native_signals.items()}
        late = torch.stack([padder.unpad(p)[:,0] for p in predictions[-self.last_n:]])
        signals['final_update'] = (late[-1]-late[-2]).abs()
        signals['late_std'] = late.std(dim=0, correction=0)
        signals['late_mean_update'] = (late[1:]-late[:-1]).abs().mean(dim=0)
        torch.cuda.synchronize()
        post_seconds=time.perf_counter()-post_start
        output_start=time.perf_counter()
        result = {k: v.float().cpu().numpy() for k,v in signals.items()}
        disparity_cpu=disparity.float().cpu().numpy()
        output_seconds=time.perf_counter()-output_start
        timing={'input_transfer_seconds':transfer_seconds,
                'model_inference_seconds':max(0,model_seconds-ambiguity_seconds),
                'signal_extraction_seconds':ambiguity_seconds+post_seconds,
                'output_transfer_seconds':output_seconds,
                'model_forward_gpu_seconds':begin.elapsed_time(end)/1000,
                'prediction_total_seconds':time.perf_counter()-start}
        return disparity_cpu, result, timing
