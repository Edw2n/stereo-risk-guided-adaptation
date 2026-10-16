"""Read-only CroCo hooks. Extended risks never enter the disparity computation."""
import math
import torch
import torch.nn.functional as F
from screening.croco_proxy_extract import ProxyAdapter as OriginalProxyAdapter

SIGNALS = [
    'head_uncertainty_raw_weighted', 'head_logscale_weighted', 'tile_disparity_std',
    'cross_attention_entropy', 'cross_attention_negative_margin',
    'epipolar_attention_entropy', 'epipolar_attention_negative_peak',
    'epipolar_attention_negative_margin', 'epipolar_attention_negative_row_mass',
    'attention_disparity_residual', 'attention_head_disagreement',
    'decoder_late_mean_relative_update', 'decoder_final_relative_update',
    'decoder_late_mean_cosine_change', 'tile_disparity_std_unweighted',
    'tile_laplace_mixture_variance',
]

def row_attention(prob, rows, cols, patch=16):
    """One head [B,Q,K]: same patch row, all columns, no GT or disparity mask.

    A zero-mass row has uniform conditional probabilities, but its separate mass
    remains zero. Disparity is patch-center xL - E[xR] in native pixel units.
    """
    assert prob.shape[-2:] == (rows * cols, rows * cols)
    query = torch.arange(rows * cols, device=prob.device)
    keys = (query // cols)[:, None] * cols + torch.arange(cols, device=prob.device)[None]
    p = prob[:, query[:, None], keys]
    mass = p.sum(-1)
    q = torch.where(mass[..., None] > 0,
                    p / mass[..., None].clamp_min(torch.finfo(p.dtype).tiny),
                    torch.full_like(p, 1 / cols))
    entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(-1) / math.log(cols)
    top = q.topk(2, dim=-1).values
    expected = (q * torch.arange(cols, device=p.device, dtype=p.dtype)).sum(-1)
    disparity = patch * ((query % cols)[None] - expected)
    return torch.stack([entropy, -top[..., 0], -(top[..., 0] - top[..., 1]), -mass], 1), disparity

def feature_update(before, after, eps=1e-8):
    relative = torch.linalg.vector_norm(after - before, dim=-1) / torch.linalg.vector_norm(before, dim=-1).clamp_min(eps)
    cosine = (1 - F.cosine_similarity(before, after, dim=-1, eps=eps)).clamp(0, 2)
    both_zero = (before.square().sum(-1) == 0) & (after.square().sum(-1) == 0)
    return relative, torch.where(both_zero, torch.zeros_like(cosine), cosine)

def mixture_variance(weighted_variance, weighted_scale_squared):
    return weighted_variance.clamp_min(0) + 2 * weighted_scale_squared

class ProxyAdapter(OriginalProxyAdapter):
    def __init__(self, config, overlap=.7):
        super().__init__(config, overlap)
        assert len(self.base.model.dec_blocks) == 12
        self.decoder_updates = []
        self.extra_attention = None
        self.decoder_handles = [b.register_forward_hook(self._decoder_hook)
                                for b in self.base.model.dec_blocks[-4:]]

    def _decoder_hook(self, module, args, output):
        if self.active:
            self.decoder_updates.append(feature_update(args[0], output[0]))

    def _attention_hook(self, module, args, prob):
        if not self.active:
            return
        assert prob.shape[1:] == (12, 968, 968)
        # Preserve original reductions exactly, without changing prob in place.
        super()._attention_hook(module, args, prob)
        ep = torch.zeros_like(self.attention[:, :1]).expand(-1, 4, -1).clone()
        expected = []
        for head in range(prob.shape[1]):
            stats, d = row_attention(prob[:, head], 22, 44)
            ep += stats
            expected.append(d)
        ep /= prob.shape[1]
        ds = torch.stack(expected, 1)
        self.extra_attention = (ep, ds.mean(1), ds.std(1, unbiased=False))

    def _tiled(self, model, criterion, img1, img2, gt, **kwargs):
        kwargs['overlap'] = self.overlap
        if not self.active:
            return self.original_tiled(model, criterion, img1, img2, gt, **kwargs)
        assert criterion is None and gt is None and kwargs['with_conf']
        assert kwargs['conf_mode'] == 'conf_expsigmoid_15_3'
        assert kwargs['crop'] == [352, 704] and img1.shape[-2:] == (1080, 1920)
        from stereoflow.engine import _overlapping, split_prediction_conf
        B, _, H, W = img1.shape
        th, tw = kwargs['crop']
        pred_sum = img1.new_zeros((B, 1, H, W))
        weights = img1.new_zeros((B, H, W)) + 1e-16
        raw_sum = img1.new_zeros((B, H, W)); log_sum = torch.zeros_like(raw_sum)
        att_sum = img1.new_zeros((B, 2, H, W))
        variance_mean = torch.zeros_like(raw_sum); variance_m2 = torch.zeros_like(raw_sum)
        # Epipolar four + attention/prediction residual + head disagreement + three feature drifts.
        extra_sum = img1.new_zeros((B, 9, H, W))
        plain_mean = torch.zeros_like(raw_sum); plain_m2 = torch.zeros_like(raw_sum)
        scale2_sum = torch.zeros_like(raw_sum)
        coverage = torch.zeros((H, W), device=img1.device, dtype=torch.int16)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for sy in _overlapping(H, th, self.overlap):
            for sx in _overlapping(W, tw, self.overlap):
                self.attention = None; self.extra_attention = None; self.decoder_updates = []
                pred, u = split_prediction_conf(model(img1[..., sy, sx], img2[..., sy, sx]), with_conf=True)
                assert self.attention.shape == (B, 2, 22 * 44) and len(self.decoder_updates) == 4
                weight = torch.exp(-15 * 2 * (torch.sigmoid(u / 3) - .5)).view(B, th, tw)
                # IDENTICAL upstream disparity operations and summation order.
                pred_sum[..., sy, sx] += pred * weight[:, None]
                weights[..., sy, sx] += weight
                raw_sum[..., sy, sx] += u.view(B, th, tw) * weight
                logscale = 6 * (torch.sigmoid(u.view(B, th, tw) / 3) - .5)
                log_sum[..., sy, sx] += logscale * weight
                d = pred[:, 0]; mu = variance_mean[..., sy, sx]; delta = d - mu
                new_mu = mu + weight / weights[..., sy, sx] * delta
                variance_m2[..., sy, sx] += weight * delta * (d - new_mu)
                variance_mean[..., sy, sx] = new_mu
                a = F.interpolate(self.attention.reshape(B, 2, 22, 44), size=(th, tw), mode='bilinear', align_corners=False)
                att_sum[..., sy, sx] += a * weight[:, None]
                ep, att_d, head_std = self.extra_attention
                relative = torch.stack([v[0] for v in self.decoder_updates], 1)
                cosine = torch.stack([v[1] for v in self.decoder_updates], 1)
                token = torch.cat([ep, att_d[:, None], head_std[:, None], relative.mean(1)[:, None],
                                   relative[:, -1, None], cosine.mean(1)[:, None]], 1)
                extra = F.interpolate(token.reshape(B, 9, 22, 44), size=(th, tw), mode='bilinear', align_corners=False)
                extra[:, 4] = (extra[:, 4] - d).abs()  # per-tile native prediction agreement
                extra_sum[..., sy, sx] += extra * weight[:, None]
                coverage[sy, sx] += 1
                mu0 = plain_mean[..., sy, sx]; delta0 = d - mu0
                next0 = mu0 + delta0 / coverage[sy, sx]
                plain_m2[..., sy, sx] += delta0 * (d - next0)
                plain_mean[..., sy, sx] = next0
                scale2_sum[..., sy, sx] += torch.exp(2 * logscale) * weight
        disparity = pred_sum / weights[:, None]; raw = raw_sum / weights
        variance = (variance_m2 / weights).clamp_min(0)
        std = variance.sqrt(); std[:, coverage < 2] = 0
        unweighted = (plain_m2 / coverage).clamp_min(0).sqrt(); unweighted[:, coverage < 2] = 0
        risk = torch.cat([torch.stack([raw, log_sum / weights, std, att_sum[:, 0] / weights,
                                      att_sum[:, 1] / weights], 1), extra_sum / weights[:, None],
                          unweighted[:, None], mixture_variance(variance, scale2_sum / weights)[:, None]], 1)
        assert risk.shape[1] == len(SIGNALS)
        assert torch.isfinite(disparity).all() and torch.isfinite(risk).all()
        end.record(); torch.cuda.synchronize(); seconds = start.elapsed_time(end) / 1000
        self.last_risks = risk.float().cpu().numpy().copy(); self.coverage = coverage.cpu().numpy().copy()
        self.attention = None; self.extra_attention = None; self.decoder_updates = []
        return disparity, None, raw, seconds
