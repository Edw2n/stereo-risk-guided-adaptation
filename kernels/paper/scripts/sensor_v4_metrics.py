"""Native Spring server formulas and sequence-cluster uncertainty estimates."""
import numpy as np
from scipy.stats import rankdata


def native_maps(prediction, gt):
    pred = np.asarray(prediction, dtype=np.float32)
    # Preserve GT storage dtype for the relative threshold, just as the public
    # server does. In particular its 0.05*float16 GT maximum rounds in float16.
    gt = np.asarray(gt)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction: fail rather than hide erroneous pixels')
    h, w = pred.shape
    if gt.shape == (2*h, 2*w):
        candidates = np.stack([gt[y::2, x::2] for y in range(2) for x in range(2)])
    elif gt.shape == pred.shape:
        candidates = gt[None]
    else:
        raise ValueError('GT shape does not match native or 2x native grid')
    # np.min, NOT nanmin: this matches the public server's four-candidate rule.
    valid = np.isfinite(candidates).all(axis=0)
    error = np.abs(candidates.astype(np.float32)-pred[None]).min(axis=0).astype(np.float64)
    gt_magnitude = np.abs(candidates).max(axis=0)
    d1 = (error > 3) & (error > .05*gt_magnitude)
    common = valid & ((candidates >= 0) & (candidates < 192)).all(axis=0)
    return error, d1, valid, common


def sufficient(error, d1, mask):
    n = int(mask.sum())
    return {'n': n, 'abs_sum': float(error[mask].sum(dtype=np.float64)),
            'px1_count': int((error[mask] > 1).sum()), 'd1_count': int(d1[mask].sum())}


def native_values(stats):
    n = stats['n']
    return {'Abs': stats['abs_sum']/n if n else None,
            '1px': 100*stats['px1_count']/n if n else None,
            'D1': 100*stats['d1_count']/n if n else None}


def bootstrap_weights(scenes, repetitions, seed):
    # The same weights are reused for conditions/signals/models for paired CIs.
    return np.random.default_rng(seed).multinomial(len(scenes), np.full(len(scenes), 1/len(scenes)), repetitions)


def ci(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return [float(v) for v in np.quantile(finite, [.025, .975])] if len(finite) else [None, None]


def table1_summary(rows, scenes, weights, support):
    keys = ['n', 'abs_sum', 'px1_count', 'd1_count']
    totals = np.array([[sum(r[support][k] for r in rows if r['scene'] == s) for k in keys] for s in scenes], dtype=float)
    pooled = dict(zip(keys, totals.sum(0)))
    point = native_values(pooled)
    boot = weights@totals
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.column_stack([boot[:,1]/boot[:,0], 100*boot[:,2]/boot[:,0], 100*boot[:,3]/boot[:,0]])
    result = {**point, 'valid_pixels': int(pooled['n']), 'sequences': len(scenes)}
    for i, name in enumerate(['Abs', '1px', 'D1']):
        result[name+'_ci_low'], result[name+'_ci_high'] = ci(values[:,i])
    return result, values


def risk_points(y, risk, error, error_ranks=None, return_sorted=False):
    if not np.isfinite(risk).all():
        raise ValueError('Nonfinite enabled signal on fixed common domain')
    n = len(y)
    if not n:
        raise ValueError('No eligible pixels')
    order = np.argsort(risk, kind='stable')
    sorted_risk, sorted_y = risk[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_risk[1:] != sorted_risk[:-1])+1, n]
    starts = np.r_[0, ends[:-1]]
    ranks = np.repeat((starts+ends+1)/2, ends-starts)
    positives = int(y.sum()); negatives = n-positives
    both = positives > 0 and negatives > 0
    auc = float((ranks[sorted_y].sum()-positives*(positives+1)/2)/(positives*negatives)) if both else None
    cumulative = np.cumsum(sorted_y, dtype=np.int64)[ends-1]
    counts = np.diff(np.r_[0, cumulative])[::-1]
    cumulative_desc = np.cumsum(counts)
    precision = cumulative_desc/np.cumsum((ends-starts)[::-1])
    ap = float(np.sum(counts*precision)/positives) if both else None
    if error_ranks is None:
        error_ranks = rankdata(error, method='average')
    a = ranks-(n+1)/2
    b = error_ranks[order]-(n+1)/2
    denominator = np.sqrt(np.sum(a*a)*np.sum(b*b))
    point = {'AUROC': auc, 'AUPR': ap, 'prevalence': positives/n,
             'Spearman': float(np.sum(a*b)/denominator) if denominator > 0 else None,
             'n': n, 'positives': positives}
    return (point, order, sorted_risk) if return_sorted else point


def histogram_metrics(positive, negative):
    tp = np.cumsum(positive[..., ::-1], axis=-1)
    fp = np.cumsum(negative[..., ::-1], axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr, fpr = tp/tp[..., -1:], fp/fp[..., -1:]
        precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=tp+fp > 0)
        tpr = np.concatenate([np.zeros_like(tpr[..., :1]), tpr], axis=-1)
        fpr = np.concatenate([np.zeros_like(fpr[..., :1]), fpr], axis=-1)
        auc = np.trapezoid(tpr, fpr, axis=-1)
        ap = np.sum(np.diff(tpr, axis=-1)*precision, axis=-1)
    bad = (positive.sum(-1) == 0) | (negative.sum(-1) == 0)
    return np.where(bad, np.nan, auc), np.where(bad, np.nan, ap)


def risk_cluster_summary(y, risk, error, scene_index, scenes, weights, cfg, error_ranks=None):
    point, order, sorted_risk = risk_points(y, risk, error, error_ranks, return_sorted=True)
    # Exact points; quantile histogram only accelerates cluster bootstrap CIs.
    # Approximation is measured on this very group and refined if necessary.
    bins = cfg['ci_score_bins']
    while True:
        positions = np.linspace(0, len(risk)-1, bins+1)
        lower, upper = positions.astype(np.int64), np.ceil(positions).astype(np.int64)
        fraction = positions-lower
        edges = np.unique(sorted_risk[lower]*(1-fraction)+sorted_risk[upper]*fraction)
        idx = np.searchsorted(edges[1:-1], sorted_risk, side='right')
        nb = max(1, len(edges)-1)
        flat = scene_index[order]*nb+idx
        positive = np.bincount(flat, weights=y[order], minlength=len(scenes)*nb).reshape(len(scenes), nb)
        negative = np.bincount(flat, weights=~y[order], minlength=len(scenes)*nb).reshape(len(scenes), nb)
        ha, hp = histogram_metrics(positive.sum(0), negative.sum(0))
        delta = max(abs(float(v)-point[k]) for k, v in [('AUROC', ha), ('AUPR', hp)] if point[k] is not None) if point['AUROC'] is not None else 0
        if delta <= cfg['ci_max_point_error']:
            break
        bins *= 2
        if bins > 65536:
            raise ValueError('Histogram CI approximation exceeds declared tolerance')
    # Keep working memory bounded even if the approximation check refines bins.
    boot = [histogram_metrics(w@positive, w@negative) for w in np.array_split(weights, max(1, (len(weights)+63)//64))]
    auroc = np.concatenate([x[0] for x in boot]); aupr = np.concatenate([x[1] for x in boot])
    pop = weights@(positive+negative).sum(1)
    prevalence = np.divide(weights@positive.sum(1), pop, out=np.full(len(weights), np.nan), where=pop > 0)
    for name, vals in [('AUROC', auroc), ('AUPR', aupr), ('prevalence', prevalence)]:
        point[name+'_ci_low'], point[name+'_ci_high'] = ci(vals)
    point.update({'ci_score_bins': nb, 'ci_point_approximation_error': delta, 'ci_valid_replicates': int(np.isfinite(auroc).sum())})
    return point, np.stack([auroc, aupr, prevalence], axis=-1)
