"""Candidate-axis rank voting and dev-only pixel fusion.

Risk inputs are spatial maps, never pre-averaged scalar risks. No fitting, GT,
error maps, or GT validity masks enter Signal-Global / Signal-Pixel decisions.
Candidate IDs are canonicalized so tied draws do not depend on loader order.
"""
import numpy as np
from sensor_v4_common import seed, assert_dev


def selection_support(shape, config):
    """Input-geometry support, not the evaluation's GT-valid mask."""
    h, w = shape
    excluded = config['table3']['rank_selection']['exclude_left_columns']
    if h < 1 or not 0 <= excluded < w:
        raise ValueError('No ranking support remains')
    support = np.ones((h, w), dtype=bool)
    support[:, :excluded] = False
    return support


def ordered_maps(maps):
    ids = sorted(maps)
    if not ids or 'identity' not in ids:
        raise ValueError('Identity and at least one spatial risk map are required')
    shape = np.shape(maps[ids[0]])
    if len(shape) != 2 or any(np.shape(maps[c]) != shape for c in ids):
        raise ValueError('Expected matching H x W risk maps, not raw mean risks')
    return ids, shape


def tie_seed(record, config, purpose):
    return seed(config['table3']['rank_selection']['tie_seed'], purpose,
                record['pair_id'], record['incoming_condition'], record['model'], record['primary_signal'])


def mean_pixel_ranks(risk_maps, config):
    if config['table3']['rank_selection']['risk_ties']!='average':
        raise ValueError('This protocol requires average ranks for equal risks')
    ids, shape = ordered_maps(risk_maps)
    support = selection_support(shape, config)
    flat_support = support.ravel()
    sums = np.zeros(len(ids), dtype=np.float64)
    pixel_ties = 0
    chunk = config['table3']['rank_selection']['chunk_pixels']
    if chunk < 1:
        raise ValueError('chunk_pixels must be positive')
    for start in range(0, support.size, chunk):
        mask = flat_support[start:start+chunk]
        if not mask.any():
            continue
        values = np.stack([np.asarray(risk_maps[c]).reshape(-1)[start:start+chunk][mask] for c in ids])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite risk on fixed support; never select a score-dependent mask')
        # Exact average rank: 1 + count(strictly lower) + (equal count - 1)/2.
        # K is small; this avoids a large H x W x K x K tensor.
        tied = np.zeros(values.shape[1], dtype=bool)
        for k in range(len(ids)):
            lower = np.sum(values < values[k], axis=0)
            equal = np.sum(values == values[k], axis=0)
            sums[k] += np.sum(1 + lower + (equal-1)/2, dtype=np.float64)
            tied |= equal > 1
        pixel_ties += int(tied.sum())
    n = int(support.sum())
    return {c: float(sums[i]/n) for i, c in enumerate(ids)}, {
        'voting_pixels': n, 'total_pixels': support.size, 'pixels_with_any_risk_tie': pixel_ties,
        'risk_tie_method': 'average ranks for exactly equal risks', 'GT_used': False}


def signal_global(risk_maps, record, config):
    if config['table3']['rank_selection']['global_score_rtol']!=0:
        raise ValueError('Only the declared absolute score tie tolerance is supported')
    scores, stats = mean_pixel_ranks(risk_maps, config)
    best = min(scores.values())
    tol = config['table3']['rank_selection']['global_score_atol']
    if tol < 0:
        raise ValueError('Global tie tolerance must be nonnegative')
    tied = sorted(c for c, value in scores.items() if abs(value-best) <= tol)
    used_seed = tie_seed(record, config, 'Signal-Global')
    chosen = str(np.random.default_rng(used_seed).choice(tied))
    return {'candidate_id': chosen, 'mean_pixel_ranks': scores, 'tied_candidates': tied,
            'tie_seed': used_seed, 'score_atol': tol, **stats}


def pixel_signal_fusion(disparities, risk_maps, record, config):
    # The public diagnostic API always stays dev-only. The production wrapper
    # validates full freezes, separate GO and explicit Fusion approval first.
    assert_dev([record])
    return _pixel_signal_fusion(disparities, risk_maps, record, config)


def _pixel_signal_fusion(disparities, risk_maps, record, config):
    """Shared GT-free mathematics; public entry points enforce execution scope."""
    ids, shape = ordered_maps(risk_maps)
    if set(disparities) != set(ids) or any(np.shape(disparities[c]) != shape for c in ids):
        raise ValueError('Disparity and risk candidates must match')
    if any(not np.isfinite(disparities[c]).all() for c in ids):
        raise ValueError('Nonfinite disparity')
    support = selection_support(shape, config)
    fused = np.asarray(disparities['identity']).copy()
    chosen = np.full(shape, ids.index('identity'), dtype=np.int16)
    rng = np.random.default_rng(tie_seed(record, config, 'Signal-Pixel Fusion'))
    chunk = config['table3']['rank_selection']['chunk_pixels']
    tie_pixels = 0
    for start in range(0, support.size, chunk):
        stop = min(start+chunk, support.size)
        mask = support.ravel()[start:stop]
        # Consume one uniform variate at every raster pixel, even outside support:
        # changing chunk boundaries never changes tied pixel choices.
        uniforms = rng.random(stop-start)[mask]
        if not mask.any():
            continue
        risk = np.stack([np.asarray(risk_maps[c]).reshape(-1)[start:stop][mask] for c in ids])
        if not np.isfinite(risk).all():
            raise ValueError('Nonfinite risk on fixed support')
        tied = risk == risk.min(axis=0)
        counts = tied.sum(axis=0)
        ordinal = np.floor(uniforms*counts).astype(np.int64)
        which = np.argmax(np.cumsum(tied, axis=0) > ordinal, axis=0)
        values = np.stack([np.asarray(disparities[c]).reshape(-1)[start:stop][mask] for c in ids])
        fused.ravel()[start:stop][mask] = values[which, np.arange(len(which))]
        chosen.ravel()[start:stop][mask] = which
        tie_pixels += int((counts > 1).sum())
    return fused, chosen, {'candidate_ids': ids, 'GT_used': False,
        'selected_pixel_counts': {c: int(((chosen == i) & support).sum()) for i, c in enumerate(ids)},
        'pixel_ties': tie_pixels, 'outside_support': 'Identity disparity retained',
        'outside_support_pixels': int((~support).sum()), 'scope': 'dev diagnostic only'}


def pixel_oracle(disparities, ground_truth, record):
    """Evaluation-only per-pixel minimum absolute error; never a real selector."""
    assert_dev([record])
    from sensor_v4_metrics import native_maps
    ids, shape = ordered_maps(disparities)
    chosen = np.full(shape, ids.index('identity'), dtype=np.int16)
    fused = np.asarray(disparities['identity']).copy()
    best = np.full(shape, np.inf, dtype=np.float64)
    for i, c in enumerate(ids):
        error, _, valid, _ = native_maps(disparities[c], ground_truth)
        take = valid & (error < best)
        best[take] = error[take]
        fused[take] = disparities[c][take]
        chosen[take] = i
    return fused, chosen, {'candidate_ids': ids, 'GT_used': True,
        'objective': 'minimum native absolute disparity error at each GT-valid pixel',
        'error_ties': 'lexicographic candidate ID', 'invalid_GT': 'Identity retained, excluded from metrics',
        'scope': 'dev-only evaluation upper bound'}
