"""Model-independent risk-map interface and explicit manual selection gate."""
import cv2
import numpy as np
from sensor_v4_common import seed, digest


class RiskExtractor:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.internal = config['models'][model_name]['internal_signals']
        self.baselines = config['signals']['baselines']
        if not self.internal or len(set(self.internal)) != len(self.internal):
            raise ValueError('Enabled internal signals must be nonempty and unique')

    def maps(self, native, disparity, left, record):
        # Adapter native signals already negate peak/margin; all maps point to risk.
        maps = {name: native[name] for name in self.internal}
        gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        bases = {
            'random': lambda: np.random.default_rng(seed('risk-random', record['pair_id'])).random(disparity.shape, dtype=np.float32),
            'negative_texture': lambda: -np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)),
            'disparity_magnitude': lambda: np.abs(disparity),
            'disparity_gradient': lambda: np.hypot(cv2.Sobel(disparity, cv2.CV_32F, 1, 0), cv2.Sobel(disparity, cv2.CV_32F, 0, 1))
        }
        maps.update({name: bases[name]() for name in self.baselines})
        return maps

    def observation_scores(self, maps):
        # Descriptive logging ONLY. These raw means must never select candidates.
        out = {}
        for name, risk in maps.items():
            region = risk[:,4:]
            if not np.isfinite(region).all():
                raise ValueError(f'Nonfinite risk in observation support: {name}')
            out[name] = float(region.mean(dtype=np.float64))
        return out


def recommend(rows, model, config):
    enabled = config['models'][model]['internal_signals']
    candidates = [r for r in rows if r['model'] == model and r['split'] == 'dev'
                  and r['condition'] == 'corruption_macro' and r['signal'] in enabled
                  and r.get('support','full')=='full']
    complete = [r for r in candidates if r.get('condition_count') == 6 and r['AUROC'] is not None and r['AUPR'] is not None]
    if len(complete) != len(enabled):
        return {'status': 'not_eligible', 'reason': 'All six dev corruptions and all enabled internal signals required; tiny dry run cannot freeze', 'recommended': None}
    best_auc = max(r['AUROC'] for r in complete)
    tied = [r for r in complete if best_auc-r['AUROC'] < config['signals']['selection']['auroc_practical_tie']]
    best_ap = max(r['AUPR'] for r in tied)
    tolerance = config['signals']['selection'].get('ap_numerical_tie_atol',1e-12)
    # Floating-point numerical equality only; no practical AP tie band.
    ap_tied = [r for r in tied if best_ap-r['AUPR'] <= tolerance]
    winner = sorted(ap_tied, key=lambda r: (-r['worst_corruption_AUROC'], r['signal']))[0]
    return {'status': 'recommendation_only_manual_gate', 'recommended': winner['signal'],
            'auroc_tie_candidates': [r['signal'] for r in tied], 'ap_numerical_tie_atol': tolerance,
            'rule': config['signals']['selection'], 'dev_evidence_sha256': digest(complete), 'frozen': False}


def select_without_gt(candidate_ids, risks, selector, frozen, record, config):
    """Inputs deliberately contain no GT/native-metric values."""
    if selector == 'Identity':
        return 'identity'
    if selector == 'Random':
        rng = np.random.default_rng(seed('random-selector', config['table3']['random_selector_seed'], record['pair_id'], record['incoming_condition']))
        return str(rng.choice(sorted(candidate_ids)))
    if not frozen or frozen.get('manually_confirmed') is not True:
        raise PermissionError('Signal-Global/Fixed-best require explicit dev-only manual freeze')
    if selector == 'Fixed-best':
        chosen = frozen['fixed_best']
    elif selector == 'Signal-Global':
        from sensor_v4_rank_selection import signal_global
        maps = {c: risks[c][frozen['primary_signal']] for c in candidate_ids}
        chosen = signal_global(maps, {**record, 'primary_signal': frozen['primary_signal']}, config)['candidate_id']
    else:
        raise ValueError('GT-based Oracle must use the separate diagnostic evaluator')
    if chosen not in candidate_ids:
        raise ValueError('Frozen candidate is absent from the bank')
    return chosen


def oracle_diagnostic(metrics):
    return min(metrics, key=lambda c: (metrics[c]['D1'], metrics[c]['Abs'], metrics[c]['1px'], c))


def fit_fixed_best(dev_rows, *, bank_frozen=False, dev_fixture=False, expected_dev_pair_ids=None):
    if not dev_rows or any(r['split'] != 'dev' for r in dev_rows):
        raise PermissionError('Fixed-best fitting requires dev only')
    if not bank_frozen and not dev_fixture:
        raise PermissionError('Freeze the candidate bank before fitting Fixed-best')
    if not dev_fixture:
        if not expected_dev_pair_ids or {r['pair_id'] for r in dev_rows} != set(expected_dev_pair_ids):
            raise ValueError('Fixed-best requires the complete declared dev pair manifest')
    coverage = {}
    for r in dev_rows:
        key = (r['pair_id'], r['condition'])
        candidate = r['candidate_id']
        seen = coverage.setdefault(candidate, set())
        if key in seen:
            raise ValueError('Repeated observation would bias pooled Fixed-best')
        seen.add(key)
    if any(v != next(iter(coverage.values())) for v in coverage.values()):
        raise ValueError('Fixed-best candidates need identical dev observations')
    from sensor_v4_metrics import native_values
    sums = {}
    for r in dev_rows:
        target = sums.setdefault(r['candidate_id'], dict.fromkeys(r['full'], 0))
        for k, value in r['full'].items():
            target[k] += value
    return oracle_diagnostic({c: native_values(s) for c, s in sums.items()})
