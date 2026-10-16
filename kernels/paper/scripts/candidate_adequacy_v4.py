import numpy as np
from sensor_v4_common import seed
from sensor_v4_rank_selection import selection_support

def geometry_pool(pair_id, shape, config):
    eligible=np.flatnonzero(selection_support(shape,config).ravel())
    pool=config['candidate_adequacy']['pool']
    return np.random.default_rng(seed(pool['seed'],pair_id,'geometry_only')).choice(
        eligible,size=min(pool['pixels'],len(eligible)),replace=False).astype(np.uint32)


def rank_scores_from_counts(counts, subset):
    ids=counts['candidate_ids'];indices=[ids.index(c) for c in subset]
    n=counts['pixels'];less=np.asarray(counts['less_counts'],dtype=np.float64);ties=np.asarray(counts['equal_counts'],dtype=np.float64)
    return {ids[i]:float((n+sum(less[j,i]+.5*ties[j,i] for j in indices if j!=i))/n) for i in indices}

