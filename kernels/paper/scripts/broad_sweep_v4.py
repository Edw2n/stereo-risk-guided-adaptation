import ctypes, subprocess
import numpy as np
from sensor_v4_common import ROOT, storage, digest, write_json
from sensor_v4_rank_selection import selection_support, tie_seed

def compile_counts(config):
    src=ROOT/'scripts/native/risk_pair_counts.cpp';folder=storage(config,'cache','native_helpers',digest(src.read_bytes())[:16]);so=folder/'risk_pair_counts.so'
    if not so.exists():
        subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared','-fopenmp',str(src),'-o',str(so)],check=True)
        write_json(folder/'BUILD.json',{'source_sha256':digest(src.read_bytes()),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'flags':['-O3','-std=c++17','-fPIC','-shared','-fopenmp'],'fast_math':False})
    return so


def exact_counts(risks,config,so):
    ids=sorted(risks);support=selection_support(next(iter(risks.values())).shape,config).ravel()
    x=np.ascontiguousarray(np.stack([risks[c].ravel()[support] for c in ids]),dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite risk on geometry support')
    k,n=x.shape;lo=np.zeros((k,k),np.int64);eq=np.zeros_like(lo)
    lib=ctypes.CDLL(str(so));fn=lib.risk_pair_counts
    fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    fn.restype=None;fn(x.ctypes.data,k,n,config['broad_sweep']['native_count_threads'],lo.ctypes.data,eq.ctypes.data)
    if not np.all(lo+lo.T+eq==n) or not np.array_equal(eq,eq.T):raise AssertionError('Pair count invariant')
    return {'candidate_ids':ids,'pixels':n,'less_counts':lo.tolist(),'equal_counts':eq.tolist(),'orientation':'less_counts[i,j] counts risk_i < risk_j','support':'full fixed geometry x>=4; no GT mask','subset_formula':'score_i=1+sum_j!=i(less[j,i]+0.5*equal[j,i])/pixels'}


def select_scores(scores,record,config):
    best=min(scores.values());tol=config['table3']['rank_selection']['global_score_atol'];tied=sorted(c for c,v in scores.items() if abs(v-best)<=tol)
    s=tie_seed(record,config,'Signal-Global');chosen=str(np.random.default_rng(s).choice(tied))
    return {'candidate_id':chosen,'mean_pixel_ranks':scores,'tied_candidates':tied,'tie_seed':s,'GT_used':False}

