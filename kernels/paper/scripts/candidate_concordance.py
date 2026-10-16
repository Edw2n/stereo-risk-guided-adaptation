import ctypes, subprocess
import numpy as np
from sensor_v4_common import ROOT,digest

def compile_helper(storage):
    src=ROOT/'scripts/native/candidate_concordance.cpp'
    folder=storage/'cache/native_helpers'/('candidate_axis_'+digest(src.read_bytes())[:16]);folder.mkdir(parents=True,exist_ok=True)
    lib=folder/'candidate_concordance.so'
    if not lib.exists():subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(src),'-o',str(lib)],check=True)
    return lib

def concordance(risk,error,lib):
    r=np.ascontiguousarray(risk.T,dtype=np.float32);e=np.ascontiguousarray(error.T,dtype=np.float32)
    out=np.empty(r.shape[0],dtype=np.float64)
    fn=ctypes.CDLL(str(lib)).candidate_concordance
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int,ctypes.c_void_p];fn.restype=None
    fn(r.ctypes.data,e.ctypes.data,len(r),r.shape[1],out.ctypes.data)
    return out
