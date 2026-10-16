import numpy as np
from scipy.stats import rankdata

def ensemble_risk(parts):
    n=parts[0].shape[0]
    if n==1:return np.zeros_like(parts[0],dtype=np.float32)
    v=np.zeros(parts[0].shape,np.float64)
    for part in parts:v+=(rankdata(part,axis=0,method='average')-1)/(n-1)
    return (v/len(parts)).astype(np.float32)

