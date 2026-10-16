import numpy as np
METHODS=['Clean','Identity','Random','Signal-Global','Signal-Pixel Fusion','Global Oracle','Pixel Oracle']
NAMES=['Global Signal gain','Global Oracle headroom','Global recovery','Pixel Signal gain','Pixel Oracle headroom','Pixel recovery','Extra pixel-wise headroom']

def quantities(i,sg,go,sp,po):
 def ratio(a,b):
  a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
  return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>1e-12)
 return [i-sg,i-go,ratio(i-sg,i-go),i-sp,i-po,ratio(i-sp,i-po),go-po]
