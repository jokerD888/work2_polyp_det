"""Threshold-constrained maximum-cardinality coverage, not greedy matching."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def xyxy(boxes):
    a=np.asarray(boxes,dtype=np.float64).reshape(-1,4).copy(); a[:,2:]+=a[:,:2]; return a


def iou_matrix(a,b):
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(2)
    return intersection/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+
                                  np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-intersection,1e-12)


def maximum_coverage(matrix,threshold,k=None):
    a=np.asarray(matrix)
    if k is not None: a=a[:,:k]
    if not a.size: return 0
    match=maximum_bipartite_matching(csr_matrix(a>=threshold),perm_type="column")
    return int(np.count_nonzero(match>=0))


def score_order(rows):
    return sorted(rows,key=lambda p:-p["score"])
