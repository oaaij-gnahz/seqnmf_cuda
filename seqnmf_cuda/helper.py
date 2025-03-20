import gc

import numpy as np
import cupy as cp
# cp = np

from . import global_params as GLOBAL_PARAMS
# from .global_params import (
#     BATCH_SIZE_BYTES,
#     NP_SEED,
#     CP_SEED,
#     _NP_RNG,
#     _CP_RNG,
#     _init_rng_from_seed
# )
# from cupy.cuda.runtime import freeArray as cp_free
# cp = np


def set_batch_size_bytes(batch_size_bytes):
    GLOBAL_PARAMS.BATCH_SIZE_BYTES = batch_size_bytes

def get_batch_size_bytes():
    return GLOBAL_PARAMS.BATCH_SIZE_BYTES

def set_seed(np_seed, cp_seed):
    GLOBAL_PARAMS.NP_SEED = np_seed
    GLOBAL_PARAMS.CP_SEED = cp_seed
    GLOBAL_PARAMS._init_rng_from_seed()

def reconstruct_padded(W_cuda, H_cuda, batch_size=None, verbose=True):
    """
    Return reconstructed data; this function assumes H is already padded
    W_cuda: LxNxK
    H_cuda: KxT_padded
    """
    L, N, K = W_cuda.shape # LxNxK
    T_padded = H_cuda.shape[1] # KxTpadded
    if batch_size is None:
        # by default allocate 4GB of VRAM for unrolled H_cuda
        # assuming H_cuda is uses float64 (8B per element)
        batch_size = int(np.floor(GLOBAL_PARAMS.BATCH_SIZE_BYTES/(H_cuda.shape[0]*H_cuda.shape[1]*8)))
    batch_size = max(min(batch_size, L), 1)
    if verbose:
        print("In <reconstruct_padded>: batch_size=%d"%(batch_size))
    # set padded area to zero
    # H_cuda[:, :L] = 0
    # H_cuda[:, -L:] = 0
    Xhat_cuda = cp.zeros((N, T_padded))
    einsum_out_buf = cp.empty_like(Xhat_cuda)
    # actually padding may not even be necessary if we manually unroll H_cuda.
    # However we still assume padding just to be sort-of (not completely) consistent with MATLAB code
    # It seems the original MATLAB code also padded twich (once in initialization, once in this function) 
    # which I think is just extra precaution
    H_cbuf = cp.empty((batch_size, K, T_padded))
    l_proc = 0
    while l_proc < L:
        l_next = min(l_proc+batch_size, L)
        bsz_this = l_next - l_proc
        for i in range(bsz_this):
            H_cbuf[i, :, (l_proc+i):] = H_cuda[:, :(T_padded-(l_proc+i))]
            H_cbuf[i, :, :(l_proc+i)] = 0 # No need to do circuilar shift
        einsum_out_buf[:] = cp.einsum(
            "lnk,lkt->nt",
            W_cuda[l_proc:l_next, :, :],
            H_cbuf[:bsz_this, :, :],
            # out=einsum_out_buf
            )
        Xhat_cuda += einsum_out_buf
        l_proc = l_next
    # not sure if relevant/necessary: release VRAM
    del(H_cbuf)
    del(einsum_out_buf)
    gc.collect()
    return Xhat_cuda


def reconstruct(W_cuda, H_cuda, batch_size=None):
    """
    Return reconstructed data; padding happens within the function
    W_cuda: LxNxK
    H_cuda: KxT_padded
    """
    L, N, K = W_cuda.shape # LxNxK
    T = H_cuda.shape[1] # KxT
    H_cuda = cp.pad(H_cuda, ((0,0),(L,L)), mode="constant", constant_values=0)
    T_padded = T + 2*L
    if batch_size is None:
        # by default allocate 4GB of VRAM for unrolled H_cuda
        batch_size = int(np.floor(2e9/(H_cuda.shape[0]*H_cuda.shape[1]*8)))
    batch_size = max(min(batch_size, L), 1)
    Xhat_cuda = cp.zeros((N, T_padded))
    einsum_out_buf = cp.empty_like(Xhat_cuda)
    # actually padding may not even be necessary if we manually unroll H_cuda.
    # However we still assume padding just to be sort-of (not completely) consistent with MATLAB code
    # It seems the original MATLAB code also padded twich (once in initialization, once in this function) 
    # which I think is just extra precaution
    H_cbuf = cp.empty((batch_size, K, T_padded))
    l_proc = 0
    while l_proc < L:
        l_next = min(l_proc+batch_size, L)
        bsz_this = l_next - l_proc
        for i in range(bsz_this):
            H_cbuf[i, :, (l_proc+i):] = H_cuda[:, :(T_padded-(l_proc+i))]
            H_cbuf[i, :, :(l_proc+i)] = 0 # No need to do circuilar shift
        einsum_out_buf[:] = cp.einsum(
            "lnk,lkt->nt",
            W_cuda[l_proc:l_next, :, :],
            H_cbuf[:bsz_this, :, :],
            # out=einsum_out_buf
            )
        Xhat_cuda += einsum_out_buf
        l_proc = l_next
    # not sure if relevant/necessary: release VRAM
    del(H_cbuf)
    del(einsum_out_buf)
    gc.collect()
    return Xhat_cuda[:, L:-L]


def shift_factors_padded(W_cuda, H_cuda):
    L, N, K = W_cuda.shape # LxNxK
    T_padded = H_cuda.shape[1] # KxTpadded
    if L==1:
        return W_cuda, H_cuda
    center = max(L//2, 1)
    # TODO the following code can be directly applied in place without copying
    Wshift = cp.copy(W_cuda)#cp.pad(W_cuda, ((L,L), (0,0), (0,0)), mode='constant') # 2L x N x K
    Hshift = cp.copy(H_cuda)
    w_compacted = cp.sum(W_cuda, axis=1) # LxK
    wmass = w_compacted/cp.sum(w_compacted, axis=0) # LxK
    # TODO nan values should be treated as zero (no shift)
    cmass_k = cp.einsum("lk,l->k", wmass, cp.arange(L, dtype=wmass.dtype)).get() # (K,) center of mass for each motif
    # np.nan_to_num(cmass_k, copy=False, nan=center) # 
    for k in range(K):
        c = cmass_k[k]
        if np.isnan(c):
            Wshift[:, :, k] = W_cuda[:, :, k]
            Hshift[k, :] = H_cuda[k, :]
            continue
        d = int(np.round(c))-center
        # print("    ", c, center)
        if d==0:
            Wshift[:, :, k] = W_cuda[:, :, k]
            Hshift[k, :] = H_cuda[k, :]
            continue
        elif d > 0:
            # leftshift W and rightshift H 
            Wshift[:-d, :, k] = W_cuda[d:, :, k]
            Wshift[-d:, :, k] = 0
            Hshift[k, d:] = H_cuda[k, :-d]
            Hshift[k, :d] = 0
        else:
            d = -d
            # rightshift W and leftshift H
            Wshift[d:, :, k] = W_cuda[:-d, :, k]
            Wshift[:d, :, k] = 0
            Hshift[k, :-d] = H_cuda[k, d:]
            Hshift[k, -d:] = 0
    del(w_compacted)
    del(wmass)
    del(cmass_k)
    gc.collect()
    return Wshift, Hshift

