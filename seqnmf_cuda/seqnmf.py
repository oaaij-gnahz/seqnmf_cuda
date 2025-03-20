# Implemntation of seqNMF algorithm in Python/CuPy
# https://github.com/FeeLab/seqNMF/blob/master/seqNMF.m
# Interface difference: Here W is LxNxK, instead of NxKxL; to leverage column-major ordering in Python
import time
from math import ceil

import numpy as np
import cupy as cp
import cupyx.scipy.signal as cupy_sig
# import scipy.signal as cupy_sig
# cp = np

from .helper import reconstruct_padded
from .helper import shift_factors_padded
from . import global_params as GLOBAL_PARAMS

# def x():
#     print(GLOBAL_PARAMS.BATCH_SIZE_BYTES, GLOBAL_PARAMS.NP_SEED, GLOBAL_PARAMS.CP_SEED)
#     print(GLOBAL_PARAMS._NP_RNG, GLOBAL_PARAMS._CP_RNG)

def run_seqnmf_cuda(X_cuda : cp.ndarray, algo_params : dict, **kwargs):
    """
    The function to call for SeqNMF on CUDA. All return tensors are on GPU

    Parameters
    ----------
    X_cuda : (N, T) cupy array where N is # units and T is # samples.
    algo_params : dict of SeqNMF algorithm parameters.
    kwargs : 
        batch_size : how many timeshifts to calculate in parallel
            Should be an integer between 1 and L.
            It will take ceil(L/batch_size) for-loop iterations to finish one iteration of the algorithm
        verbose : bool
    
    Returns
    ----------
    W_cuda : (L, N, K) the kernels; L is the temporal length of each kernel and K is # kernels.
    H_cuda : (K, T) the dim-reduced timeseries.
    cost : (maxiters+1, ) training reconstruction error history.
        `cost[0]` is the initial reconstruction error of the randomly initiazed W and H
        `cost[i] (i > 0)` is the reconstruction error after i-th iteration.
    """
    assert cp.all(X_cuda>=0), "X must be non-negative"
    batch_size = kwargs.pop("batch_size", None)
    verbose = kwargs.pop("verbose", True)
    if batch_size is None:
        batch_size = int(ceil(GLOBAL_PARAMS.BATCH_SIZE_BYTES/X_cuda.nbytes))
    if verbose:
        print("In <run_seqnmf_cuda>, batchsize=%d"%(batch_size))

    X_cuda, N, T, K, L, params = _parse_seqnmf_params(X_cuda, **algo_params)
    batch_size = max(min(batch_size, L), 1)
    p_maxiter = params["maxiter"]
    p_tolerance = params["tolerance"]
    p_lambda = params["lambda_"]
    p_lambdaL1H = params["lambdaL1H"]
    p_lambdaOrthoH = params["lambdaOrthoH"]
    p_lambdaOrthoW = params["lambdaOrthoW"]
    p_lambdaL1W = params["lambdaL1W"]
    p_shift = params["shift"]
    p_useWupdate = params["useWupdate"]
    W_cuda = params["W_init"]
    H_cuda = params["H_init"]
    Xhat = reconstruct_padded(W_cuda, H_cuda)
    mask = (params["M"]==0)
    # print(np.any(mask)) # Gives False
    # print("----")
    X_cuda[mask] = Xhat[mask] # Author says masked data are replaced by reconstructed data so that they do not affect fit
    
    lasttime = 0
    smoothkernel = cp.ones((1, 2*L-1))
    cross_tempo_terms = (-cp.eye(K)+1)
    cp_eps = cp.finfo(float).eps
    smallnum = cp.max(X_cuda)*1e-6

    # initialize cost
    cost = cp.empty(p_maxiter+1, dtype=float)
    cost[:] = cp.nan
    cost[0] = cp.sqrt(cp.mean((X_cuda-Xhat)**2))
    print("Cost at initialization = %.4f" % (cost[0]))
    # temporary reusable buffers without needing to initialize
    X_buf = cp.empty((batch_size, N, T), dtype=X_cuda.dtype)
    Xhat_buf = cp.empty((batch_size, N, T), dtype=X_cuda.dtype)
    # WTX_buf = cp.empty((K, T), dtype=X_cuda.dtype)
    # WTXhat_buf = cp.empty((K, T), dtype=X_cuda.dtype)
    # reusable arrays that need to be initialized
    WTX = cp.zeros((K, T), dtype=X_cuda.dtype)
    WTXhat = cp.zeros((K, T), dtype=X_cuda.dtype)
    ### regularization for H update
    dRdH_buf = cp.empty_like(WTX)
    dHHdH_buf = cp.empty_like(WTX)
    ### for W update
    H_shifted_buf = cp.empty((batch_size, K, T))
    # best config
    best_W_cuda = cp.copy(W_cuda)
    best_H_cuda = cp.copy(H_cuda)
    best_cost = cost[0]
    best_iter = 0
    # iters
    ts = time.time()
    for iter in range(p_maxiter):
        # TODO move the buffer allocations out of the loop where doable
        if ( (iter==p_maxiter-1) or (iter>=5 and cost[iter]+p_tolerance)>=cp.mean(cost[iter-5:iter]) ):
            if verbose:
                print("Converged at iteration %d" % (iter+1))
            lasttime = 1
            if iter>0:
                p_lambda = 0 # In AUTHOR's implementation, final round is unregularized
        
        # standarmd ConvNMF partial H
        WTX[:] = 0
        WTXhat[:] = 0
        # X_buf[:] = 0
        # Xhat_buf[:] = 0
        # WTX_buf[:] = 0
        # WTXhat_buf[:] = 0
        l_proc = 0
        while l_proc < L:
            l_next = min(l_proc+batch_size, L)
            bsz_this = l_next - l_proc
            for i in range(bsz_this):
                # left shift
                X_buf[i, :, :T-(l_proc+i)] = X_cuda[:, (l_proc+i):]
                X_buf[i, :, T-(l_proc+i):] = 0 # No need to do circuilar shift
                Xhat_buf[i, :, :T-(l_proc+i)] = Xhat[:, (l_proc+i):]
                Xhat_buf[i, :, T-(l_proc+i):] = 0 # No need to do circuilar shift
            WTX_buf    = cp.einsum("lnk,lnt->kt", W_cuda[l_proc:l_next, :, :], X_buf[:bsz_this, :, :])#, out=WTX_buf)
            WTXhat_buf = cp.einsum("lnk,lnt->kt", W_cuda[l_proc:l_next, :, :], Xhat_buf[:bsz_this, :, :])#, out=WTXhat_buf)
            WTX    += WTX_buf
            WTXhat += WTXhat_buf
            l_proc = l_next
        # regularization for H update
        # dRdH_buf = cp.empty_like(WTX)
        # dHHdH_buf = cp.empty_like(WTX)
        cp.dot(cross_tempo_terms, cupy_sig.convolve2d(WTX, smoothkernel, mode="same"), out=dRdH_buf)
        cp.dot(cross_tempo_terms, cupy_sig.convolve2d(H_cuda, smoothkernel, mode="same"), out=dHHdH_buf)
        dRdH_buf *= p_lambda
        dHHdH_buf *= p_lambdaOrthoH
        dRdH_buf += dHHdH_buf + p_lambdaL1H
        # update H
        H_cuda *= (WTX/(WTXhat+dRdH_buf+cp_eps))
        
        # shift to center factors
        if p_shift:
            W_cuda, H_cuda = shift_factors_padded(W_cuda, H_cuda)
            W_cuda += smallnum

        # renormalize rows of H
        norms = cp.sqrt(cp.sum(H_cuda**2, axis=1))[:,None] # TODO preallocate
        H_cuda /= (norms+cp_eps)
        W_cuda *= norms.squeeze()
        

        # TODO fix: dRdW should have an extra dim of L
        if not params["W_fixed"]:
            Xhat = reconstruct_padded(W_cuda, H_cuda)
            X_cuda[mask] = Xhat[mask] # Author says masked data are replaced by reconstructed data so that they do not affect fit
            if p_lambdaOrthoW>0:
                # Wflat = cp.sum(W_cuda, axis=0) # (NxK)
                # dWWdW = p_lambdaOrthoW*cp.dot(Wflat, cross_tempo_terms) # NxK
                dWWdW = p_lambdaOrthoW*cp.dot(cp.sum(W_cuda, axis=0), cross_tempo_terms) # NxK
            else:
                dWWdW = 0
            if p_lambda>0 and p_useWupdate:
                XS = cupy_sig.convolve2d(X_cuda, smoothkernel, mode="same")
            # H_shifted_buf = cp.empty((batch_size, K, T))
            l_proc = 0
            while l_proc < L:
                l_next = min(l_proc+batch_size, L)
                bsz_this = l_next - l_proc
                for i in range(bsz_this):
                    H_shifted_buf[i, :, (l_proc+i):] = H_cuda[:, :T-(l_proc+i)]
                    H_shifted_buf[i, :, :(l_proc+i)] = 0 # No need to do circuilar shift
                XHT_batch = cp.einsum("nt,lkt->nk", X_cuda, H_shifted_buf[:bsz_this, :, :])
                XhatHT_batch = cp.einsum("nt,lkt->nk", Xhat, H_shifted_buf[:bsz_this, :, :])
                # regularization for W update
                if p_lambda>0 and p_useWupdate:
                    dRdW = p_lambda*cp.einsum("nt,lkt,km->lnm", XS, H_shifted_buf[:bsz_this, :, :], cross_tempo_terms) # BxNxK
                else:
                    dRdW = 0
                dRdW += dWWdW + p_lambdaL1W
                # update W
                W_cuda[l_proc:l_next, :, :] *= XHT_batch/(XhatHT_batch+dRdW+cp_eps)
                l_proc = l_next
        
        Xhat = reconstruct_padded(W_cuda, H_cuda)
        X_cuda[mask] = Xhat[mask] # Author says masked data are replaced by reconstructed data so that they do not affect fit
        cost[iter+1] = cp.sqrt(cp.mean((X_cuda-Xhat)**2))
        if cost[iter+1] < best_cost:
            best_cost = cost[iter+1]
            best_H_cuda[:] = H_cuda
            best_W_cuda[:] = W_cuda
            best_iter = iter+1
        if verbose:
            print("Cost after iteration %d = %.4f; total elapsed=%.2f sec" % (iter+1, cost[iter+1], time.time()-ts))
        if lasttime:
            break
    # end of iters
    res_dict = {
        "W_cuda": W_cuda,
        "H_cuda": H_cuda[:, L:-L],
        "cost": cost.get(), # cost should be returned on host memory
        "best_W_cuda": best_W_cuda,
        "best_H_cuda": best_H_cuda[:, L:-L],
        "best_iter": best_iter,
        "last_iter": iter+1
    }
    # input("Paused ... Press Enter to continue")
    # return W_cuda, H_cuda[:, L:-L], cost
    return res_dict



def _parse_seqnmf_params(X_cuda, **kwargs):
    params = {}
    params['K'] = kwargs.pop('K', 10)
    params['L'] = kwargs.pop('L', 100)
    params['lambda_'] = kwargs.pop('lambda_', 0.001)
    params['maxiter'] = kwargs.pop('maxiter', 100)
    params['tolerance'] = cp.asarray(kwargs.pop('tol', -cp.Inf))
    params['shift'] = kwargs.pop('shift',True)
    params['lambdaL1W'] = kwargs.pop('lambdaL1W',0)
    params['lambdaL1H'] = kwargs.pop('lambdaL1H',0)
    params['W_fixed'] = kwargs.pop('W_fixed', False)
    params['W_init'] = kwargs.pop('W_init', None) # depends on K--initialize post parse
    params['H_init'] = kwargs.pop('H_init', None) # depends on K--initialize post parse
    params['SortFactors'] = kwargs.pop('sortFactors', True) # sort factors by loading?
    params['lambdaOrthoW'] = kwargs.pop('lambdaOrthoW', 0) # for this regularization: ||Wflat^TWflat||_1,i!=j
    params['lambdaOrthoH'] = kwargs.pop('lambdaOrthoH', 0) # for this regularization: ||HSH^T||_1,i!=j
    params['useWupdate'] = kwargs.pop('useWupdate', True) # W update for cross orthogonality often doesn't change results much, and can be slow, so option to skip it 
    params['M'] = kwargs.pop('M', None) # Masking matrix: default is ones; set elements to zero to hold out as masked test set
    L = params['L']
    K = params['K']
    # zeropad data matrix X by L
    X_cuda = cp.pad(X_cuda, [(0,0), (L, L)], mode='constant', constant_values=0)
    N, T = X_cuda.shape
    if params["W_init"] is None:
        # Will throw error if W_init is array
        params["W_init"] = GLOBAL_PARAMS._CP_RNG.uniform(low=0, high=cp.max(X_cuda), size=(L, N, K))
    if params["H_init"] is None:
        params["H_init"] = GLOBAL_PARAMS._CP_RNG.uniform(low=0, high=cp.max(X_cuda), size=(K, T))/np.sqrt(T/3)
    else:
        params["H_init"] = cp.pad(cp.asarray(params["H_init"]), [(0,0), (L,L)], mode='constant', constant_values=0)
    if params["M"] is None:
        params["M"] = cp.ones((N,T))
    else:
        params["M"] = cp.pad(cp.asarray(params["M"]), [(0,0), (L,L)], mode='constant', constant_values=1)
    return X_cuda, N, T, K, L, params
