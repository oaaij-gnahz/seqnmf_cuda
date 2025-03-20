import numpy as np
import cupy as cp
# cp = np
NP_SEED = 0
CP_SEED = 0
_NP_RNG = None
_CP_RNG = None

# default batch size in bytes, this can get overwritten if batch_size is provided in function calls
# Source of parallelization is how many shifts can be processed at once in parrallel
BATCH_SIZE_BYTES = (1<<28) # In conv-like operation, 

def _init_rng_from_seed():
    global _NP_RNG, _CP_RNG
    _NP_RNG = np.random.default_rng(NP_SEED)
    _CP_RNG = cp.random.default_rng(CP_SEED)

def initialize():
    # CAN EXPAND WITH OTHER INITIALIZATIONS
    _init_rng_from_seed()
