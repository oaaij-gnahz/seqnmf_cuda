# seqnmf_cuda
GPU acceleration of SeqNMF([paper](https://elifesciences.org/articles/38471), [original code](https://github.com/FeeLab/seqNMF)) using CuPy.
Important credit also goes to the [Python version](https://github.com/ContextLab/seqnmf) of this algorithm by [ContextLab](https://github.com/ContextLab)

Parallelism mainly comes from unrolling the computation along the `L` dimension of the procedure.

## Depedency
NumPy, CuPy

Tested with `numpy==1.24.0` and `cupy==12.1.0` with python3.11 in a Conda enviroment on Linux Ubuntu. 

## set up
Copy the subfolder `seqnmf_cuda` to your local computer, e.g. to `/path/to/this/repo/seqnmf_cuda`.

Where `/path/to/this/repo/` contains the `seqnmf_cuda` module folder.

And then do the following:

```
import sys
sys.append("/path/to/this/repo/")
import seqnmf_cuda as seqnmf
```

Now you can use SeqNMF's CuPy implementation!

## Minimal working example
See [example.py](./example.py)

## TODO
- Package as a Pip module.

