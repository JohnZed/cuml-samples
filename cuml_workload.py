"""cuml_workload.py: Simple cuML examples to generate load

Run with --help for documentation or just run everything with:
  cuml_workload.py --size large

"""

import cuml
import cudf
import xgboost

import pandas as pd
import numpy as np
import gzip
import os
import time
import argparse

# These load functions are adapted from the cuML
# benchmark notebook in the RAPIDS notebooks repo
def load_data_mortgage_Xy(nrows, ncols, dtype=np.float32, cached = 'data/mortgage.npy.gz'):
    if os.path.exists(cached):
        print('Loading mortgage data...')
        with gzip.open(cached) as f:
            X = np.load(f)
        idx = np.random.randint(0,X.shape[0]-1,nrows)
        y = X[idx, -1].astype(dtype)
        X = X[idx,:ncols].astype(dtype)
    else:
        raise IOError("Could not load mortgage data from %s" % cached)

    return X, y


unsupervised_algos = dict(
    kmeans=cuml.cluster.KMeans(n_clusters=8, max_iter=300),
    dbscan=cuml.cluster.DBSCAN(eps=3, min_samples=2),
    umap=cuml.manifold.UMAP(n_neighbors=5, n_epochs=500)
    )


def run_xgboost(X, y):
    # Use the params from mortgage example, but more roudns
    xgb_params = dxgb_gpu_params = {
        'nround':            2000,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'min_child_weight':  30,
        'tree_method':       'gpu_hist',
        'n_gpus':            1,
        'loss':              'ls',
        'objective':         'reg:linear',
        'max_features':      'auto',
        'criterion':         'friedman_mse',
        'grow_policy':       'lossguide',
        'verbose':           True
    }
    # XXX using public pip version of xgboot which does not support cudf yet
    train_data = xgboost.DMatrix(data=X, label=y)
    t0 = time.time()  # Begin tracing here
    result = xgboost.train(xgb_params, dtrain=train_data)
    t1 = time.time()
    return t1 - t0

def run_benchmark(algo_name, nrows=1000, ncols=101, dtype=np.float32,
                  cached='data/mortgage.npy.gz'):
    X, y = load_data_mortgage_Xy(nrows, ncols, dtype, cached)

    print("Running: ", algo_name)
    if algo_name.lower() == 'xgboost':
        timing = run_xgboost(X, y)
    else:
        algo = unsupervised_algos[algo_name.lower()]

        t0 = time.time()  # Begin tracing here
        algo.fit(X)
        t1 = time.time()
        timing = t1 - t0

    print("Complete, wall time to run %s on %d x %d data = %10.5f" %
          (algo_name, nrows, ncols, timing))

    return timing

if __name__ == '__main__':
    ALL_ALGO_NAMES = ['xgboost', *unsupervised_algos.keys()]
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithms', nargs='*',
                        help='Algorithms to run or omit to run all, options: %s' % (
                            "\n  ".join(ALL_ALGO_NAMES)))
    parser.add_argument('--size', default='small', type=str,
                        help='Dataset size [small,large,<nrows>x<ncols>]')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'float64'],
                        help='Computation data type')
    parser.add_argument('--dataset', default='data/mortgage.npy.gz',
                        help='Path to the dataset used for benchmarking')
    args = parser.parse_args()
    args.dtype = np.float32 if args.dtype == 'float32' else np.float64

    print("CuML: %s, CuDF: %s, XGboost: %s" %
          (cuml.__version__, cudf.__version__, xgboost.__version__))

    np.random.seed(42)
    if args.algorithms:
        algo_names = args.algorithms
    else:
        algo_names = ALL_ALGO_NAMES

    if args.size == 'small':
        nrows = 10000
        ncols = 81
    elif args.size == 'large':
        nrows = 3000183 # datasets have arbitrary, not powers-of-2 sizes often
        ncols = 401     # about 4.8gb of raw data, large but fits on GPU
    else: # assumed to be dataset dimension specified in the form of <nrows>x<ncols>
        dims = args.size.split('x')
        nrows = int(dims[0])
        ncols = int(dims[1])

    for a in algo_names:
        run_benchmark(a, nrows=nrows, ncols=ncols, dtype=args.dtype,
                      cached=args.dataset)
