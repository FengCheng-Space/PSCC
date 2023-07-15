from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import h5py



def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, select_hvg=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if logtrans_input:
        sc.pp.log1p(adata)

    if select_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]

    if size_factors or normalize_input or logtrans_input or select_hvg:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0


    if normalize_input:
        sc.pp.scale(adata)

    return adata

def preprocess(path):
    data_mat = h5py.File(path)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                    transpose=False,
                    test_split=False,
                    copy=True)

    adata = normalize(adata,
                    size_factors=True,
                    normalize_input=True, 
                    select_hvg=True)
    return adata,y