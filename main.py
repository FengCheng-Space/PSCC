import h5py
import numpy as np
from preprocess import preprocess
import scanpy as sc
from Utils import  post_proC,seed_all,read_config,work_device,eva
from PSCC import Positive_Sample_Contrastive_Clustering
from sklearn.cluster import KMeans
from torch.autograd import Variable
import torch
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
seed_all(seed=20114687)

if __name__ == "__main__":

    path = './Example_data/Quake_10x_Bladder.h5'

    adata,lable = preprocess(path)
    device = work_device()
    n_cluster = len(np.unique(lable))

    # Parameters overwriting
    config = read_config(config_path="./config.yml")
    config["data_length"] = len(adata.X)
    config["device"] = device 
    config["n_cluster"]  = n_cluster
    config["Model"]["online_input"] = adata.X.shape[1]
    config["Model"]["target_input"] = adata.X.shape[1]

    model = Positive_Sample_Contrastive_Clustering(config=config).to(device)

    weight = model.forwaard(adata.X,adata.obs.size_factors)


    pred_label, L = post_proC(weight,n_cluster)                
    y = lable.astype(np.int64)
    pred_label = pred_label.astype(np.int64)
    NMI, ARI = eva(y, pred_label)

    em = model.model.online(Variable(torch.Tensor(adata.X)).to(device))