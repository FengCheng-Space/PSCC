import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize as nor
import torch
import random
import yaml
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns


def cluster_acc(y_true, y_pred, name=None, path=None):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, 1


def eva(y_true, y_pred,pp=True, name=None, path=None):
    # acc, f1 = cluster_acc(y_true, y_pred, name, path)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    #nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = ari_score(y_true, y_pred)
    if pp:
        print( ' nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
    return  nmi, ari

def seed_all(seed):
    
    if not seed:
        seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)



def read_config(config_path):
    ''' 
    Read the parameters to obtain the algorithm model's default parameters from the.yml configuration file.
    '''
    
    file = open(config_path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict


def post_proC(C, K, d=11, alpha=7.0): #
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = nor(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize').fit(L)
    pred_lable = spectral.fit_predict(L) + 1
    return pred_lable, L


def work_device():
    """
    Check the equipment of the machine so that the model algorithm training process obtains high performance equipment.
    Select 'CUDA' or 'cpu' as training device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Pytorch version: ",torch.__version__)
    print("Training device: ",device,"\n")
    return device


def Zero_percent(data):
    """ 
    Compute this data "Zero" percent in all file data matrix. 
    """
    cnt_array = np.where(data,0,1)
    
    P_zero = np.sum(cnt_array)/np.sum(data)
    
    return P_zero

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self,savepath, patience=7, verbose=False, delta=0.1,):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            savepath(char): model save path
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.savepath = savepath

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model,self.savepath)
        self.val_loss_min = val_loss
        
def Cluster_heatmap_Visualization(Embed_data,lable,title=""):
    data = pd.DataFrame(Embed_data,index=None)
    data["lable"] = lable
    data_copy = data.sort_values(by="lable",ascending=True)
    data_copy.drop("lable",axis=1)
    sns.heatmap(data_copy.T.corr())
    
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    

def Cluster_Visualization(Embeding_data,lable,title=""):
    Embeding_2D = TSNE(n_components=2).fit_transform(Embeding_data)
    Embeding_2D = pd.DataFrame(Embeding_2D,columns=["x","y"])
    Embeding_2D["lable"] = lable
    cluster_n = len(np.unique(lable))
    
    sns.set_style("white")
    sns.scatterplot(x="x", y="y",hue="lable", data=Embeding_2D,palette=sns.color_palette("hls",cluster_n ))
    plt.xlabel("")
    plt.ylabel("")
    plt.legend([])
    plt.title(title) 

