import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x

class Cluster_loss(nn.Module):
    
    def __init__(self, config, alpha=1.0):
        super(Cluster_loss, self).__init__()
        self.config = config
        self.in_features = self.config["Model"]["predictor"]
        self.out_features = self.config["Model"]["predictor"]
        self.alpha = alpha
        self.tau = 1
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))     # Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True)
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        dist1 = self.tau*torch.sum(torch.square(x.unsqueeze(1) - self.weight), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        kmeans_loss = torch.mean(torch.sum(dist2, dim=1))
        return dist1, dist2, kmeans_loss
    

# class reconst_loss(nn.Module):
#     def __init__(self):
#         super(reconst_loss,self).__init__()
    
#     def forward(self,target,mean):
#         loss = 1 / 2 * torch.sum(torch.pow((target - mean), 2))
#         return loss
    
# class regression_loss(nn.Module):
#     def __init__(self) :
#         super(regression_loss,self).__init__()
        
#     def forward(self,weight):
#         loss = torch.sum(torch.pow(weight, 2))
#         return loss
    
# class selfexpress_loss(nn.Module):
#     def __init__(self):
#         super(selfexpress_loss).__init__()
        
#     def forward(self,weight,online):
#         z_c_1 = torch.matmul(weight, online)
#         loss = 1 / 2 * torch.sum(torch.pow((online- z_c_1), 2))
#         return loss