import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from Loss import ZINBLoss, MeanAct, DispAct,Cluster_loss
from sklearn.cluster import KMeans
from sklearn import metrics


class Online(nn.Module):
    def __init__(self,config):
        super(Online,self).__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(self.config["Model"]["online_input"],self.config["Model"]["online_hidden"], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.config["Model"]["online_hidden"], self.config["Model"]["online_output"], bias=True),
            nn.ReLU(inplace=True),
        )
        
        self.predictor = nn.Linear(self.config["Model"]["online_output"], self.config["Model"]["predictor"], bias=True)
        
        
    def forward(self,data):
        encoded = self.encoder(data)
        online = self.predictor(encoded)
        return online


class Target(nn.Module):
    def __init__(self,config):
        super(Target,self).__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(self.config["Model"]["target_input"],self.config["Model"]["target_hidden"], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.config["Model"]["target_hidden"], self.config["Model"]["target_output"], bias=True),
            nn.ReLU(inplace=True),
            
        )
        
    def forward(self,data):
        target = self.encoder(data)
        return target

class Augment():
    def __init__(self):
        super(Augment,self).__init__()
        
    def shuffle(self,data):
        idx = torch.randperm(data.shape[1])
        data_shuffled = data[:, idx].view(data.size())
        return data_shuffled,data
    

class PSCC(nn.Module):
    def __init__(self,config):
        super(PSCC,self).__init__()
        self.config = config
        self.aug = Augment()
        
        self.online = Online(config=self.config)
        self.target = Target(config=self.config)
        
        self.mean = nn.Sequential(nn.Linear(self.config["Model"]["online_output"],self.config["Model"]["predictor"]), MeanAct())
        self.disp = nn.Sequential(nn.Linear(self.config["Model"]["online_output"],self.config["Model"]["predictor"]), DispAct())
        self.pi = nn.Sequential(nn.Linear(self.config["Model"]["online_output"],self.config["Model"]["predictor"]), nn.Sigmoid())
        

        
    def forward(self,data,data_shuffle):
        
        online_1 = self.online(data)
        target_1 = self.target(data_shuffle).detach()
        mean_1 = self.mean(online_1)
        disp_1 = self.disp(online_1)
        pi_1 = self.pi(online_1)
        
        online_2 = self.target(data_shuffle)
        target_2 = self.online(data).detach()
        mean_2 = self.mean(online_2)
        disp_2 = self.disp(online_2)
        pi_2 = self.pi(online_2)
        
        return online_1,target_1,mean_1,disp_1,pi_1,online_2,target_2,mean_2,disp_2,pi_2


class Positive_Sample_Contrastive_Clustering(nn.Module):

    def __init__(self,config):
        super(Positive_Sample_Contrastive_Clustering, self).__init__()
        self.config = config
    
        self.model = PSCC(config=self.config)
        self.zinb_loss = ZINBLoss()
        self.cluster_loss = Cluster_loss(config=self.config)

        self.weigth  = self._initialize_weights()
        

    def _initialize_weights(self):
        return nn.Parameter(1.0e-4 * torch.ones(size=(self.config["data_length"],self.config["data_length"])))

    def shuffle(self,data):
        idx = torch.randperm(data.shape[1])
        data_shuffled = data[:, idx].view(data.size())
        return data_shuffled,data
    
    
    def contrastive_loss(self,target,mean):
        # loss of branch network learning 
        loss = 1 / 2 * torch.sum(torch.pow((target - mean), 2))
        
        return loss
    
    
    
    def Spatial_mapping_loss(self,weight,online):
        
        z_c_1 = torch.matmul(weight, online)
        loss = 1 / 2 * torch.sum(torch.pow((online- z_c_1), 2))
        return loss


    def pre_train(self,data,data_shuffle,factor):
        print("Pre-training ...")
        self.train()
        
        optimizer = Adam(self.parameters(), lr=self.config["pre_lr"])
        
        for epoch in range(1, self.config["pre_epoches"]+1):
            
            x_tensor = Variable(torch.Tensor(data))
            x_tensor_shuffle = Variable(torch.Tensor(data_shuffle))
            
            sf_tensor = Variable(torch.Tensor(factor))
            
            
            online_1,target_1,mean_1,disp_1,pi_1,online_2,target_2,mean_2,disp_2,pi_2 = self.model(x_tensor,x_tensor_shuffle)
            
            
            contrastive_loss_1 = self.contrastive_loss(target_1,mean_1)
            mapping_loss = self.Spatial_mapping_loss(self.weigth,online_1)
            loss_zinb_1 = self.zinb_loss(x=target_1, mean=mean_1, disp=disp_1, pi=pi_1, scale_factor=sf_tensor)
            
            
            contrastive_loss_2 = self.contrastive_loss(target_2,mean_2)
            mapping_loss = self.Spatial_mapping_loss(self.weigth,online_2)
            loss_zinb_2 = self.zinb_loss(x=target_2, mean=mean_2, disp=disp_2, pi=pi_2, scale_factor=sf_tensor)
            
            
            
            contrastive_loss = 0.5*(contrastive_loss_1+contrastive_loss_2)
            loss_zinb = 0.5*(loss_zinb_1+contrastive_loss_2)
            
            # loss = (0.2*loss_reconst + self.config["lambda_1"] * loss_reg + self.config["lambda_2"] * loss_selfexpress)**1/10 + loss_zinb
            loss =  (0.2*contrastive_loss +  self.config["lambda_2"] * mapping_loss)**1/10 +loss_zinb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if epoch == self.config["pre_epoches"]:
                print('Pre-training completed')




    def deep_train(self,data,data_shuffle,factor):
        # self.model.train()
        
        print("Deep training ...")
        optimizer = Adam(self.parameters(), lr=self.config["alt_lr"])
        for epoch in range(1, self.config["alt_epoches"] + 1):                                   
            x_tensor = Variable(torch.Tensor(data))
            x_tensor_shuffle = Variable(torch.Tensor(data_shuffle))
            
            sf_tensor = Variable(torch.Tensor(factor))
            online_1,target_1,mean_1,disp_1,pi_1,online_2,target_2,mean_2,disp_2,pi_2 = self.model(x_tensor,x_tensor_shuffle)
            
            #
            contrastive_loss_1 = self.contrastive_loss(target_1,mean_1)

            mapping_loss = self.Spatial_mapping_loss(self.weigth,online_1)
            loss_zinb_1 = self.zinb_loss(x=target_1, mean=mean_1, disp=disp_1, pi=pi_1, scale_factor=sf_tensor)
            #
          
            contrastive_loss_2 = self.contrastive_loss(target_2,mean_2)

            mapping_loss = self.Spatial_mapping_loss(self.weigth,online_2)
            loss_zinb_2 = self.zinb_loss(x=target_2, mean=mean_2, disp=disp_2, pi=pi_2, scale_factor=sf_tensor)
            #
            
            contrastive_loss = 0.5*(contrastive_loss_1+contrastive_loss_2)
 
            loss_zinb = 0.5*(loss_zinb_1+contrastive_loss_2)
            
            
            loss = (0.2*contrastive_loss +  self.config["lambda_2"] * mapping_loss)**(0.1) + loss_zinb
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch == self.config["alt_epoches"]:
                print('Deep training completed')
                
        return self.weigth.cpu().detach().numpy()
    
    def forwaard(self,data,factors):
        data = Variable(torch.Tensor(data)).to(self.config["device"])
        _,data_shuffle = self.shuffle(data)
        factors = Variable(torch.Tensor(factors)).to(self.config["device"])
        
        self.pre_train(data,data,factors)
        
        weight = self.deep_train(data,data,factors)
        
        return weight
