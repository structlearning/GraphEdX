import torch
from torch_geometric.nn import GCNConv
import torch_geometric as pyg
import time
import itertools
import numpy as np
from loguru import logger
import collections
from torch.nn.utils.rnn import pad_sequence


class SimGNNTensorized(torch.nn.Module):
    def __init__(self, conf):
        """
        """
        super(SimGNNTensorized, self).__init__()
        self.conf = conf
        #Conv layers
        self.conv1 = pyg.nn.GCNConv(self.conf.dataset.one_hot_dim, self.conf.model.filters_1)
        self.conv2 = pyg.nn.GCNConv(self.conf.model.filters_1, self.conf.model.filters_2)
        self.conv3 = pyg.nn.GCNConv(self.conf.model.filters_2, self.conf.model.filters_3)
        
        #Attention
        self.attention_layer = torch.nn.Linear(self.conf.model.filters_3,self.conf.model.filters_3, bias=False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        #NTN
        self.ntn_a = torch.nn.Bilinear(self.conf.model.filters_3,self.conf.model.filters_3,self.conf.model.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_a.weight)
        self.ntn_b = torch.nn.Linear(2*self.conf.model.filters_3,self.conf.model.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_b.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(self.conf.model.tensor_neurons,1))
        torch.nn.init.xavier_uniform_(self.ntn_bias)
        #Final FC
        feature_count = (self.conf.model.tensor_neurons+self.conf.model.bins) if self.conf.model.histogram else self.conf.model.tensor_neurons
        self.fc1 = torch.nn.Linear(feature_count, self.conf.model.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.conf.model.bottle_neck_neurons, 1)

    def GNN (self, data):
        """
        """
        features = self.conv1(data.x,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.conf.training.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.conf.training.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        return features

    def forward(self, batch_data,batch_data_sizes,batch_adj=None):
        """
          batch_adj is unused
        """
        q_graphs = batch_data[0::2]
        c_graphs = batch_data[1::2]
        
        qgraph_sizes = batch_data_sizes[0::2]
        cgraph_sizes = batch_data_sizes[1::2]
        
        
        query_batch = pyg.data.Batch.from_data_list(q_graphs)
            
        query_batch.x = self.GNN(query_batch)
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]
        
        corpus_batch = pyg.data.Batch.from_data_list(c_graphs)
        corpus_batch.x = self.GNN(corpus_batch)
        corpus_gnode_embeds = [g.x for g in corpus_batch.to_data_list()]

        preds = []
        q = pad_sequence(query_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()

        c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(c),dim=1).T,cgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(c@context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()
        
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())
        

        #TODO: Figure out how to tensorize this
        if self.conf.model.histogram == True:
          h = torch.histc(q@c.permute(0,2,1),bins=self.conf.model.bins)
          h = h/torch.sum(h)

          scores = torch.cat((scores, h),dim=1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds.append(score)
        p = torch.stack(preds).squeeze()

        ged = -0.5 * (qgraph_sizes + cgraph_sizes) * torch.log(p)

        # GT : 2*GED/(N_Q + N_C) -> MSE( log(GT) - log(p) )


        return ged
    
    def compute_loss(self, lb, ub, pred):
        loss = torch.nn.functional.relu(lb-pred)**2 + torch.nn.functional.relu(pred-ub)**2
        loss = torch.mean(loss)
        return loss
  