import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch_scatter
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
import torch_geometric as pyg


def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'min', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0

    return out

    
class SETensorNetworkModule(torch.nn.Module):
    def __init__(self,args, dim_size):
        super(SETensorNetworkModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim_size*2
        reduction = 4
        self.fc_se = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

        # self.fc0 = nn.Sequential(
        #                 nn.Linear(channel,  channel),
        #                 nn.ReLU(inplace = True),
        #                 nn.Linear(channel, channel),
        #                 nn.ReLU(inplace = True)
        #         )

        self.fc1 = nn.Sequential(
                        nn.Linear(channel,  channel),
                        nn.ReLU(inplace = True),
                         nn.Linear(channel, self.dim_size // 2), #nn.Linear(channel, self.args.tensor_neurons),
                        nn.ReLU(inplace = True)
                )

    def forward(self, embedding_1, embedding_2):

        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        se_feat_coefs = self.fc_se(combined_representation)
        se_feat = se_feat_coefs * combined_representation + combined_representation
        scores = self.fc1(se_feat)

        return scores


class SEAttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        super(SEAttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

    def forward(self, x):
        x = self.fc(x)
        return x


class AttentionModule(torch.nn.Module):
    def __init__(self, args, dim_size):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size)) 
        self.weight_matrix1 = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size))

        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Tanh()
                )

        self.fc1 =  nn.Linear(channel,  channel)
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        attention = self.fc(x)
        x = attention * x + x

        size = batch[-1].item() + 1 if size is None else size # size is the quantity of batches: 128 eg
        mean = scatter_('mean', x, batch, dim_size=size) # dim of mean: 128 * 16

        transformed_global = \
        torch.tanh(torch.mm(mean, self.weight_matrix)) 
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1)) # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x 

        return scatter_('add', weighted, batch, dim_size=size) # 128 * 16
        
    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))





class EGSCT_generator(torch.nn.Module):
    def __init__(self, conf):
        super(EGSCT_generator, self).__init__()
        self.conf = conf
        self.args = conf.model
        self.number_labels = self.conf.dataset.one_hot_dim
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = (self.args.filters_1 + self.args.filters_2 + self.args.filters_3) // 2

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))

            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)

        elif self.args.gnn_operator == 'gat':
            self.convolution_1 = GATConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)

        elif self.args.gnn_operator == 'sage':
            self.convolution_1 = SAGEConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.attention_level3 = AttentionModule(self.args, self.args.filters_3 * self.scaler_dim)
        self.attention_level2 = AttentionModule(self.args, self.args.filters_2 * self.scaler_dim)
        self.attention_level1 = AttentionModule(self.args, self.args.filters_1 * self.scaler_dim)

        
        self.tensor_network_level3 = SETensorNetworkModule(self.args,dim_size=self.args.filters_3 * self.scaler_dim)
        self.tensor_network_level2 = SETensorNetworkModule(self.args,dim_size=self.args.filters_2 * self.scaler_dim)
        self.tensor_network_level1 = SETensorNetworkModule(self.args,dim_size=self.args.filters_1 * self.scaler_dim)
        
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)

        self.score_attention = SEAttentionModule(self.args, self.feature_count)


    def convolutional_pass_level1(self, edge_index, features):
        """
        Making convolutional pass.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.conf.training.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features):
        features_2 = self.convolution_2(features, edge_index)
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.conf.training.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features):
        features_3 = self.convolution_3(features, edge_index)
        features_3 = F.relu(features_3)
        features_3 = F.dropout(features_3, p=self.conf.training.dropout, training=self.training)
        return features_3

        
    def forward(self, data):

        q_graphs = data[0::2]
        c_graphs = data[1::2]      
        
        query_batch = pyg.data.Batch.from_data_list(q_graphs)
        corpus_batch = pyg.data.Batch.from_data_list(c_graphs)

        edge_index_1 = query_batch.edge_index
        edge_index_2 = corpus_batch.edge_index
        features_1 = query_batch.x
        features_2 = corpus_batch.x

        batch_1 = query_batch.batch
        batch_2 = corpus_batch.batch
   

        # edge_index_1 = data["g1"].edge_index
        # edge_index_2 = data["g2"].edge_index
        # features_1 = data["g1"].x
        # features_2 = data["g2"].x

        # batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        # batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        
        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2)

        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2)

        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) # 128 * 32
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2)

        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)

        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        
        scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))
        
        return  scores 

class EGSCT_classifier(torch.nn.Module):
    def __init__(self, conf):
        super(EGSCT_classifier, self).__init__()
        self.conf = conf
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.scoring_layer = torch.nn.Linear(self.conf.model.bottle_neck_neurons, 1)

    def forward(self, scores):
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1) # dim of score: 128 * 0

        return  score 
    

class SE_EGSCT(torch.nn.Module):
    def __init__(self, conf):
        super(SE_EGSCT, self).__init__()
        self.conf = conf
        self.epsilon = 1e-9
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.generator = EGSCT_generator(self.conf)
        self.classifier = EGSCT_classifier(self.conf)

    def forward(self, data, batch_data_sizes):
        qgraph_sizes = batch_data_sizes[0::2]
        cgraph_sizes = batch_data_sizes[1::2]

        scores = self.generator(data)
        score = self.classifier(scores)
        score = -0.5 * (qgraph_sizes + cgraph_sizes) * torch.log(score + self.epsilon)
        return  score
    
    def compute_loss(self, lower_bound, upper_bound, out):
        loss = (
            torch.nn.functional.relu(lower_bound - out) ** 2
            + torch.nn.functional.relu(out - upper_bound) ** 2
        )
        return loss.mean()