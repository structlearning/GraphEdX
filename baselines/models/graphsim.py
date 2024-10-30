import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import  pad_sequence
from torch_geometric.data import Batch
from lapjv import lapjv
from utils import model_utils


def cudavar(conf, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    #return x.cuda() if av.has_cuda and av.want_cuda else x
    return x.to(conf.training.device)

class CNNLayerV1(torch.nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.kernel_size - 1)//2
        if self.kernel_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=self.kernel_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = []
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])))
        return result
    
class MaxPoolLayerV1(torch.nn.Module):
    def __init__(self, stride, pool_size, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.pool_size = pool_size
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.pool_size - 1)//2
        if self.pool_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = []
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])))
        return result    


class GraphSim(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super(GraphSim, self).__init__()
        self.conf = conf
        self.config = gmn_config
        self.input_dim = conf.dataset.one_hot_dim
        self.build_layers()

    def build_layers(self):

        self.gcn_layers = torch.nn.ModuleList([])
        self.conv_layers = torch.nn.ModuleList([])
        self.pool_layers = torch.nn.ModuleList([])
        self.linear_layers = torch.nn.ModuleList([])
        self.num_conv_layers = len(self.config['graphsim']['conv_kernel_size'])
        self.num_linear_layers = len(self.config['graphsim']['linear_size'])
        self.num_gcn_layers = len(self.config['graphsim']['gcn_size'])

        num_ftrs = self.input_dim
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                pyg_nn.GCNConv(num_ftrs, self.config['graphsim']['gcn_size'][i]))
            num_ftrs = self.config['graphsim']['gcn_size'][i]

        in_channels = 1
        for i in range(self.num_conv_layers):
            self.conv_layers.append(CNNLayerV1(kernel_size=self.config['graphsim']['conv_kernel_size'][i],
                stride=1, in_channels=in_channels, out_channels=self.config['graphsim']['conv_out_channels'][i],
                num_similarity_matrices=self.num_gcn_layers))
            self.pool_layers.append(MaxPoolLayerV1(pool_size=self.config['graphsim']['conv_pool_size'][i],
                stride=self.config['graphsim']['conv_pool_size'][i], num_similarity_matrices=self.num_gcn_layers))
            in_channels = self.config['graphsim']['conv_out_channels'][i]

        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(self.config['graphsim']['linear_size'][i],
                self.config['graphsim']['linear_size'][i+1]))

        self.scoring_layer = torch.nn.Linear(self.config['graphsim']['linear_size'][-1], 1)

    def GCN_pass(self, data):
        features, edge_index = data.x, data.edge_index
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)


        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def Conv_pass(self, similarity_matrices_list):
        features = [_.unsqueeze(1) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features)

            features = [torch.nn.functional.dropout(_,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)  for _ in features]
        return features

    def linear_pass(self, features):
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,p=self.config['graphsim']['dropout'],
                                               training=self.training)
        return features
    
    def pad_matrix(self, matrix):
        M = self.conf.dataset.max_node_set_size
        if matrix.shape[-1] < M or matrix.shape[-2] < M:
            pad_x = max(0, M - matrix.shape[-1])
            pad_y = max(0, M - matrix.shape[-2])
            padded_tensor = torch.nn.functional.pad(matrix, (0, pad_x, 0, pad_y))
            return padded_tensor
        return matrix
    
    def forward(self, batch_data,batch_data_sizes):
        # q_graphs,c_graphs = zip(*batch_data)
        # a,b = zip(*batch_data_sizes)

        q_graphs = batch_data[0::2]
        c_graphs = batch_data[1::2]
        
        a = batch_data_sizes[0::2]
        b = batch_data_sizes[1::2]
        
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)

        query_abstract_features_list = self.GCN_pass(query_batch)
        query_abstract_features_list = [pad_sequence(torch.split(query_abstract_features_list[i], list(a), dim=0), batch_first=True) \
                                        for i in range(self.num_gcn_layers)]


        corpus_abstract_features_list = self.GCN_pass(corpus_batch)
        corpus_abstract_features_list = [pad_sequence(torch.split(corpus_abstract_features_list[i], list(b), dim=0), batch_first=True) \
                                          for i in range(self.num_gcn_layers)]

        similarity_matrices_list = [torch.matmul(query_abstract_features_list[i],\
                                    corpus_abstract_features_list[i].permute(0,2,1))
                                    for i in range(self.num_gcn_layers)]
        
        similarity_matrices_list = [self.pad_matrix(_) for _ in similarity_matrices_list]
        
        # print(f'similarity_matrices_list shape: {similarity_matrices_list[0].shape} && length = {len(similarity_matrices_list)}')

        features = torch.cat(self.Conv_pass(similarity_matrices_list), dim=1).view(-1,
                              self.config['graphsim']['linear_size'][0])
        
        # print(f'features shape: {features.shape}')
        features = self.linear_pass(features)
        # print(f'features shape: {features.shape}')

        score_logits = self.scoring_layer(features)
        # print(f'score_logits shape: {score_logits.shape}')
        
        if self.conf.model.is_sig:
          score = torch.sigmoid(score_logits)
          return score.view(-1)
        else:
          return score_logits.view(-1)
        
    
    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)

def dense_wasserstein_distance_v3(cost_matrix):
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(cost_matrix)

    return np.eye(cost_matrix.shape[0])[col_ind_lapjv]


class GOTSim(torch.nn.Module):
    def __init__(self, conf,gmn_config):
        """
        """
        super(GOTSim, self).__init__()
        self.conf = conf
        self.config = gmn_config
        self.input_dim = conf.dataset.one_hot_dim

        #Conv layers
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.conf.model.filters_1)
        self.conv2 = pyg_nn.GCNConv(self.conf.model.filters_1, self.conf.model.filters_2)
        self.conv3 = pyg_nn.GCNConv(self.conf.model.filters_2, self.conf.model.filters_3)
        self.num_gcn_layers = 3

        # self.n1 = self.av.MAX_QUERY_SUBGRAPH_SIZE
        # self.n2 = self.av.MAX_CORPUS_SUBGRAPH_SIZE
        self.n1 = self.conf.dataset.max_node_set_size
        self.n2 = self.conf.dataset.max_node_set_size

        self.insertion_constant_matrix = cudavar(self.conf,99999 * (torch.ones(self.n1, self.n1)
                                                - torch.diag(torch.ones(self.n1))))
        self.deletion_constant_matrix = cudavar(self.conf,99999 * (torch.ones(self.n2, self.n2)
                                                - torch.diag(torch.ones(self.n2))))


        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)

        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.conf.model.filters_1)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.conf.model.filters_2)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.conf.model.filters_3)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.conf.model.filters_1)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.conf.model.filters_2)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.conf.model.filters_3)))

    def GNN (self, data):
        """
        """
        gcn_feature_list = []
        features = self.conv1(data.x,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.conf.training.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.conf.training.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        gcn_feature_list.append(features)
        return gcn_feature_list


    def forward(self, batch_data,batch_data_sizes):
        q_graphs = batch_data[0::2]
        c_graphs = batch_data[1::2]
        
        a = batch_data_sizes[0::2]
        b = batch_data_sizes[1::2]
        batch_sz = len(q_graphs)
        # q_graphs,c_graphs = zip(*batch_data)
        # a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.conf,torch.tensor(a))
        cgraph_sizes = cudavar(self.conf,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)
        query_gcn_feature_list = self.GNN(query_batch)
        corpus_gcn_feature_list = self.GNN(corpus_batch)

        pad_main_similarity_matrices_list=[]
        pad_deletion_similarity_matrices_list = []
        pad_insertion_similarity_matrices_list = []
        pad_dummy_similarity_matrices_list = []
        for i in range(self.num_gcn_layers):

            q = pad_sequence(torch.split(query_gcn_feature_list[i], list(a), dim=0), batch_first=True)
            c = pad_sequence(torch.split(corpus_gcn_feature_list[i],list(b), dim=0), batch_first=True)
            q = F.pad(q,pad=(0,0,0,self.n1-q.shape[1],0,0))
            c = F.pad(c,pad=(0,0,0,self.n2-c.shape[1],0,0))
            #NOTE THE -VE HERE. BECAUSE THIS IS ACTUALLY COST MAT
            pad_main_similarity_matrices_list.append(-torch.matmul(q,c.permute(0,2,1)))

            pad_deletion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(q, self.deletion_params[i]))+\
                                                    self.insertion_constant_matrix)

            pad_insertion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(c, self.insertion_params[i]))+\
                                                     self.deletion_constant_matrix)

            pad_dummy_similarity_matrices_list.append(cudavar(self.conf,torch.zeros(batch_sz,self.n2, self.n1, \
                                                      dtype=q.dtype)))


        sim_mat_all = []
        for j in range(batch_sz):
            for i in range(self.num_gcn_layers):
                a = pad_main_similarity_matrices_list[i][j]
                b =pad_deletion_similarity_matrices_list[i][j]
                c = pad_insertion_similarity_matrices_list[i][j]
                d = pad_dummy_similarity_matrices_list[i][j]
                s1 = qgraph_sizes[j]
                s2 = cgraph_sizes[j]
                sim_mat_all.append(torch.cat((torch.cat((a[:s1,:s2], b[:s1,:s1]), dim=1),\
                               torch.cat((c[:s2,:s2], d[:s2,:s1]), dim=1)), dim=0))


        sim_mat_all_cpu = [x.detach().cpu().numpy() for x in sim_mat_all]
        plans = [dense_wasserstein_distance_v3(x) for x in sim_mat_all_cpu ]
        mcost = [torch.sum(torch.mul(x,cudavar(self.conf,torch.Tensor(y)))) for (x,y) in zip(sim_mat_all,plans)]
        sz_sum = qgraph_sizes.repeat_interleave(3)+cgraph_sizes.repeat_interleave(3)
        mcost_norm = 2*torch.div(torch.stack(mcost),sz_sum)
        scores_new =  self.ot_scoring_layer(mcost_norm.view(-1,3)).squeeze()
        #return scores_new.view(-1)

        if self.conf.model.is_sig:
            return torch.sigmoid(scores_new).view(-1)
        else:
            return scores_new.view(-1)

    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)