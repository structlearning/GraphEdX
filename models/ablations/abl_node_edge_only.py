import torch
from utils import model_utils
import GMN.graphembeddingnetwork as gmngen
from loguru import logger

class ABL_EDGE(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super(ABL_EDGE, self).__init__()
        self.conf = conf
        self.use_max = conf.model.use_max
        self.use_second_sinkhorn = conf.model.use_second_sinkhorn
        self.use_second_sinkhorn_log = conf.model.use_second_sinkhorn_log
        self.use_h_hp_node = conf.model.use_h_hp_node
        self.use_m_ms_edge = conf.model.use_m_ms_edge

        logger.info(f"Running with use_max: {self.use_max}, use_second_sinkhorn: {self.use_second_sinkhorn}, use_second_sinkhorn_log: {self.use_second_sinkhorn_log}, use_h_hp_node: {self.use_h_hp_node}, use_m_ms_edge: {self.use_m_ms_edge}")
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.gmn_config = gmn_config
        self.max_edge_set_size = conf.model.edge_scale * conf.dataset.max_edge_set_size
        self.max_node_set_size = conf.dataset.max_node_set_size
        self.device = conf.training.device

        if conf.data_mode == "unequal":
            self.node_ins_cost = conf.dataset.node_ins_cost
            self.node_del_cost = conf.dataset.node_del_cost
            self.node_rel_cost = conf.dataset.node_rel_cost
            self.edge_ins_cost = conf.dataset.edge_ins_cost
            self.edge_del_cost = conf.dataset.edge_del_cost
            self.edge_rel_cost = conf.dataset.edge_rel_cost
        else:
            self.node_ins_cost = self.node_del_cost = self.node_rel_cost = self.edge_ins_cost = self.edge_del_cost = self.edge_rel_cost = 1
            logger.info(f"Costs: {self.node_ins_cost}, {self.node_del_cost}, {self.node_rel_cost}, {self.edge_ins_cost}, {self.edge_del_cost}, {self.edge_rel_cost}")


        print("EDGE ONLY")

        self.encoder = gmngen.GraphEncoder(**self.gmn_config["encoder"])
        prop_config = self.gmn_config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.propagation_steps = self.gmn_config["graph_embedding_net"]["n_prop_layers"]

        self.node_set_size_nC2 = (self.max_node_set_size * (self.max_node_set_size - 1)) // 2

        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(prop_config["node_state_dim"], self.max_node_set_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_node_set_size, self.max_node_set_size)
        )

        self.source_destination_list = torch.ones(self.max_node_set_size, self.max_node_set_size, device=self.device).triu(1).nonzero()
        self.source_list = self.source_destination_list[:, 0]
        self.destination_list = self.source_destination_list[:, 1]

        self.edge_source_dest_idx = torch.ones(self.node_set_size_nC2,self.node_set_size_nC2, device=self.device).nonzero()
        self.edge_source_idx = self.edge_source_dest_idx[:,0]
        self.edge_dest_idx = self.edge_source_dest_idx[:,1]

        self.lrl_network  = torch.nn.Sequential(
            torch.nn.Linear(
                # prop_config["edge_hidden_sizes"][-1] + 1, self.max_node_set_size
                prop_config["edge_hidden_sizes"][-1] + 1, prop_config["edge_hidden_sizes"][-1]
            ),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.max_node_set_size, self.max_node_set_size),
            torch.nn.Linear(prop_config["edge_hidden_sizes"][-1], prop_config["edge_hidden_sizes"][-1]),
        )
        
        self.LAMBDA = conf.model.LAMBDA


    def nC2_edge_embeddings(self, H, adj):
        source = H[:, self.source_list,:]
        destination =  H[:, self.destination_list,:]
        edge_emb = adj[:, self.source_list, self.destination_list].unsqueeze(-1)

        #Undirected graphs - hence do both forward and backward concat for each edge 
        forward_batch = torch.cat((source,destination,edge_emb),dim=-1)
        backward_batch = torch.cat((destination,source,edge_emb),dim=-1)
        #use message encoding network from GMN encoding to obtain forward and backward score for each edge
        forward_msg_batch = self.lrl_network(forward_batch)
        backward_msg_batch = self.lrl_network(backward_batch)
        #design choice to add forward and backward scores to get total edge score
        bidirectional_msg_batch = forward_msg_batch+backward_msg_batch
        #note the reshape here to get M matrix
        return bidirectional_msg_batch


    def forward(self, graphs, graph_sizes, query_adj, corpus_adj):
        # query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = graph_sizes[0::2]
        corpus_sizes = graph_sizes[1::2]

        (
            node_features,
            edge_features,
            from_idx,
            to_idx,
            graph_idx,
        ) = model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(
            node_features, edge_features
        )
        for _ in range(self.propagation_steps):
            node_features_enc = self.prop_layer(
                node_features_enc, from_idx, to_idx, edge_features_enc
            )

        stacked_query_node_features, stacked_corpus_node_features = model_utils.split_and_stack(
            node_features_enc, graph_sizes.tolist(), self.max_node_set_size)

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_query_node_features)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_corpus_node_features)


        cost_matrix = torch.cdist(transformed_features_query, transformed_features_corpus, p=1)

        node_sinkhorn_input = -cost_matrix # - masked_cost_matrix
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input,
            device=self.device,
            temperature=self.sinkhorn_temp,
            noise_factor= self.conf.training.sinkhorn_noise
        )

        
        nC2_query_edge_embeddings = self.nC2_edge_embeddings(stacked_query_node_features, query_adj)
        nC2_corpus_edge_embeddings = self.nC2_edge_embeddings(stacked_corpus_node_features, corpus_adj)

        node_perm_lookup_idx = torch.cat((self.source_destination_list[self.edge_source_idx],\
                                          self.source_destination_list[self.edge_dest_idx]), dim=-1)
        straight_score = node_transport_plan[:,node_perm_lookup_idx[:,0], node_perm_lookup_idx[:,2]] *\
                        node_transport_plan[:,node_perm_lookup_idx[:,1], node_perm_lookup_idx[:,3]] 
        cross_score = node_transport_plan[:,node_perm_lookup_idx[:,0], node_perm_lookup_idx[:,3]] *\
                        node_transport_plan[:,node_perm_lookup_idx[:,1], node_perm_lookup_idx[:,2]] 

        if self.use_max:
            edge_sinkhorn_input = torch.maximum(straight_score,cross_score).reshape(-1, self.node_set_size_nC2, self.node_set_size_nC2)
        else:
            edge_sinkhorn_input = (straight_score+cross_score).reshape(-1, self.node_set_size_nC2, self.node_set_size_nC2)
        
        if self.use_second_sinkhorn:
            
            edge_transport_plan = model_utils.pytorch_sinkhorn_iters(
             log_alpha=edge_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp, noise_factor= self.conf.training.sinkhorn_noise
            )
        elif self.use_second_sinkhorn_log:
            edge_transport_plan = model_utils.pytorch_sinkhorn_iters(
             log_alpha=torch.log(edge_sinkhorn_input + 1e-6), device=self.device, temperature=self.sinkhorn_temp, noise_factor= self.conf.training.sinkhorn_noise
            )
        else:
            edge_transport_plan = edge_sinkhorn_input

        if self.use_m_ms_edge:
            edge_emb_pairwise_diff = nC2_query_edge_embeddings[:, :, None, :] - nC2_corpus_edge_embeddings[:, None, :, :]
            # edge_emb_pairwise_diff : (batch_size, nC2, nC2, emb_dim)
            edge_align_dist = (edge_transport_plan * (self.edge_del_cost * torch.nn.functional.relu(edge_emb_pairwise_diff).sum(dim=-1)\
                                + self.edge_ins_cost * torch.nn.functional.relu(-edge_emb_pairwise_diff).sum(dim=-1))).sum(dim=(-1, -2))
        else:
            edge_align_dist = model_utils.asymm_embed_mat_l1_dist(nC2_query_edge_embeddings, nC2_corpus_edge_embeddings, edge_transport_plan, self.edge_ins_cost, self.edge_del_cost)

        return edge_align_dist
        
        # return edge_align_dist + self.LAMBDA * node_align_dist


    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)

    

class ABL_NODE(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super(ABL_NODE, self).__init__()
        self.conf = conf
        self.use_max = conf.model.use_max
        self.use_second_sinkhorn = conf.model.use_second_sinkhorn
        self.use_second_sinkhorn_log = conf.model.use_second_sinkhorn_log
        self.use_h_hp_node = conf.model.use_h_hp_node
        self.use_m_ms_edge = conf.model.use_m_ms_edge

        logger.info(f"Running with use_max: {self.use_max}, use_second_sinkhorn: {self.use_second_sinkhorn}, use_second_sinkhorn_log: {self.use_second_sinkhorn_log}, use_h_hp_node: {self.use_h_hp_node}, use_m_ms_edge: {self.use_m_ms_edge}")
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.gmn_config = gmn_config
        self.max_edge_set_size = conf.model.edge_scale * conf.dataset.max_edge_set_size
        self.max_node_set_size = conf.dataset.max_node_set_size
        self.device = conf.training.device

        if conf.data_mode == "unequal":
            self.node_ins_cost = conf.dataset.node_ins_cost
            self.node_del_cost = conf.dataset.node_del_cost
            self.node_rel_cost = conf.dataset.node_rel_cost
            self.edge_ins_cost = conf.dataset.edge_ins_cost
            self.edge_del_cost = conf.dataset.edge_del_cost
            self.edge_rel_cost = conf.dataset.edge_rel_cost
        else:
            self.node_ins_cost = self.node_del_cost = self.node_rel_cost = self.edge_ins_cost = self.edge_del_cost = self.edge_rel_cost = 1
            logger.info(f"Costs: {self.node_ins_cost}, {self.node_del_cost}, {self.node_rel_cost}, {self.edge_ins_cost}, {self.edge_del_cost}, {self.edge_rel_cost}")

        print("NODE ONLY")
        self.encoder = gmngen.GraphEncoder(**self.gmn_config["encoder"])
        prop_config = self.gmn_config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.propagation_steps = self.gmn_config["graph_embedding_net"]["n_prop_layers"]

        self.node_set_size_nC2 = (self.max_node_set_size * (self.max_node_set_size - 1)) // 2

        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(prop_config["node_state_dim"], self.max_node_set_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_node_set_size, self.max_node_set_size)
        )

        self.source_destination_list = torch.ones(self.max_node_set_size, self.max_node_set_size, device=self.device).triu(1).nonzero()
        self.source_list = self.source_destination_list[:, 0]
        self.destination_list = self.source_destination_list[:, 1]

        self.edge_source_dest_idx = torch.ones(self.node_set_size_nC2,self.node_set_size_nC2, device=self.device).nonzero()
        self.edge_source_idx = self.edge_source_dest_idx[:,0]
        self.edge_dest_idx = self.edge_source_dest_idx[:,1]

        self.lrl_network  = torch.nn.Sequential(
            torch.nn.Linear(
                # prop_config["edge_hidden_sizes"][-1] + 1, self.max_node_set_size
                prop_config["edge_hidden_sizes"][-1] + 1, prop_config["edge_hidden_sizes"][-1]
            ),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.max_node_set_size, self.max_node_set_size),
            torch.nn.Linear(prop_config["edge_hidden_sizes"][-1], prop_config["edge_hidden_sizes"][-1]),
        )
        
        self.LAMBDA = conf.model.LAMBDA


    def nC2_edge_embeddings(self, H, adj):
        source = H[:, self.source_list,:]
        destination =  H[:, self.destination_list,:]
        edge_emb = adj[:, self.source_list, self.destination_list].unsqueeze(-1)

        #Undirected graphs - hence do both forward and backward concat for each edge 
        forward_batch = torch.cat((source,destination,edge_emb),dim=-1)
        backward_batch = torch.cat((destination,source,edge_emb),dim=-1)
        #use message encoding network from GMN encoding to obtain forward and backward score for each edge
        forward_msg_batch = self.lrl_network(forward_batch)
        backward_msg_batch = self.lrl_network(backward_batch)
        #design choice to add forward and backward scores to get total edge score
        bidirectional_msg_batch = forward_msg_batch+backward_msg_batch
        #note the reshape here to get M matrix
        return bidirectional_msg_batch


    def forward(self, graphs, graph_sizes, query_adj, corpus_adj, diagnostic_mode=False):
        # query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = graph_sizes[0::2]
        corpus_sizes = graph_sizes[1::2]

        (
            node_features,
            edge_features,
            from_idx,
            to_idx,
            graph_idx,
        ) = model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(
            node_features, edge_features
        )
        for _ in range(self.propagation_steps):
            node_features_enc = self.prop_layer(
                node_features_enc, from_idx, to_idx, edge_features_enc
            )

        stacked_query_node_features, stacked_corpus_node_features = model_utils.split_and_stack(
            node_features_enc, graph_sizes.tolist(), self.max_node_set_size)

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_query_node_features)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_corpus_node_features)


        cost_matrix = torch.cdist(transformed_features_query, transformed_features_corpus, p=1)

        node_sinkhorn_input = -cost_matrix # - masked_cost_matrix
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input,
            device=self.device,
            temperature=self.sinkhorn_temp,
            noise_factor= self.conf.training.sinkhorn_noise
        )

        

        if self.use_h_hp_node:
            node_emb_pairwise_diff = stacked_query_node_features[:, :, None, :] - stacked_corpus_node_features[:, None, :, :]
            node_align_dist = (node_transport_plan * (self.node_del_cost * torch.nn.functional.relu(node_emb_pairwise_diff).sum(dim=-1)\
                                + self.node_ins_cost * torch.nn.functional.relu(-node_emb_pairwise_diff).sum(dim=-1))).sum(dim=(-1, -2))    
        else:
            node_align_dist = model_utils.asymm_embed_mat_l1_dist(stacked_query_node_features, stacked_corpus_node_features, node_transport_plan, self.node_ins_cost, self.node_del_cost)

        if diagnostic_mode: 
            return node_align_dist, node_transport_plan, stacked_query_node_features, stacked_corpus_node_features
        
        return node_align_dist

        # return edge_align_dist + self.LAMBDA * node_align_dist


    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)
