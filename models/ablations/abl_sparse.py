import torch
from utils import model_utils
import GMN.graphembeddingnetwork as gmngen


class ABL_SPARSE(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super(ABL_SPARSE, self).__init__()
        self.conf = conf
        self.use_max = conf.model.use_max
        self.use_second_sinkhorn = conf.model.use_second_sinkhorn
        self.use_second_sinkhorn_log = conf.model.use_second_sinkhorn_log
        self.use_h_hp_node = conf.model.use_h_hp_node
        self.use_m_ms_edge = conf.model.use_m_ms_edge
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.gmn_config = gmn_config
        self.max_edge_set_size =  conf.dataset.max_edge_set_size
        self.max_node_set_size = conf.dataset.max_node_set_size
        self.device = conf.training.device

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

        if conf.data_mode == "unequal":
            self.node_ins_cost = conf.dataset.node_ins_cost
            self.node_del_cost = conf.dataset.node_del_cost
            self.node_rel_cost = conf.dataset.node_rel_cost
            self.edge_ins_cost = conf.dataset.edge_ins_cost
            self.edge_del_cost = conf.dataset.edge_del_cost
            self.edge_rel_cost = conf.dataset.edge_rel_cost
        else:
            self.node_ins_cost = self.node_del_cost = self.node_rel_cost = self.edge_ins_cost = self.edge_del_cost = self.edge_rel_cost = 1
            print(f"Costs: {self.node_ins_cost}, {self.node_del_cost}, {self.node_rel_cost}, {self.edge_ins_cost}, {self.edge_del_cost}, {self.edge_rel_cost}")

        self.lrl_network  = torch.nn.Sequential(
            torch.nn.Linear(
                prop_config["edge_hidden_sizes"][-1] + 1, prop_config["edge_hidden_sizes"][-1]
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(prop_config["edge_hidden_sizes"][-1], prop_config["edge_hidden_sizes"][-1]),
        )
        
        self.LAMBDA = conf.model.LAMBDA

    def E_edge_embeddings(self, node_features_enc, from_idx, to_idx, edge_features_enc):
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
        forward_edge_msg = self.lrl_network(forward_edge_input)
        backward_edge_msg = self.lrl_network(backward_edge_input)
        bidirectional_msg_enc = forward_edge_msg + backward_edge_msg
        return bidirectional_msg_enc


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

        
        # nC2_query_edge_embeddings = self.nC2_edge_embeddings(stacked_query_node_features, query_adj)
        # nC2_corpus_edge_embeddings = self.nC2_edge_embeddings(stacked_corpus_node_features, corpus_adj)
        edge_msg_enc = self.E_edge_embeddings(node_features_enc, from_idx, to_idx, edge_features_enc)
        edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, len(graph_sizes)
        )
        stacked_query_edge_features, stacked_corpus_edge_features = model_utils.split_and_stack(
            edge_msg_enc, edge_counts, self.max_edge_set_size)

        straight_score, cross_score = model_utils.kronecker_product_on_nodes(node_transport_plan, from_idx, to_idx, edge_counts, graph_sizes, self.max_edge_set_size)


        if self.use_max:
            # edge_sinkhorn_input = torch.maximum(straight_score,cross_score).reshape(-1, self.node_set_size_nC2, self.node_set_size_nC2)
            edge_sinkhorn_input = torch.maximum(straight_score,cross_score).reshape(-1, self.max_edge_set_size, self.max_edge_set_size)
        else:
            # edge_sinkhorn_input = (straight_score+cross_score).reshape(-1, self.node_set_size_nC2, self.node_set_size_nC2)
            edge_sinkhorn_input = (straight_score+cross_score).reshape(-1, self.max_edge_set_size, self.max_edge_set_size)
        
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
            edge_emb_pairwise_diff = stacked_query_edge_features[:, :, None, :] - stacked_corpus_edge_features[:, None, :, :]
            # edge_emb_pairwise_diff : (batch_size, nC2, nC2, emb_dim)
            edge_align_dist = (edge_transport_plan * (self.edge_del_cost * torch.nn.functional.relu(edge_emb_pairwise_diff).sum(dim=-1)\
                                + self.edge_ins_cost * torch.nn.functional.relu(-edge_emb_pairwise_diff).sum(dim=-1))).sum(dim=(-1, -2))
        else:
            edge_align_dist = model_utils.asymm_embed_mat_l1_dist(stacked_query_edge_features, stacked_corpus_edge_features, edge_transport_plan, self.edge_ins_cost, self.edge_del_cost)



        if self.use_h_hp_node:
            node_emb_pairwise_diff = stacked_query_node_features[:, :, None, :] - stacked_corpus_node_features[:, None, :, :]
            node_align_dist = (node_transport_plan * (self.node_del_cost * torch.nn.functional.relu(node_emb_pairwise_diff).sum(dim=-1)\
                                + self.node_ins_cost * torch.nn.functional.relu(-node_emb_pairwise_diff).sum(dim=-1))).sum(dim=(-1, -2))    
        else:
            node_align_dist = model_utils.asymm_embed_mat_l1_dist(stacked_query_node_features, stacked_corpus_node_features, node_transport_plan, self.node_ins_cost, self.node_del_cost)


        if diagnostic_mode:
            return edge_align_dist, node_align_dist, node_transport_plan, edge_transport_plan, stacked_query_node_features, stacked_corpus_node_features, stacked_query_edge_features, stacked_corpus_edge_features

        return edge_align_dist + self.LAMBDA * node_align_dist


    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)
