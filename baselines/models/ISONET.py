import torch
from utils import model_utils
import GMN.graphembeddingnetwork as gmngen
import torch.nn.functional as F

class ISONET(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super(ISONET, self).__init__()
        self.conf = conf
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.gmn_config = gmn_config
        self.max_edge_set_size = conf.model.edge_scale * conf.dataset.max_edge_set_size
        self.device = conf.training.device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=self.max_edge_set_size,
            lateral_dim=self.max_edge_set_size,
            device=self.device,
        )

        self.encoder = gmngen.GraphEncoder(**self.gmn_config["encoder"])
        prop_config = self.gmn_config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.propagation_steps = self.gmn_config["graph_embedding_net"]["n_prop_layers"]

        self.edge_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(
                prop_config["edge_hidden_sizes"][-1], self.max_edge_set_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_edge_set_size, self.max_edge_set_size),
        )

    def forward(self, graphs, graph_sizes, query_adj, target_adj):
        # query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = graph_sizes[0::2]
        corpus_sizes = graph_sizes[1::2]
        # query_sizes = torch.tensor(query_sizes, device=self.device)
        # corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

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

        # Computation of edge embeddings
        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx,
        )

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, len(graph_sizes)
        )
        (
            stacked_edge_features_query,
            stacked_edge_features_corpus,
        ) = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        # Computation of edge transport plan
        transformed_features_query = self.edge_sinkhorn_feature_layers(
            stacked_edge_features_query
        )
        transformed_features_corpus = self.edge_sinkhorn_feature_layers(
            stacked_edge_features_corpus
        )

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features

        num_edges_query = paired_edge_counts[0::2]
        masked_features_query = mask_graphs(transformed_features_query, num_edges_query)
        num_edges_corpus = paired_edge_counts[1::2]
        masked_features_corpus = mask_graphs(
            transformed_features_corpus, num_edges_corpus
        )

        edge_sinkhorn_input = torch.matmul(
            masked_features_query, masked_features_corpus.permute(0, 2, 1)
        )
        edge_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=edge_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp, noise_factor= self.conf.training.sinkhorn_noise
        )


        return F.relu(
                stacked_edge_features_query - edge_transport_plan @ stacked_edge_features_corpus,
            ).sum(dim=(-1, -2))
    
    

    def compute_loss(self, lower_bound, upper_bound, out):
        return model_utils.compute_loss(lower_bound, upper_bound, out)