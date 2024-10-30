import torch
import GMN.graphembeddingnetwork as gmngen
import GMN.graphmatchingnetwork as gmngmn
import GMN.utils as gmnutils
import utils.model_utils as model_utils
import GMN.fast_graphmatchingnetwork as fast_gmngmn

class Match(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        """ """
        super(Match, self).__init__()
        self.output_mode = conf.model.output_mode
        self.max_set_size = conf.dataset.max_set_size
        self.config = gmn_config
        self.build_layers()
        self.diagnostic_mode = False
        self.device = conf.training.device

        self.node_ins_cost = conf.dataset.node_ins_cost
        self.node_del_cost = conf.dataset.node_del_cost
        self.node_rel_cost = conf.dataset.node_rel_cost
        self.edge_ins_cost = conf.dataset.edge_ins_cost
        self.edge_del_cost = conf.dataset.edge_del_cost
        self.edge_rel_cost = conf.dataset.edge_rel_cost
        self.norm_mode = conf.model.norm_mode


    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        self.similarity_func = self.config["graph_matching_net"]["similarity"]
        prop_config = self.config["graph_matching_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        prop_config.pop("similarity", None)
        self.prop_layer = fast_gmngmn.GraphPropMatchingLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])
    
        
    def get_graph(self, graphs):
        return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx

    def forward(self, batch_data, batch_data_sizes, batch_adj=None):
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(
            batch_data
        )

        node_features_enc, edge_features_enc = self.encoder(
            node_features, edge_features
        )
        for i in range(self.config["graph_matching_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(
                node_features_enc,
                from_idx,
                to_idx,
                graph_idx,
                len(batch_data_sizes),
                self.similarity_func,
                edge_features_enc,
                batch_data_sizes_flat = batch_data_sizes, # [Q1,T1, Q2, T2.....]
                max_node_size = self.max_set_size
            )

        graph_vectors = self.aggregator(
            node_features_enc, graph_idx, len(batch_data_sizes)
        )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        return torch.norm(x - y, dim=-1, p=int(self.output_mode[-1]))


    def compute_loss(self, lower_bound, upper_bound, out):
        loss = (
            torch.nn.functional.relu(lower_bound - out) ** 2
            + torch.nn.functional.relu(out - upper_bound) ** 2
        )
        return loss.mean()
    

class Embed(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        """ """
        super(Embed, self).__init__()
        self.output_mode = conf.model.output_mode
        self.config = gmn_config
        self.build_layers()
        self.diagnostic_mode = False
        self.device = conf.training.device
        self.node_ins_cost = conf.dataset.node_ins_cost
        self.node_del_cost = conf.dataset.node_del_cost
        self.node_rel_cost = conf.dataset.node_rel_cost
        self.edge_ins_cost = conf.dataset.edge_ins_cost
        self.edge_del_cost = conf.dataset.edge_del_cost
        self.edge_rel_cost = conf.dataset.edge_rel_cost
        self.norm_mode = conf.model.norm_mode

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])

        
    def get_graph(self, graphs):
        return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx

    def forward(self, batch_data, batch_data_sizes, batch_adj=None):
        """ """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(
            batch_data
        )

        node_features_enc, edge_features_enc = self.encoder(
            node_features, edge_features
        )
        for i in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(
                node_features_enc, from_idx, to_idx, edge_features_enc
            )

        graph_vectors = self.aggregator(
            node_features_enc, graph_idx, len(batch_data_sizes)
        )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)


        return torch.norm(x - y, dim=-1, p=int(self.output_mode[-1]))

    def compute_loss(self, lower_bound, upper_bound, out):
        loss = (
            torch.nn.functional.relu(lower_bound - out) ** 2
            + torch.nn.functional.relu(out - upper_bound) ** 2
        )
        return loss.mean()

