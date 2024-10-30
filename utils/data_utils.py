import collections
import itertools
import torch
import time
import numpy as np
import torch.nn.functional as F
import networkx as nx
import torch_geometric as pyg

class GraphEditDatasetCombination:
    def __init__(self, conf, mode="train", logger=None):
        self.batch_size = conf.training.batch_size
        self.device = conf.training.device
        self.data_type = conf.dataset.data_type
        self.return_adj = conf.dataset.return_adj
        self.mode = mode
        start = time.perf_counter()

        self.graphs = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_graphs.pt")
        self.features = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_features.pt")
        self.features = [x.to(self.device) for x in self.features]
        self.max_node_set_size = conf.dataset.max_node_set_size

        if self.return_adj:
            self.build_adjacency_info()

        if conf.data_mode == "unequal":
            self.lower_bounds, self.upper_bounds, _ = list(zip(*torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_{conf.dataset.node_ins_cost}{conf.dataset.node_del_cost}{conf.dataset.node_rel_cost}{conf.dataset.edge_ins_cost}{conf.dataset.edge_del_cost}{conf.dataset.edge_rel_cost}_result.pt")))
        else:
            self.lower_bounds, self.upper_bounds, _ = list(zip(*torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_result.pt")))
        self.lower_bounds = torch.tensor(list(self.lower_bounds), dtype=torch.float32, device=self.device)
        self.upper_bounds = torch.tensor(list(self.upper_bounds), dtype=torch.float32, device=self.device)
        assert (self.lower_bounds == self.upper_bounds).all(), f"Lower and upper bounds are not equal for {conf.dataset.name}: {self.mode}"

        

        self.query_indices, self.target_indices = list(zip(*list(itertools.combinations_with_replacement(range(len(self.graphs)), 2))))
        self.query_indices = np.array(self.query_indices)
        self.target_indices = np.array(self.target_indices)
        if logger is not None:
            logger.info(
                f"Finished loading data in {round(time.perf_counter() - start, 3)} seconds"
            )


        if conf.dataset.data_type == 'gmn':
            self.GraphData = collections.namedtuple(
                "GraphData",
                [
                    "from_idx",
                    "to_idx",
                    "node_features",
                    "edge_features",
                    "n_graphs",
                    "graph_idx",
                ],
            )

        else:
            if logger is not None:
                logger.info(f'Creating data objects on {self.device}')
            self.create_pyg_data()
        self.sizes = torch.tensor(
            list(map(lambda g: g.number_of_nodes(), self.graphs)), device=self.device)
        
        self.graph_idxs = np.arange(len(self.query_indices))
        if logger is not None:
            logger.info(f"Number of data points: {len(self.graph_idxs)}")

    def __create_pyg_data_object(self, graph, feature):
        l = list(graph.edges)
        edge_list = [[x,y] for (x,y) in l ]+ [[y,x] for (x,y) in l]
        x1 = feature
        edge_list = (torch.tensor(edge_list, dtype=torch.long).T).to(self.device)
        return pyg.data.Data(edge_index=edge_list, x=x1)
        
    
    def create_pyg_data(self):
        self.data = []
        for g, f in zip(self.graphs, self.features):
            self.data.append(self.__create_pyg_data_object(g, f))

    @staticmethod
    def split_given_size(a, size):
        return np.split(a, np.arange(size, len(a), size))

    def create_batches(self, shuffle=False):
        self.batches = []
        if shuffle:
            np.random.shuffle(self.graph_idxs)

        self.batches = list(self.split_given_size(self.graph_idxs, self.batch_size))
        self.num_batches = len(self.batches)
        return self.num_batches

    def build_adjacency_info(self):
        self.graphs_adj_list = []
        for graph in self.graphs:
            unpadded_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float, device=self.device)
            assert unpadded_adj.shape[0] == unpadded_adj.shape[1]
            num_nodes = len(unpadded_adj)
            padded_adj = F.pad(unpadded_adj, pad = (0, self.max_node_set_size - num_nodes, 0, self.max_node_set_size - num_nodes))
            self.graphs_adj_list.append(padded_adj)
    

    def _pack_batch(self, graphs, features):
        from_idx = []
        to_idx = []
        total_edges = 0
        total_nodes = 0
        graph_idx = []

        for i, g in enumerate(graphs):
            num_nodes, num_edges = g.number_of_nodes(), g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32) 
            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * i)
            total_nodes += num_nodes
            total_edges += num_edges

        return self.GraphData(
            from_idx = torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx = torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx = torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            n_graphs = len(graphs),
            node_features = torch.cat(features),
            edge_features = torch.ones(total_edges, 1, dtype=torch.float, device=self.device)
        )

    def fetch_batched_data_by_id(self, i):
        batch = self.batches[i]
        query_idxs = self.query_indices[batch]
        target_idxs = self.target_indices[batch]
        # alternate_idxs = list(itertools.chain.from_iterable(zip(query_idxs,target_idxs)))
        
        if self.data_type == 'gmn':
            query_graphs = [self.graphs[k] for k in query_idxs]
            query_features = [self.features[k] for k in query_idxs]
            if self.return_adj:
                query_adjs = torch.stack([self.graphs_adj_list[k] for k in query_idxs])

            target_features = [self.features[k] for k in target_idxs]
            target_graphs = [self.graphs[k] for k in target_idxs]
            if self.return_adj:
                target_adjs = torch.stack([self.graphs_adj_list[k] for k in target_idxs])

            all_data = self._pack_batch(
                list(itertools.chain.from_iterable(zip(query_graphs, target_graphs))),
                list(itertools.chain.from_iterable(zip(query_features, target_features))),
            )
        else:
            query_data = [self.data[k] for k in query_idxs]
            target_data = [self.data[k] for k in target_idxs]
            all_data =  list(itertools.chain.from_iterable(zip(query_data, target_data)))
        query_sizes = self.sizes[query_idxs]
        target_sizes = self.sizes[target_idxs]
        upper_bounds = self.upper_bounds[batch]
        lower_bounds = self.lower_bounds[batch]
        # Alternate values
        if self.return_adj: 
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
                query_adjs,
                target_adjs,
            )
        else:
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
            )
        




class GraphEditDatasetCombinationCost:
    def __init__(self, conf, mode="train", logger=None):
        self.batch_size = conf.training.batch_size
        self.device = conf.training.device
        self.data_type = conf.dataset.data_type
        self.return_adj = conf.dataset.return_adj
        self.mode = mode
        start = time.perf_counter()
        all_costs = torch.tensor([conf.dataset.node_ins_cost, conf.dataset.node_del_cost, conf.dataset.edge_ins_cost, conf.dataset.edge_del_cost], device=self.device, dtype=torch.float32)

        self.graphs = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_graphs.pt")
        features = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_features.pt")
        # self.features = [x.to(self.device) for x in self.features]
        self.features = [all_costs.unsqueeze(0).repeat(features[i].shape[0], 1) for i in range(len(features))]
        self.max_node_set_size = conf.dataset.max_node_set_size

        if self.return_adj:
            self.build_adjacency_info()

        self.lower_bounds, self.upper_bounds, _ = list(zip(*torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_{conf.dataset.node_ins_cost}{conf.dataset.node_del_cost}{conf.dataset.node_rel_cost}{conf.dataset.edge_ins_cost}{conf.dataset.edge_del_cost}{conf.dataset.edge_rel_cost}_result.pt")))
        self.lower_bounds = torch.tensor(list(self.lower_bounds), dtype=torch.float32, device=self.device)
        self.upper_bounds = torch.tensor(list(self.upper_bounds), dtype=torch.float32, device=self.device)
        assert (self.lower_bounds == self.upper_bounds).all(), f"Lower and upper bounds are not equal for {conf.dataset.name}: {self.mode}"

        

        self.query_indices, self.target_indices = list(zip(*list(itertools.combinations_with_replacement(range(len(self.graphs)), 2))))
        self.query_indices = np.array(self.query_indices)
        self.target_indices = np.array(self.target_indices)
        if logger is not None:
            logger.info(
                f"Finished loading data in {round(time.perf_counter() - start, 3)} seconds"
            )


        if conf.dataset.data_type == 'gmn':
            self.GraphData = collections.namedtuple(
                "GraphData",
                [
                    "from_idx",
                    "to_idx",
                    "node_features",
                    "edge_features",
                    "n_graphs",
                    "graph_idx",
                ],
            )

        else:
            if logger is not None:
                logger.info(f'Creating data objects on {self.device}')
            self.create_pyg_data()
        self.sizes = torch.tensor(
            list(map(lambda g: g.number_of_nodes(), self.graphs)), device=self.device)
        
        self.graph_idxs = np.arange(len(self.query_indices))
        if logger is not None:
            logger.info(f"Number of data points: {len(self.graph_idxs)}")

    def __create_pyg_data_object(self, graph, feature):
        l = list(graph.edges)
        edge_list = [[x,y] for (x,y) in l ]+ [[y,x] for (x,y) in l]
        x1 = feature
        edge_list = (torch.tensor(edge_list, dtype=torch.long).T).to(self.device)
        return pyg.data.Data(edge_index=edge_list, x=x1)
        
    
    def create_pyg_data(self):
        self.data = []
        for g, f in zip(self.graphs, self.features):
            self.data.append(self.__create_pyg_data_object(g, f))

    @staticmethod
    def split_given_size(a, size):
        return np.split(a, np.arange(size, len(a), size))

    def create_batches(self, shuffle=False):
        self.batches = []
        if shuffle:
            np.random.shuffle(self.graph_idxs)

        self.batches = list(self.split_given_size(self.graph_idxs, self.batch_size))
        self.num_batches = len(self.batches)
        return self.num_batches

    def build_adjacency_info(self):
        self.graphs_adj_list = []
        for graph in self.graphs:
            unpadded_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float, device=self.device)
            assert unpadded_adj.shape[0] == unpadded_adj.shape[1]
            num_nodes = len(unpadded_adj)
            padded_adj = F.pad(unpadded_adj, pad = (0, self.max_node_set_size - num_nodes, 0, self.max_node_set_size - num_nodes))
            self.graphs_adj_list.append(padded_adj)
    

    def _pack_batch(self, graphs, features):
        from_idx = []
        to_idx = []
        total_edges = 0
        total_nodes = 0
        graph_idx = []

        for i, g in enumerate(graphs):
            num_nodes, num_edges = g.number_of_nodes(), g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32) 
            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * i)
            total_nodes += num_nodes
            total_edges += num_edges

        return self.GraphData(
            from_idx = torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx = torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx = torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            n_graphs = len(graphs),
            node_features = torch.cat(features),
            edge_features = torch.ones(total_edges, 1, dtype=torch.float, device=self.device)
        )

    def fetch_batched_data_by_id(self, i):
        batch = self.batches[i]
        query_idxs = self.query_indices[batch]
        target_idxs = self.target_indices[batch]
        # alternate_idxs = list(itertools.chain.from_iterable(zip(query_idxs,target_idxs)))
        
        if self.data_type == 'gmn':
            query_graphs = [self.graphs[k] for k in query_idxs]
            query_features = [self.features[k] for k in query_idxs]
            if self.return_adj:
                query_adjs = torch.stack([self.graphs_adj_list[k] for k in query_idxs])

            target_features = [self.features[k] for k in target_idxs]
            target_graphs = [self.graphs[k] for k in target_idxs]
            if self.return_adj:
                target_adjs = torch.stack([self.graphs_adj_list[k] for k in target_idxs])

            all_data = self._pack_batch(
                list(itertools.chain.from_iterable(zip(query_graphs, target_graphs))),
                list(itertools.chain.from_iterable(zip(query_features, target_features))),
            )
        else:
            query_data = [self.data[k] for k in query_idxs]
            target_data = [self.data[k] for k in target_idxs]
            all_data =  list(itertools.chain.from_iterable(zip(query_data, target_data)))
        query_sizes = self.sizes[query_idxs]
        target_sizes = self.sizes[target_idxs]
        upper_bounds = self.upper_bounds[batch]
        lower_bounds = self.lower_bounds[batch]
        # Alternate values
        if self.return_adj: 
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
                query_adjs,
                target_adjs,
            )
        else:
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
            )

class GraphEditDatasetCombinationSeparateLabel:
    def __init__(self, conf, mode="train", logger=None):
        self.batch_size = conf.training.batch_size
        self.device = conf.training.device
        self.data_type = conf.dataset.data_type
        self.return_adj = conf.dataset.return_adj
        self.mode = mode
        self.is_baseline = conf.model.is_baseline
        start = time.perf_counter()

        self.graphs = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_graphs.pt")




        self.one_hot_labels = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_features.pt")
        self.one_hot_labels = [x.to(self.device) for x in self.one_hot_labels]

        if conf.dataset.use_labels_as_features == False:
            self.features = list(map(lambda n: torch.ones(n, 1, device=self.device), list(map(lambda g: g.number_of_nodes(), self.graphs))))
        else:
            self.features = torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_features.pt")
            self.features = [x.to(self.device) for x in self.features]
            

        self.max_node_set_size = conf.dataset.max_node_set_size

        if self.return_adj:
            self.build_adjacency_info()

        self.lower_bounds, self.upper_bounds, _ = list(zip(*torch.load(f"{conf.dataset.path}/{conf.dataset.name}/{self.mode}_result.pt")))
        self.lower_bounds = torch.tensor(list(self.lower_bounds), dtype=torch.float32, device=self.device)
        self.upper_bounds = torch.tensor(list(self.upper_bounds), dtype=torch.float32, device=self.device)
        assert (self.lower_bounds == self.upper_bounds).all(), f"Lower and upper bounds are not equal for {conf.dataset.name}: {self.mode}"

        

        self.query_indices, self.target_indices = list(zip(*list(itertools.combinations_with_replacement(range(len(self.graphs)), 2))))
        self.query_indices = np.array(self.query_indices)
        self.target_indices = np.array(self.target_indices)
        if logger is not None:
            logger.info(
                f"Finished loading data in {round(time.perf_counter() - start, 3)} seconds"
            )


        if conf.dataset.data_type == 'gmn':
            self.GraphData = collections.namedtuple(
                "GraphData",
                [
                    "from_idx",
                    "to_idx",
                    "node_features",
                    "edge_features",
                    "n_graphs",
                    "graph_idx",
                ],
            )

        else:
            if logger is not None:
                logger.info(f'Creating data objects on {self.device}')
            self.create_pyg_data()
        self.sizes = torch.tensor(
            list(map(lambda g: g.number_of_nodes(), self.graphs)), device=self.device)
        
        self.graph_idxs = np.arange(len(self.query_indices))
        if logger is not None:
            logger.info(f"Number of data points: {len(self.graph_idxs)}")

    def __create_pyg_data_object(self, graph, feature):
        l = list(graph.edges)
        edge_list = [[x,y] for (x,y) in l ]+ [[y,x] for (x,y) in l]
        x1 = feature
        edge_list = (torch.tensor(edge_list, dtype=torch.long).T).to(self.device)
        return pyg.data.Data(edge_index=edge_list, x=x1)
        
    
    def create_pyg_data(self):
        self.data = []
        for g, f in zip(self.graphs, self.features):
            self.data.append(self.__create_pyg_data_object(g, f))

    @staticmethod
    def split_given_size(a, size):
        return np.split(a, np.arange(size, len(a), size))

    def create_batches(self, shuffle=False):
        self.batches = []
        if shuffle:
            np.random.shuffle(self.graph_idxs)

        self.batches = list(self.split_given_size(self.graph_idxs, self.batch_size))
        self.num_batches = len(self.batches)
        return self.num_batches

    def build_adjacency_info(self):
        self.graphs_adj_list = []
        for graph in self.graphs:
            unpadded_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float, device=self.device)
            assert unpadded_adj.shape[0] == unpadded_adj.shape[1]
            num_nodes = len(unpadded_adj)
            padded_adj = F.pad(unpadded_adj, pad = (0, self.max_node_set_size - num_nodes, 0, self.max_node_set_size - num_nodes))
            self.graphs_adj_list.append(padded_adj)
    

    def _pack_batch(self, graphs, features):
        from_idx = []
        to_idx = []
        total_edges = 0
        total_nodes = 0
        graph_idx = []

        for i, g in enumerate(graphs):
            num_nodes, num_edges = g.number_of_nodes(), g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32) 
            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * i)
            total_nodes += num_nodes
            total_edges += num_edges

        return self.GraphData(
            from_idx = torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx = torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx = torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            n_graphs = len(graphs),
            node_features = torch.cat(features),
            edge_features = torch.ones(total_edges, 1, dtype=torch.float, device=self.device),
        )

    def fetch_batched_data_by_id(self, i):
        batch = self.batches[i]
        query_idxs = self.query_indices[batch]
        target_idxs = self.target_indices[batch]
        # alternate_idxs = list(itertools.chain.from_iterable(zip(query_idxs,target_idxs)))
        
        if self.data_type == 'gmn':
            query_graphs = [self.graphs[k] for k in query_idxs]
            query_features = [self.features[k] for k in query_idxs]
            if self.return_adj:
                query_adjs = torch.stack([self.graphs_adj_list[k] for k in query_idxs])

            query_labels = [self.one_hot_labels[k] for k in query_idxs]

            target_features = [self.features[k] for k in target_idxs]
            target_graphs = [self.graphs[k] for k in target_idxs]
            if self.return_adj:
                target_adjs = torch.stack([self.graphs_adj_list[k] for k in target_idxs])

            target_labels = [self.one_hot_labels[k] for k in target_idxs]
            all_data = self._pack_batch(
                list(itertools.chain.from_iterable(zip(query_graphs, target_graphs))),
                list(itertools.chain.from_iterable(zip(query_features, target_features))),
            )
        else:
            query_data = [self.data[k] for k in query_idxs]
            target_data = [self.data[k] for k in target_idxs]
            all_data =  list(itertools.chain.from_iterable(zip(query_data, target_data)))
        query_sizes = self.sizes[query_idxs]
        target_sizes = self.sizes[target_idxs]
        upper_bounds = self.upper_bounds[batch]
        lower_bounds = self.lower_bounds[batch]
        # Alternate values
        if self.is_baseline:
            if self.return_adj: 
                return (
                    all_data,
                    torch.dstack((query_sizes, target_sizes)).flatten(),
                    upper_bounds,
                    lower_bounds,
                    query_adjs,
                    target_adjs,
                )
            else:
                return (
                    all_data,
                    torch.dstack((query_sizes, target_sizes)).flatten(),
                    upper_bounds,
                    lower_bounds,
                )
        
        if self.return_adj: 
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
                torch.cat(list(itertools.chain.from_iterable(zip(query_labels, target_labels)))),
                query_adjs,
                target_adjs,
            )
        else:
            return (
                all_data,
                torch.dstack((query_sizes, target_sizes)).flatten(),
                upper_bounds,
                lower_bounds,
                torch.cat(list(itertools.chain.from_iterable(zip(query_labels, target_labels)))),
            )

def get_ged_comb_data(conf, mode="train", logger=None):
    if conf.data_mode == "equal":
        return GraphEditDatasetCombination(conf, mode, logger)
    elif conf.data_mode == "unequal":
        return GraphEditDatasetCombinationCost(conf, mode, logger)        
    elif conf.data_mode == "label":
        return GraphEditDatasetCombinationSeparateLabel(conf, mode, logger)
    else:
        raise NotImplementedError
    