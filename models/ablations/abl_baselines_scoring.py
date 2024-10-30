import torch
import torch_geometric as pyg
import utils.model_utils as model_utils

import GMN.graphembeddingnetwork as gmngen
import GMN.utils as gmnutils
import utils.model_utils as model_utils
import GMN.fast_graphmatchingnetwork as fast_gmngmn


class EmbedModel(torch.nn.Module):
    def __init__(self, conf, *_):
        super().__init__()
        self.n_layers = conf.model.n_layers
        self.input_dim = conf.dataset.one_hot_dim
        self.hidden_dim = conf.model.hidden_dim
        self.output_dim = conf.model.output_dim
        conv = conf.model.conv
        pool = conf.model.pool
        
        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)
        
        if conv == 'gin':
            make_conv = lambda:\
                pyg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                ))
        elif conv == 'gcn':
            make_conv = lambda:\
                pyg.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'sage':
            make_conv = lambda:\
                pyg.nn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv == 'gat':
            make_conv = lambda:\
                pyg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        else:
            assert False
            
        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(make_conv())
        
        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        if pool == 'add':
            self.pool = pyg.nn.global_add_pool
        elif pool == 'mean':
            self.pool = pyg.nn.global_mean_pool
        elif pool == 'max':
            self.pool = pyg.nn.global_max_pool
        elif pool == 'sort':
            self.pool = pyg.nn.global_sort_pool
        elif pool == 'att':
            self.pool = pyg.nn.GlobalAttention(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1)
            ))
        elif pool == 'set':
            self.pool = pyg.nn.Set2Set(self.hidden_dim*(self.n_layers+1),1)
        self.pool_str = pool
        
    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i&1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)
        
        x = emb
        if self.pool_str == 'sort':
            x = self.pool(x, g.batch, k=1)
        else:
            x = self.pool(x, g.batch)
        
        x = self.post(x)
        return x





class SiameseModel(torch.nn.Module):
    def __init__(self, conf, *_):
        super().__init__()
        self.embed_model = None
        self.weighted = False
        self.device = conf.training.device

        
    def forward_emb(self, gx, hx):
        raise NotImplementedError

    def forward(self, g, h):
        # if self.weighted:
        #     self.gs = torch.tensor([x.num_nodes for x in g.to_data_list()], device=self.device)
        #     self.hs = torch.tensor([x.num_nodes for x in h.to_data_list()], device=self.device)
        gx = self.embed_model(g)
        hx = self.embed_model(h)
        return self.forward_emb(gx, hx)
    
    def predict_inner(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries) <= batch_size:
            #logger.info(f'direct predict inner dataset')
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            h = pyg.data.Batch.from_data_list(targets).to(self.device)
            with torch.no_grad():
                return self.forward(g, h)
        else:
            #logger.info(f'batch predict inner dataset')
            #logger.info(f'1: {1}')
            loader = pyg.data.DataLoader(list(zip(queries, targets)), batch_size, num_workers=1)
            ret = torch.empty(len(queries), device=self.device)
            for i, (g, h) in enumerate((loader, 'batches')):
                g = g.to(self.device)
                h = h.to(self.device)
                with torch.no_grad():
                    ret[i*batch_size:(i+1)*batch_size] = self.forward(g, h)
            return ret
    
    def predict_outer(self, queries, targets, batch_size=None):
        self = self.to(self.device)
        if batch_size is None or len(queries)*len(targets) <= batch_size:
            #logger.info(f'direct predict outer dataset')
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            h = pyg.data.Batch.from_data_list(targets).to(self.device)
            gx = self.embed_model(g)
            hx = self.embed_model(h)
            with torch.no_grad():
                return self.forward_emb(gx[:,None,:], hx)
        else:
            #logger.info(f'batch predict outer dataset')
            #logger.info(f'1: {1}')
            g = pyg.data.Batch.from_data_list(queries).to(self.device)
            gx = self.embed_model(g)
            loader = pyg.data.DataLoader(targets, batch_size//len(queries), num_workers=1)
            ret = torch.empty(len(queries), len(targets), device=self.device)
            for i, h in enumerate((loader, 'batches')):
                h = h.to(self.device)
                hx = self.embed_model(h)
                with torch.no_grad():
                    ret[:,i*loader.batch_size:(i+1)*loader.batch_size] = self.forward_emb(gx[:,None,:], hx)
            return ret

    def compute_loss(self, lb, ub, pred):
        loss = torch.nn.functional.relu(lb-pred)**2 + torch.nn.functional.relu(pred-ub)**2
        loss = torch.mean(loss)
        return loss
  

class Greed_Hinge(SiameseModel):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.embed_model = EmbedModel(conf)
        self.output_mode = conf.model.output_mode
        self.node_ins_cost = conf.dataset.node_ins_cost
        self.node_del_cost = conf.dataset.node_del_cost
        self.node_rel_cost = conf.dataset.node_rel_cost
        self.edge_ins_cost = conf.dataset.edge_ins_cost
        self.edge_del_cost = conf.dataset.edge_del_cost
        self.edge_rel_cost = conf.dataset.edge_rel_cost
        self.norm_mode = conf.model.norm_mode
        assert self.norm_mode == 'asymm'

        
    def forward_emb(self, x, y):
        # return torch.norm(gx-hx, dim=-1, p=int(self.output_mode[-1]))
        return model_utils.asymm_norm(x, y, int(self.output_mode[-1]), self.node_ins_cost, self.node_del_cost, self.edge_ins_cost, self.edge_del_cost)



class Embed_Hinge(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        """ """
        super(Embed_Hinge, self).__init__()
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
        assert self.norm_mode == 'asymm'

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

        return model_utils.asymm_norm(x, y, int(self.output_mode[-1]), self.node_ins_cost, self.node_del_cost, self.edge_ins_cost, self.edge_del_cost)

    def compute_loss(self, lower_bound, upper_bound, out):
        loss = (
            torch.nn.functional.relu(lower_bound - out) ** 2
            + torch.nn.functional.relu(out - upper_bound) ** 2
        )
        return loss.mean()


class Match_Hinge(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        """ """
        super(Match_Hinge, self).__init__()
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
        return model_utils.asymm_norm(x, y, int(self.output_mode[-1]), self.node_ins_cost, self.node_del_cost, self.edge_ins_cost, self.edge_del_cost)


    def compute_loss(self, lower_bound, upper_bound, out):
        loss = (
            torch.nn.functional.relu(lower_bound - out) ** 2
            + torch.nn.functional.relu(out - upper_bound) ** 2
        )
        return loss.mean()
    
