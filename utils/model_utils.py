import torch
from GMN.segment import unsorted_segment_sum
import torch.nn.functional as F

def get_default_gmn_config(conf):
    """The default configs."""
    model_type = "matching"
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 32
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type="embedding",
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"  # other: euclidean, cosine
    graph_matching_net_config["prop_type"] = "matching"  # other: euclidean, cosine
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=1,
            edge_hidden_sizes=None,
        ),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem="graph_edit_distance",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000,
            ),
        ),
        training=dict(
            batch_size=20,
            learning_rate=1e-4,
            mode="pair",
            loss="margin",  # other: hamming
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=500000,
            # Print training information every this many training steps.
            print_after=100,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10,
        ),
        evaluation=dict(batch_size=20),
        seed=conf.training.seed,
    )


def modify_gmn_main_config(gmn_config, conf, logger=None):
    gmn_config["encoder"]["node_hidden_sizes"] = [conf.gmn.filters_3, conf.gmn.filters_3]  # [10]
    gmn_config["encoder"]["node_feature_dim"] = conf.dataset.one_hot_dim
    gmn_config["encoder"]["edge_feature_dim"] = 1
    gmn_config["aggregator"]["node_hidden_sizes"] = [conf.gmn.filters_3, conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["graph_transform_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["input_size"] = [conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_matching_net"]["node_hidden_sizes"] = [conf.gmn.filters_3, conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["n_prop_layers"] = 5
    gmn_config["graph_embedding_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_embedding_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_embedding_net"]["node_hidden_sizes"] = [
        conf.gmn.filters_3, conf.gmn.filters_3
    ]  # [10]
    gmn_config["graph_embedding_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    # logger.info("gmn_config param")
    # logger.info(gmn_config['graph_embedding_net'] ['n_prop_layers'] )

    gmn_config["training"]["batch_size"] = conf.training.batch_size
    # gmn_config['training']['margin']  = av.MARGIN
    gmn_config["evaluation"]["batch_size"] = conf.training.batch_size
    gmn_config["model_type"] = "embedding"
    gmn_config["graphsim"] = {}
    gmn_config["graphsim"]["conv_kernel_size"] = [10, 6, 4, 2]
    gmn_config["graphsim"]["linear_size"] = [40, 10]
    gmn_config["graphsim"]["gcn_size"] = [10, 10, 10, 10, 10]
    gmn_config["graphsim"]["conv_pool_size"] = [3, 3, 2, 2]
    gmn_config["graphsim"]["conv_out_channels"] = [2, 4, 6, 8]
    gmn_config["graphsim"]["dropout"] = conf.training.dropout

    if logger is not None:
        logger.info("Modified GMN config:")
        for k, v in gmn_config.items():
            logger.info("%s= %s" % (k, v))
    return gmn_config


def modify_gmn_main_config_shallow(gmn_config, conf, logger):
    gmn_config["encoder"]["node_hidden_sizes"] = [conf.gmn.filters_3] # [10]
    gmn_config["encoder"]["node_feature_dim"] = conf.dataset.one_hot_dim
    gmn_config["encoder"]["edge_feature_dim"] = 1
    gmn_config["aggregator"]["node_hidden_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["graph_transform_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["input_size"] = [conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_matching_net"]["node_hidden_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["n_prop_layers"] = 5
    gmn_config["graph_embedding_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_embedding_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_embedding_net"]["node_hidden_sizes"] = [ conf.gmn.filters_3]  # [10]
    gmn_config["graph_embedding_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    # logger.info("gmn_config param")
    # logger.info(gmn_config['graph_embedding_net'] ['n_prop_layers'] )

    gmn_config["training"]["batch_size"] = conf.training.batch_size
    # gmn_config['training']['margin']  = av.MARGIN
    gmn_config["evaluation"]["batch_size"] = conf.training.batch_size
    gmn_config["model_type"] = "embedding"
    # gmn_config["graphsim"] = {}
    # gmn_config["graphsim"]["conv_kernel_size"] = [10, 4, 2]
    # gmn_config["graphsim"]["linear_size"] = [24, 16]
    # gmn_config["graphsim"]["gcn_size"] = [10, 10, 10]
    # gmn_config["graphsim"]["conv_pool_size"] = [3, 3, 2]
    # gmn_config["graphsim"]["conv_out_channels"] = [2, 4, 8]
    # gmn_config["graphsim"]["dropout"] = conf.training.dropout
    gmn_config["graphsim"] = {}
    # gmn_config["graphsim"]["conv_kernel_size"] = [10, 4, 2]
    # # gmn_config["graphsim"]["linear_size"] = [24, 16]
    # gmn_config["graphsim"]["linear_size"] = [96,12]
    # # gmn_config["graphsim"]["linear_size"] = [10, 10]
    # # gmn_config["graphsim"]["gcn_size"] = [10, 10, 10]
    # gmn_config["graphsim"]["gcn_size"] = [conf.gmn.filters_3, conf.gmn.filters_3, conf.gmn.filters_3] #, conf.gmn.filters_3, conf.gmn.filters_3]
    # gmn_config["graphsim"]["conv_pool_size"] = [3, 3, 2]
    # gmn_config["graphsim"]["conv_out_channels"] = [2, 4, 8]
    gmn_config["graphsim"]["conv_kernel_size"] = [10, 6, 4, 2]
    gmn_config["graphsim"]["linear_size"] = [40, 10]
    gmn_config["graphsim"]["gcn_size"] = [10, 10, 10, 10, 10]
    gmn_config["graphsim"]["conv_pool_size"] = [3, 3, 2, 2]
    gmn_config["graphsim"]["conv_out_channels"] = [2, 4, 6, 8]
    gmn_config["graphsim"]["dropout"] = conf.training.dropout
    logger.info("Modified GMN config:")
    for k, v in gmn_config.items():
        logger.info("%s= %s" % (k, v))
    return gmn_config


def get_graph_features(graphs):
    return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx    

def pytorch_sample_gumbel(shape, device, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape, device=device, dtype=torch.float)
    return -torch.log(eps - torch.log(U + eps))

def pytorch_sinkhorn_iters(log_alpha, device, temperature=0.1, noise_factor=0, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    noise = pytorch_sample_gumbel([batch_size, num_objs, num_objs], device) * noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, num_objs, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, num_objs)
    return torch.exp(log_alpha)

def graph_size_to_mask_map(max_set_size, lateral_dim, device=None):
    return [torch.cat((
        torch.tensor([1], device=device, dtype=torch.float).repeat(x, 1).repeat(1, lateral_dim),
        torch.tensor([0], device=device, dtype=torch.float).repeat(max_set_size - x, 1).repeat(1, lateral_dim)
    )) for x in range(0, max_set_size + 1)]

def set_size_to_mask_map(max_set_size, device=None):
    # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
    return [torch.cat(
            (
                torch.repeat_interleave(torch.tensor([1, 0], device=device, dtype=torch.float), torch.tensor([x, max_set_size - x], device=device)).repeat(x, 1),
                torch.repeat_interleave(torch.tensor([1, 0], device=device, dtype=torch.float), torch.tensor([0,max_set_size], device=device)).repeat(max_set_size - x, 1),
            )
        ) for x in range(0, max_set_size + 1)]

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_padded_indices(paired_sizes, max_set_size, device):
    num_pairs = len(paired_sizes)
    max_set_size_arange = torch.arange(max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(num_pairs * 2, 1)
    flattened_sizes = torch.tensor(flatten_list_of_lists(paired_sizes), device=device)
    presence_mask = max_set_size_arange < flattened_sizes.unsqueeze(1)

    cumulative_set_sizes = torch.cumsum(torch.tensor(
        max_set_size, dtype=torch.long, device=device
    ).repeat(len(flattened_sizes)), dim=0)
    max_set_size_arange[1:, :] += cumulative_set_sizes[:-1].unsqueeze(1)
    return max_set_size_arange[presence_mask]

def split_to_query_and_corpus(features, graph_sizes):
    features_split = torch.split(features, graph_sizes, dim=0)
    features_query = features_split[0::2]
    features_corpus = features_split[1::2]
    return features_query, features_corpus

def split_and_stack(features, graph_sizes, max_set_size):
    features_query, features_corpus = split_to_query_and_corpus(features, graph_sizes)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_query), stack_features(features_corpus)


def embed_mat_l1_dist(query_features, corpus_features, transport_plan):
    return torch.abs(
        query_features - transport_plan @ corpus_features,
    ).sum(dim=(-1, -2))

def query_uncoverage_dist(query_features, corpus_features, transport_plan):
    return torch.maximum(
        query_features - transport_plan @ corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(-1, -2))


def corpus_uncoverage_dist(query_features, corpus_features, transport_plan):
    return torch.maximum(
        transport_plan @ corpus_features - query_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(-1, -2))
    
def node_align_add_and_delete_distance(all_node_masks, transport_plan, add_cost, del_cost):
    qnode_masks = all_node_masks[0::2]
    cnode_masks = all_node_masks[1::2]
    add_masks = qnode_masks[:,:,None]@((1-cnode_masks)[:,None,:])
    del_masks = ((1-qnode_masks)[:,:,None])@cnode_masks[:,None,:]
    return add_cost* torch.sum(add_masks*transport_plan, dim=(-1,-2)) +\
           del_cost* torch.sum(del_masks*transport_plan, dim=(-1,-2))
           

def our_matmul(a,b):
    #b*i*j, b*j ->  b*i 
    #return torch.einsum("bij,bj -> bi", a,b)
    return torch.bmm(a,b[:,:,None]).squeeze()
        
def node_align_add_and_delete_distance_alternate(all_node_masks, transport_plan, add_cost, del_cost):
    qnode_masks = all_node_masks[0::2]
    cnode_masks = all_node_masks[1::2]

    return del_cost *  torch.sum(torch.nn.ReLU()(qnode_masks - our_matmul(transport_plan,cnode_masks)) , dim=(-1)) +\
    add_cost *  torch.sum(torch.nn.ReLU()( our_matmul(transport_plan,cnode_masks)-qnode_masks), dim=(-1))    


def node_align_add_and_delete_distance_one_norm(all_node_masks, transport_plan, add_cost, del_cost):
    assert del_cost == add_cost
    qnode_masks = all_node_masks[0::2]
    cnode_masks = all_node_masks[1::2]

    return  add_cost* torch.sum(torch.abs(qnode_masks - our_matmul(transport_plan,cnode_masks)) , dim=(-1))    


def get_paired_edge_counts(from_idx, to_idx, graph_idx, num_graphs):
    edges_per_src_node = unsorted_segment_sum(torch.ones_like(from_idx, dtype=torch.float), from_idx, len(graph_idx))
    edges_per_graph_from = unsorted_segment_sum(edges_per_src_node, graph_idx, num_graphs)

    edges_per_dest_node = unsorted_segment_sum(torch.ones_like(to_idx, dtype=torch.float), to_idx, len(graph_idx))
    edges_per_graph_to = unsorted_segment_sum(edges_per_dest_node, graph_idx, num_graphs)

    assert (edges_per_graph_from == edges_per_graph_to).all()

    return edges_per_graph_to.int().tolist()

def propagation_messages(propagation_layer, node_features, edge_features, from_idx, to_idx):
    edge_src_features = node_features[from_idx]
    edge_dest_features = node_features[to_idx]

    forward_edge_msg = propagation_layer._message_net(torch.cat([
        edge_src_features, edge_dest_features, edge_features
    ], dim=-1))
    backward_edge_msg = propagation_layer._reverse_message_net(torch.cat([
        edge_dest_features, edge_src_features, edge_features
    ], dim=-1))
    return forward_edge_msg + backward_edge_msg

def kronecker_product_on_nodes(node_transport_plan, from_idx, to_idx, paired_edge_counts, graph_sizes, max_edge_set_size):
    segregated_edge_vertices = torch.split(
        torch.cat([from_idx.unsqueeze(-1), to_idx.unsqueeze(-1)], dim=-1),
        paired_edge_counts, dim=0
    )
    batched_edge_vertices_shifted = torch.cat([
        F.pad(z_idxs, pad=(0, 0, 0, max_edge_set_size - len(z_idxs)), value=-1).unsqueeze(0)
    for z_idxs in segregated_edge_vertices]).long()

    node_count_prefix_sum = torch.zeros(len(graph_sizes), device=to_idx.device, dtype=torch.long)
    node_count_prefix_sum[1:] = torch.cumsum(torch.tensor(graph_sizes, device=to_idx.device), dim=0)[:-1]
    
    batched_edge_vertices = batched_edge_vertices_shifted - node_count_prefix_sum.view(-1, 1, 1)
    batched_edge_vertices[batched_edge_vertices < 0] = -1

    query_from_vertices = batched_edge_vertices[::2, :, 0].unsqueeze(-1)
    query_to_vertices = batched_edge_vertices[::2, :, 1].unsqueeze(-1)
    corpus_from_vertices = batched_edge_vertices[1::2, :, 0].unsqueeze(1)
    corpus_to_vertices = batched_edge_vertices[1::2, :, 1].unsqueeze(1)
    batch_arange = torch.arange(len(graph_sizes)//2, device=to_idx.device, dtype=torch.long).view(-1, 1, 1)
    edge_presence_mask = (query_from_vertices >= 0) * (corpus_from_vertices >= 0)

    straight_mapped_scores = torch.mul(
        node_transport_plan[batch_arange, query_from_vertices, corpus_from_vertices],
        node_transport_plan[batch_arange, query_to_vertices, corpus_to_vertices]
    ) * edge_presence_mask
    cross_mapped_scores = torch.mul(
        node_transport_plan[batch_arange, query_from_vertices, corpus_to_vertices],
        node_transport_plan[batch_arange, query_to_vertices, corpus_from_vertices]
    ) * edge_presence_mask
    return straight_mapped_scores, cross_mapped_scores



def compute_loss(lower_bound, upper_bound, out):
    loss = (
        torch.nn.functional.relu(lower_bound - out) ** 2
        + torch.nn.functional.relu(out - upper_bound) ** 2
    )
    return loss.mean()

def asymm_embed_mat_l1_dist(query_features, corpus_features, transport_plan, add_cost, del_cost):
    # return torch.abs(
        # query_features - transport_plan @ corpus_features,
    # ).sum(dim=(-1, -2))
    transformed_corpus = transport_plan @ corpus_features
    return del_cost * F.relu(query_features - transformed_corpus).sum(dim=(-1, -2)) +\
              add_cost * F.relu(transformed_corpus - query_features).sum(dim=(-1, -2))


def asymm_norm(x, y, p=1, node_ins_cost=1, node_del_cost=1, edge_ins_cost=1, edge_del_cost=1):
    # return torch.norm(x - y, dim=-1, p=p)
    return (F.relu(x - y) * (node_del_cost + edge_del_cost) * 0.5).sum(dim=-1) + (F.relu(y - x) * (node_ins_cost + edge_ins_cost) * 0.5).sum(dim=-1)