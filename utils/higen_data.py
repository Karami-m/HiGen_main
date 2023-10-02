import time, pickle
import numpy as np
import torch
import torch.nn.functional as F

## The following might be needed for this issue:
# https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open"""
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Manager
from torch_geometric.data import Batch
from utils.graph_helper import *

__all__ = ['HigenData', 'get_augmented_graph', 'get_augmented_all_biparts', 'get_augmented_AR_biparts']


def get_augmented_graph(adj_full_part, parent_graph, part_start, node_id_in_part, part_id_of_node,
                        ord_, l_, no_self_edge, edge_feat_dim, max_num_nodes,
                        sum_leafedges, num_leafedges, config,
                        to_train=True, **kwargs):
    hp = config.model
    _K = 1
    use_gran_feat = config.model.get('use_gran_feat', 'only')
    has_edge_weight = hp['has_edge_weight']
    tril_initfeat = 1
    # if is_connected_graph: one edge in each cutset is always reserved to make sure cutset_weight is greater than 1
    is_connected_graph = int(hp.get('is_connected_graph', 1) )
    model_selfedge = False if no_self_edge else  hp['model_selfedge']

    if no_self_edge:
        adj_full_part -= torch.diag(torch.diag(adj_full_part)) if adj_full_part.numel() > 0 else adj_full_part

    num_nodes_ = node_id_in_part + _K
    adj_0 = adj_full_part[:node_id_in_part, :node_id_in_part]
    adj_block = F.pad(adj_0, (0, _K, 0, _K), 'constant', value=1.0)
    adj_block_sp = adj_block.to_sparse().coalesce()
    edge_index, edge_weight = adj_block_sp.indices().long(), adj_block_sp.values()
    adj_0_sp = adj_0.to_sparse().coalesce()
    edge_index_orig, edge_weight_orig = adj_0_sp.indices().long(), adj_0_sp.values()

    ## to get attention index. if existing node: 0, else for newly added node: 1, ..., _K
    if node_id_in_part == 0:
        att_idx = torch.arange(1, _K + 1, dtype=torch.int64, device=adj_full_part.device)
    else:
        att_idx = torch.cat(
            [torch.zeros(node_id_in_part, dtype=torch.int64, device=adj_full_part.device),
             torch.arange(1, _K + 1, dtype=torch.int64, device=adj_full_part.device)]
        )
    att_idx = att_idx.view(-1, 1)
    ## edge attributes: tpo create a one-hot feature
    edge_feat = torch.zeros(edge_index.shape[1], edge_feat_dim, device=edge_weight.device)
    if edge_index.shape[1] > 0:
        edge_feat = edge_feat.scatter(1, att_idx[[edge_index[0, :]]], edge_weight.unsqueeze(1))
        edge_feat = edge_feat.scatter(1, att_idx[[edge_index[1, :]]] + int(edge_feat_dim / 2), edge_weight.unsqueeze(1))

    ## get node feature index for GNN input
    if node_id_in_part == 0:
        node_idx_feat = torch.zeros(_K, dtype=torch.float, device=edge_weight.device)
    else:
        node_idx_feat = torch.cat(
            [torch.arange(part_start, part_start + node_id_in_part) + ord_ * max_num_nodes + 1, torch.zeros(_K)])

    ## to get node index for GNN output
    edge_aug_from, edge_aug_to = torch.meshgrid(
        torch.arange(node_id_in_part, node_id_in_part + _K, device = adj_full_part.device),
        torch.arange(node_id_in_part + _K, device = adj_full_part.device)
    )

    self_weight = torch.tensor(0., device=edge_weight.device)
    if model_selfedge or no_self_edge:
        edge_aug_from, edge_aug_to = edge_aug_from[:, :node_id_in_part], edge_aug_to[:,:node_id_in_part]
        self_weight = adj_full_part[num_nodes_ - 1, num_nodes_ - 1] if to_train else torch.nan
    edge_index_aug = torch.cat([edge_aug_from, edge_aug_to], dim=0).long()

    ## to get labels (edge weights)
    label = torch.tensor([]).float()
    all_weight = parent_graph.adj(is_weighted=has_edge_weight)[part_id_of_node, part_id_of_node]
    cutset_weight = torch.nan
    existing_weight = torch.tril(adj_0).sum()
    if to_train:
        if has_edge_weight and (torch.tril(adj_full_part).sum() != all_weight.item()):
            all_weight = torch.tril(adj_full_part).sum()
        _adj_full2 = F.pad(adj_full_part, (0, _K, 0, _K), 'constant', value=0.)
        label = _adj_full2[edge_aug_from, edge_aug_to].flatten()
        cutset_weight = label.sum()

    init_feat = tril_and_pad(
        adj_block[:node_id_in_part, :node_id_in_part],
        padded_size=max_num_nodes,
        zero_rows=_K,
        tril_mode=tril_initfeat
    )

    remaining_weight = all_weight - existing_weight - is_connected_graph if num_nodes_ > 1 else all_weight
    cutself_weight = self_weight + cutset_weight - is_connected_graph if num_nodes_ > 1 else self_weight

    graph_name = f"{parent_graph.id}_O{ord_}_L{l_}_part{part_id_of_node}_subg{node_id_in_part}" if to_train else \
        f"{parent_graph.id}_O{ord_}_L{l_}_part{part_id_of_node}"
    graph_sample = Data2(
        ord=ord_,
        graph_name=graph_name,
        level=l_,
        num_nodes=num_nodes_,
        num_nodes_aug_graph=torch.tensor(num_nodes_).long(),
        num_nodes_of_part=adj_full_part.shape[0],
        max_num_nodes=max_num_nodes,
        parent_graph_name=parent_graph.graph_name,
        part_id=part_id_of_node,
        parent_node_id= torch.tensor([part_id_of_node]*num_nodes_, device=edge_weight.device),
        is_part=True,
        is_subg=True,
        ## edge level attributes
        edge_index=edge_index,
        edge_feat=edge_feat,
        edge_weight=edge_weight,
        edge_index_orig=edge_index_orig,
        edge_weight_orig=edge_weight_orig,
        ## node level attributes
        x=init_feat.float(),  # node features
        node_idx_feat=node_idx_feat.long(),
        part_id_of_node=torch.tensor([part_id_of_node] * num_nodes_).long(),
        ## new (augmented) hypothetical edges
        edge_index_aug=edge_index_aug.long(),
        num_aug_edges=edge_index_aug.shape[1],
        label=label,
        cutself_weight=cutself_weight,
        cutset_weight=cutset_weight,
        self_weight=self_weight,
        remaining_weight=remaining_weight,
        existing_weight=existing_weight,
        all_weight=all_weight,
        num_parts_inlevel=parent_graph.num_nodes if not (parent_graph is None) else 1,
        sum_leafedges=sum_leafedges, num_leafedges=num_leafedges,
        **kwargs
    )

    # to add PESE node features
    if use_gran_feat in ['plus', 'none']:
        g_smpl = Data2(num_nodes=num_nodes_, edge_index=edge_index_orig, edge_weight=edge_weight_orig)
        g_smpl = g_smpl.get_PESE_node_feat(config.model.node_feat_types, config=config, is_weighted=has_edge_weight)
        for key_ in g_smpl.node_feat_PESE_names:
            graph_sample[key_] = g_smpl[key_]

        if use_gran_feat in ['none']:
            del (graph_sample.edge_feat)
            graph_sample.x = torch.zeros([num_nodes_, 0], dtype=torch.float, device=edge_weight.device)
            graph_sample.edge_index = graph_sample.edge_index_orig
            graph_sample.edge_weight = graph_sample.edge_weight_orig
            graph_sample.edge_weight_aug = torch.ones(graph_sample.edge_index_aug.shape[1],
                                                      dtype=torch.float, device=edge_weight.device) * all_weight

    return graph_sample


def get_augmented_all_biparts(graph_parent: Data2, to_joint_aug_bp, ord, level,
                              to_train, edge_feat_dim, sum_leafedges, num_leafedges, config):
    hp = config.model
    has_edge_weight = hp['has_edge_weight']

    device = graph_parent.edge_index.device
    # add positional and structural embedding to all parts
    for part in graph_parent.child_part:
        part = part.get_PESE_node_feat(config.model.node_feat_types, config=config) #.to(device)
        node_feat_names = part.node_feat_PESE_names
        part.edge_index_dense = part.get_complement_edges()

    # add augmented edges for all biparts
    bp_aug_list = []
    bp_aug_t_list = []
    visited_edges = set()
    for e_, bipart_sample in enumerate(graph_parent.child_bipart):
        edge_ = (graph_parent.edge_index[0, e_].item(), graph_parent.edge_index[1, e_].item())
        if (edge_[0] == edge_[1]):
            # self edges has a None bipart
            assert bipart_sample is None
            continue
        if (edge_[1], edge_[0]) in visited_edges:  # this is a self edge which does not have a bipart
            continue
        visited_edges.add(edge_)

        bp_aug = bipart_sample.__copy__()
        part_left = bp_aug.part_left
        part_right = bp_aug.part_right

        edge_aug_from, edge_aug_to = torch.meshgrid(
            torch.arange(part_left.num_nodes, device=device),
            torch.arange(part_right.num_nodes, device=device)
        )
        edge_aug_from = edge_aug_from.reshape(1, -1)
        edge_aug_to = edge_aug_to.reshape(1, -1)
        bp_aug.edge_index_aug = torch.cat([edge_aug_from, edge_aug_to], dim=0)
        bp_aug.edge_weight_aug = torch.ones(bp_aug.edge_index_aug.shape[1],
                                            dtype=torch.float, device=device) * graph_parent.edge_weight[e_]
        bp_aug.edge_index = torch.zeros([2, 0], dtype=torch.long, device=device)
        bp_aug.edge_weight = torch.zeros([0], dtype=torch.long, device=device)
        ### get predict label
        label = torch.tensor([], device=device, dtype=torch.float)
        if to_train:
            adj_bipart = bipart_sample.adj(is_weighted=has_edge_weight)
            label = adj_bipart[edge_aug_from, edge_aug_to].flatten().float()
            if has_edge_weight:
                assert graph_parent.edge_weight[e_].item() == label.sum().item()
        bp_aug.label = label
        bp_aug_list.append(bp_aug)
        bp_aug_t_list.append(bp_aug.t())

    bp_aug_list = bp_aug_list + bp_aug_t_list

    # collate all
    connected_aug_graph = collate_partitions(
        level = level,
        part_ls = graph_parent.child_part,
        bipart_ls = bp_aug_list,
        exclude_keys=['child_bipart', 'child_part'],
        compact=True, add_aug_edge=True, add_dense_edge=True, add_label=True,
        node_feat_names = node_feat_names,
        device=device,
        add_edge_feat=False
    )

    num_nodes_ = connected_aug_graph.num_nodes
    num_biparts_inlevel = sum([0 if a_ is None else 1 for a_ in graph_parent.child_bipart])
    graph_aug = Data2(
        ord=ord,
        # graph_name=graph_name,
        level=level,
        num_nodes=num_nodes_,
        num_nodes_aug_graph=torch.tensor(num_nodes_).long(),
        parent_graph_name=graph_parent.graph_name,
        is_part=False,
        is_subg=False,
        ## edge level attributes
        edge_index=connected_aug_graph.edge_index,
        edge_feat=connected_aug_graph.edge_feat,
        edge_weight=connected_aug_graph.edge_weight,
        # new (augmented) edges
        edge_index_aug = connected_aug_graph.edge_index_aug.long(),
        edge_weight_aug = connected_aug_graph.edge_weight_aug,
        num_aug_edges=connected_aug_graph.num_aug_edges,
        label=connected_aug_graph.label,
        edge_index_dense=connected_aug_graph.edge_index_dense.long(),
        # node level attributes
        parent_node_id = torch.tensor(connected_aug_graph.part_id_of_node, device=connected_aug_graph.edge_index.device),
        sum_leafedges=sum_leafedges, num_leafedges=num_leafedges,
        num_biparts_inlevel= num_biparts_inlevel ,
    )
    # add extra node features
    graph_aug.node_features_names = node_feat_names
    for feat_name in node_feat_names:
        graph_aug[feat_name] = connected_aug_graph[feat_name]

    del connected_aug_graph
    return graph_aug


def get_augmented_AR_biparts(bipart_sample: Bipart, graph_parent: Data2, to_joint_aug_bp, ord, level,
                              to_train, edge_feat_dim, sum_leafedges, num_leafedges, parent_edge_id, config):
    '''
    returns the collated graph of the union of all preceding parts and biparts {part_0, ...., part_{max(bp_edge)}}
    To apply GNN on union of preceding parts and biparts, edge attributes are used to distinguish the augmented
    edges from existing ones
    '''
    hp = config.model

    assert (graph_parent.edge_index[0, parent_edge_id] > graph_parent.edge_index[1, parent_edge_id])
    add_PESE_per_community = True

    has_edge_weight = hp['has_edge_weight']
    device = graph_parent.edge_index.device

    part_left = bipart_sample.part_left
    part_right = bipart_sample.part_right
    bp_aug = bipart_sample.__copy__()
    bp_aug.edge_index = torch.zeros([2, 0], dtype=torch.long, device=device)
    bp_aug.edge_weight = torch.zeros([0], dtype=torch.long, device=device)

    edge_aug_from, edge_aug_to = torch.meshgrid(
        torch.arange(part_left.num_nodes, device=device),
        torch.arange(part_right.num_nodes, device=device)
    )
    edge_aug_from = edge_aug_from.reshape(1, -1)
    edge_aug_to = edge_aug_to.reshape(1, -1)
    bp_aug.edge_index_aug = torch.cat([edge_aug_from, edge_aug_to], dim=0)
    bp_aug.edge_weight_aug = torch.ones(bp_aug.edge_index_aug.shape[1],
                                        dtype=torch.float, device=device) * graph_parent.edge_weight[parent_edge_id]
    ### get predict label
    label = torch.tensor([], device=device, dtype=torch.float)
    if to_train:
        adj_bipart = bipart_sample.adj(is_weighted=has_edge_weight)
        label = adj_bipart[edge_aug_from, edge_aug_to].flatten().float()
    bp_aug.label = label

    bp_pre_ls, part_pre_ls = [], []
    for i_ in range(max(bp_aug.part_left.part_id, bp_aug.part_right.part_id) + 1):
        part_abs_ = graph_parent.child_part[i_].clone_abs() #.to(device='cpu')
        if add_PESE_per_community:
            # add positional and structural embedding to all parts
            part_abs_ = part_abs_.get_PESE_node_feat(config.model.node_feat_types, config=config)
            node_feat_names = part_abs_.node_feat_PESE_names
        part_pre_ls.append(part_abs_)
    bp_aug_parent_edge = graph_parent.edge_index[:, parent_edge_id]
    assert (bp_aug_parent_edge == torch.tensor([bp_aug.part_left.part_id, bp_aug.part_right.part_id]).to(
        graph_parent.edge_index)).all()
    for bp_pre_id, bp_edge_ in enumerate(graph_parent.edge_index.t()):
        if (bp_edge_[1] >= bp_edge_[0]):
            continue
        if (bp_edge_[0] > bp_aug_parent_edge[0]) or ((bp_edge_[0] == bp_aug_parent_edge[0]) and (bp_edge_[1] >= bp_aug_parent_edge[1])):
            continue
        bp_pre_ = graph_parent.child_bipart[bp_pre_id] #.to(device='cpu')
        bp_pre_t_ = bp_pre_.t()
        bp_pre_ls += [bp_pre_, bp_pre_t_]

    connected_aug_graph = collate_partitions(
        level=level,
        part_ls=part_pre_ls,
        bipart_ls=bp_pre_ls + [bp_aug, bp_aug.t()],
        exclude_keys=['child_bipart', 'child_part'],
        compact=True,
        add_aug_edge=True,
        add_label=True,
        device = device,
        add_edge_feat = False,
        node_feat_names=node_feat_names if add_PESE_per_community else [],
    )

    bp_edge_bias = torch.tensor(
        [[connected_aug_graph.part_start_dict[bp_aug.part_left.part_id]],
        [connected_aug_graph.part_start_dict[bp_aug.part_right.part_id]]],
        device= device)

    # add positional and structural embedding to augmented graph
    if not add_PESE_per_community:
        connected_aug_graph = connected_aug_graph.get_PESE_node_feat(config.model.node_feat_types, config=config) #.to(device)
        node_feat_names = connected_aug_graph.node_feat_PESE_names

    num_nodes_ = connected_aug_graph.num_nodes
    num_biparts_inlevel = sum([0 if a_ is None else 1 for a_ in graph_parent.child_bipart])
    graph_aug = Data2(
        ord=ord,
        # graph_name=graph_name,
        level=level,
        num_nodes=num_nodes_,
        num_nodes_aug_graph=torch.tensor(num_nodes_).long(),
        parent_graph_name=graph_parent.graph_name,
        is_part=False,
        is_subg=False,
        ## edge level attributes
        edge_index=connected_aug_graph.edge_index,
        edge_feat=connected_aug_graph.edge_feat,
        edge_weight=connected_aug_graph.edge_weight,
        # new (augmented) edges
        edge_index_aug=connected_aug_graph.edge_index_aug.long(),
        edge_weight_aug=connected_aug_graph.edge_weight_aug,
        num_aug_edges=connected_aug_graph.num_aug_edges,
        label=connected_aug_graph.label,
        # node level attributes
        parent_node_id=torch.tensor(connected_aug_graph.part_id_of_node, device=connected_aug_graph.edge_index.device),
        sum_leafedges=sum_leafedges, num_leafedges=num_leafedges,
        num_biparts_inlevel=num_biparts_inlevel,
    )
    # to add extra node features
    graph_aug.node_features_names = node_feat_names
    for feat_name in node_feat_names:
        graph_aug[feat_name] = connected_aug_graph[feat_name]

    if (to_train==False):
        graph_aug.bp_edge_bias = bp_edge_bias

    del connected_aug_graph
    return graph_aug


class HigenData(object):

    def __init__(self, config, file_names, num_graphs, tag='train'):
        self.config = config
        self.tag = tag
        self.num_graphs = num_graphs
        self.batch_size = config.train.batch_size
        self.device = 'cpu'
        self.device_item = 'cpu' if self.batch_size > 1 else self.device
        self.to_pin_memory = False
        self.data_path = config.dataset.data_path
        self.model_name = config.model.name
        self.add_EOS_node = ("NoNewEdge" in config.model.gen_completion)
        self.verbose = config.get('verbose', 1)

        self.np_rand = np.random.RandomState(config.seed)
        self.node_order = config.dataset.node_order
        self.is_sample_subgraph = config.dataset.is_sample_subgraph
        self.no_self_edge = config.dataset.no_self_edge
        self.is_overwrite_precompute = config.dataset.is_overwrite_precompute
        self.has_edge_weight = config.model.has_edge_weight
        self.to_joint_aug_bp = config.model.to_joint_aug_bp
        self.model_selfedge = config.model.get('model_selfedge', True)
        self.is_connected_graph = config.model.get("is_connected_graph", 1)

        self.edge_feat_dim = config.model.edge_feat_dim
        self.max_num_nodes = np.array(config.model.max_num_nodes)
        self.max_num_nodes_augbp = np.array(config.model.max_num_nodes)
        self.num_levels = config.dataset.num_levels
        self.num_subgraph_batch = config.dataset.num_subgraph_batch
        self.num_bigraph_batch = config.dataset.num_bigraph_batch
        self.num_canonical_order = config.model.num_canonical_order

        if self.is_sample_subgraph:
            assert self.num_subgraph_batch > 0

        manager = Manager()
        self.file_names = manager.list(file_names)

    def __len__(self):
        return self.num_graphs

    def file_name_unpack(self, file_name):
        return file_name

    def __getitem__(self, index):

        file_name = self.file_name_unpack(self.file_names[index])
        HG_ls = pickle.load(open(file_name, 'rb'))

        data_batch_levels = []
        # to loop over all levels
        for l_ind in range(self.num_levels):
            no_self_edge = self.no_self_edge == 'all' or (self.no_self_edge == 'last' and l_ind == self.num_levels - 1)

            parent_graph_abs = None
            part_sample_ls, bipartite_sample_ls = [], []

            # To loop over all orderings, in this work only one ordering was used
            for ord_ind, HG in enumerate(HG_ls):
                G_l = HG[l_ind]
                if G_l is None:
                    continue
                assert G_l.level == l_ind

                G_l.parent_graph = parent_graph = HG[l_ind - 1] if (l_ind > 0) and not (HG[l_ind - 1] is None) \
                    else HG.get_root_graph()
                if (l_ind > 1) and not (HG[l_ind - 2] is None):
                    parent_graph.parent_graph_name = HG[l_ind - 2].graph_name
                elif 'root' in parent_graph.graph_name:
                    parent_graph.parent_graph_name = None
                else:
                    parent_graph.parent_graph_name = HG.root_graph.graph_name

                # to get parent graph features
                parent_graph_abs = parent_graph.clone_abs()
                parent_graph_abs.get_edge_feat(edge_feat_dim=self.edge_feat_dim, is_weighted=self.has_edge_weight)
                parent_graph_abs = parent_graph_abs.get_PESE_node_feat(self.config.model.node_feat_types, config=self.config)
                parent_graph_abs.node_feat_names = parent_graph_abs.node_feat_PESE_names

                # To sample subgraph for level l_ind
                _num_nodes_withEOS = G_l.num_nodes
                if self.add_EOS_node:
                    _num_nodes_withEOS = G_l.num_nodes + len(G_l.part_ls)

                rand_perm_idx = self.np_rand.permutation(_num_nodes_withEOS).tolist()
                ff_idx_start, ff_idx_end = 0, min(self.num_subgraph_batch, _num_nodes_withEOS)
                rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

                for i_smpl in range(0, _num_nodes_withEOS):
                    if (i_smpl not in rand_idx):
                        continue

                    if i_smpl < G_l.num_nodes:
                        jj = i_smpl
                        part_start = G_l.get_part_start_of_node(node_idx=jj)
                        g_part_ = G_l.get_partition_graph_of_node(node_idx=jj)
                        part_id_of_node = G_l.part_id_of_node[jj]
                        node_id_in_part = G_l.node_id_in_part[jj]

                    else:
                        jj = i_smpl - G_l.num_nodes
                        part_start = G_l.part_start_dict[jj]
                        part_id_of_node = G_l.part_id_of_node[part_start]
                        g_part_ = G_l.get_partition_graph_of_node(node_idx=part_start)
                        node_id_in_part = g_part_.num_nodes

                    adj_full_part = g_part_.adj(is_weighted=self.has_edge_weight)
                    graph_sample = get_augmented_graph(
                        adj_full_part=adj_full_part,
                        parent_graph=parent_graph,
                        node_id_in_part=node_id_in_part,
                        part_start=part_start,
                        part_id_of_node=part_id_of_node,
                        no_self_edge=no_self_edge,
                        ord_=ord_ind,
                        l_=l_ind,
                        edge_feat_dim=self.edge_feat_dim,
                        max_num_nodes=self.max_num_nodes[l_ind],
                        config=self.config,
                        sum_leafedges=HG.sum_leafedges,
                        num_leafedges=torch.tensor(HG.num_leafedges // 2).long(),
                        num_edges_of_part=torch.tensor(g_part_.num_edges // 2).long(),
                    )
                    if ((graph_sample.edge_index_aug.numel() > 0) or (not no_self_edge)) \
                            and  not (self.is_connected_graph and graph_sample.cutset_weight == 0. and graph_sample.num_nodes > 1) :
                        part_sample_ls.append(graph_sample.to(self.device_item))


                # To sample bipartite graph
                if hasattr(parent_graph, 'is_root') and parent_graph.is_root:  # l_ind == 0:
                    continue

                parent_graph = parent_graph
                assert parent_graph.num_edges == len(G_l.biparts_ls)
                assert parent_graph.num_edges == len(parent_graph.child_bipart)
                num_bp = parent_graph.num_edges

                if self.to_joint_aug_bp.lower() == 'all':
                    bipart_aug = get_augmented_all_biparts(
                        graph_parent=parent_graph,
                        to_joint_aug_bp=self.to_joint_aug_bp,
                        ord=ord_ind,
                        level=l_ind,
                        to_train=True,
                        edge_feat_dim=self.edge_feat_dim,
                        sum_leafedges=HG.sum_leafedges,
                        num_leafedges=torch.tensor(HG.num_leafedges // 2).long(),
                        config= self.config,
                    )
                    bipartite_sample_ls.append(bipart_aug.to(self.device_item))

                elif self.to_joint_aug_bp.upper() == 'AR':
                    rand_perm_idx_bp = self.np_rand.permutation(num_bp).tolist()

                    ff_idx_start_bp, ff_idx_end_bp = 0, min(num_bp, self.num_bigraph_batch)
                    rand_idx = rand_perm_idx_bp[ff_idx_start_bp:ff_idx_end_bp]

                    num_sampled_bp = 0
                    for bp_sample_idx in rand_perm_idx_bp:
                        if num_sampled_bp == self.num_bigraph_batch:
                            break

                        bipart_sample = parent_graph.child_bipart[bp_sample_idx]
                        if bipart_sample is None:
                            # to make sure it is a self edges when a None bipart
                            assert parent_graph.edge_index[0, bp_sample_idx] == parent_graph.edge_index[1, bp_sample_idx]
                            continue
                        if (parent_graph.edge_index[0, bp_sample_idx] < parent_graph.edge_index[1, bp_sample_idx]):
                            continue
                        num_sampled_bp += 1

                        bipart_aug = get_augmented_AR_biparts(
                            bipart_sample = bipart_sample,
                            graph_parent=parent_graph,
                            to_joint_aug_bp=self.to_joint_aug_bp,
                            ord=ord_ind,
                            level=l_ind,
                            to_train=True,
                            edge_feat_dim=self.edge_feat_dim,
                            sum_leafedges=HG.sum_leafedges,
                            num_leafedges=torch.tensor(HG.num_leafedges // 2).long(),
                            parent_edge_id = bp_sample_idx,
                            config=self.config,
                        )
                        bipartite_sample_ls.append(bipart_aug.to(self.device_item))

            data_batch_levels += [(part_sample_ls, bipartite_sample_ls, parent_graph_abs)]
        return data_batch_levels

    @classmethod
    def batch_from_list_parts(cls, part_ls,
                              follow_batch=['edge_index', 'part_id', 'edge_feat', 'edge_weight', 'x', 'node_idx_feat',
                                            'part_id_of_node', 'edge_index_aug', 'label', 'parent_graph', 'bch_hg','num_samples'],
                              exclude_keys=['edge_attr_keys_extra', 'node_attr_keys_extra']):
        return Batch.from_data_list(part_ls, follow_batch=follow_batch, exclude_keys=exclude_keys)

    @classmethod
    def batch_from_list_biparts(cls, bipart_ls,
                                follow_batch=['edge_index', 'edge_feat', 'edge_weight', 'edge_index_aug', 'label',
                                              'parent_graph', 'parent_edge_index', 'parent_edge_weight',
                                              'edge_weight_aug', 'bch_hg', 'num_samples', 'x'],
                                exclude_keys=None):
        return Batch.from_data_list(bipart_ls, follow_batch=follow_batch, exclude_keys=exclude_keys)

    def collate_fn(self, batch):
        assert isinstance(batch, list)
        start_time = time.time()

        data_batch_levels = []
        for l_ in range(self.num_levels):
            parts_all_l = []
            biparts_all_l = []
            parent_graph_all_l = []
            _parent_graph_bias = 0
            for i_hg, hg_tuple in enumerate(batch):
                part_batch, bipart_batch, parent_graph = hg_tuple[l_]

                if parent_graph is None:
                    continue

                if len(part_batch) > 0:
                    for part_ in part_batch:
                        part_.bch_hg = i_hg
                        part_.num_samples = len(part_batch)
                        part_.parent_node_id += _parent_graph_bias
                else:
                    part_batch = []
                parts_all_l += part_batch

                if len(bipart_batch) > 0:
                    for bipart_ in bipart_batch:
                        bipart_.bch_hg = i_hg
                        bipart_.num_samples = len(bipart_batch)
                        bipart_.parent_node_id += _parent_graph_bias
                else:
                    bipart_batch = []
                biparts_all_l += bipart_batch

                parent_graph_all_l.append(parent_graph)
                _parent_graph_bias += parent_graph.num_nodes

            part_batch_l = None
            if len(parts_all_l) > 0:
                part_batch_l = HigenData.batch_from_list_parts(parts_all_l).to(self.device)

            bipart_batch_l = None
            if len(biparts_all_l) > 0:
                if self.to_joint_aug_bp.lower() in ['all', 'ar']:
                    bipart_batch_l = HigenData.batch_from_list_biparts(
                        biparts_all_l,
                        follow_batch=['edge_index', 'edge_feat', 'edge_weight', 'edge_index_aug', 'label',
                                      'parent_graph', 'parent_edge_index', 'parent_edge_weight',
                                      'edge_weight_aug', 'bch_hg', 'num_samples'] + \
                                     ['num_aug_edges'],
                        exclude_keys=['node_features_names']
                    ).to(self.device)
                    bipart_batch_l.node_features_names = biparts_all_l[0].node_features_names
                else:
                    bipart_batch_l = HigenData.batch_from_list_biparts(biparts_all_l).to(self.device)

            parent_batch_l = None
            if len(parent_graph_all_l) > 0:
                parent_batch_l = HigenData.batch_from_list_parts(
                    parent_graph_all_l,
                    exclude_keys=['node_features_names', 'part_start_dict', 'node_id_in_part', 'is_part', 'part_id_ls', 'ord',]
                ).to(self.device)

            data_batch_levels += [(part_batch_l, bipart_batch_l, parent_batch_l)]

        if self.verbose > 3:
            print(f"Time to collate a graph batch is {time.time() - start_time}")

        return data_batch_levels

    def to_device(self, data_batch_levels, device):
        start_time = time.time()

        data_batch_levels_out = []
        for batch_l in data_batch_levels:
            (part_bch_l, bipart_bch_l, parent_bch_l) = batch_l

            if not part_bch_l is None:
                if self.to_pin_memory:
                    part_bch_l.pin_memory()

                part_bch_l.to(device)
                for key, value in part_bch_l._inc_dict.items():
                    if torch.is_tensor(value):
                        part_bch_l._inc_dict[key] = value.to(device)
                for key, value in part_bch_l._slice_dict.items():
                    if torch.is_tensor(value):
                        part_bch_l._slice_dict[key] = value.to(device)

            if not bipart_bch_l is None:
                if self.to_pin_memory:
                    bipart_bch_l.pin_memory()

                bipart_bch_l.to(device)
                for key, value in bipart_bch_l._inc_dict.items():
                    if torch.is_tensor(value):
                        bipart_bch_l._inc_dict[key] = value.to(device)
                for key, value in bipart_bch_l._slice_dict.items():
                    if torch.is_tensor(value):
                        bipart_bch_l._slice_dict[key] = value.to(device)

            if not parent_bch_l is None:
                if self.to_pin_memory:
                    parent_bch_l.pin_memory()

                parent_bch_l.to(device)
                for key, value in parent_bch_l._inc_dict.items():
                    if torch.is_tensor(value):
                        parent_bch_l._inc_dict[key] = value.to(device)
                for key, value in parent_bch_l._slice_dict.items():
                    if torch.is_tensor(value):
                        parent_bch_l._slice_dict[key] = value.to(device)

            data_batch_levels_out += [(part_bch_l, bipart_bch_l, parent_bch_l)]

        if self.verbose > 3:
            print("Time to move a graph batch to cuda is  {time.time() - start_time}")

        return data_batch_levels_out

