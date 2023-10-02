import sys
sys.path.append('')

import os, pickle, glob
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import networkx as nx
import torch_geometric as pyg
from torch_geometric.typing import PairTensor
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
from utils.transform.posenc_stats import compute_posenc_stats

__all__ = ['Bipart', 'collate_partitions', 'Data2', 'HG', 'tril_and_pad', 'PMF_stat_HG', 'save_load_graphs']


def tril_and_pad(adj, padded_size, zero_rows=0, tril_mode=1):
    if tril_mode == 1:
        adj = torch.tril(adj, diagonal=-1)
    elif tril_mode == 0:
        adj = torch.tril(adj, diagonal=0)

    if padded_size:
        adj = F.pad(adj, (0, padded_size - adj.shape[1], 0, zero_rows), 'constant', value=0.0)

    return adj


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    # the code is adopted from torch-geometric-2.0.4
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """

    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


# To make a modified Graph Data structure that will be used in HiGen
class Data2(Data):
    def __init__(self, *args, **kwargs):
        super(Data2, self).__init__(*args, **kwargs)
        self.node_attr_keys_extra = []
        self.edge_attr_keys_extra = []
        self.id = '' if not hasattr(self, 'id') else self.id
        self.assign_name()


    @property
    def parent_node_ind(self):
        return self.part_id_of_node

    def is_node_attr(self, key: str) -> bool:
        return self._store.is_node_attr(key) or (key in self.node_attr_keys_extra)

    def is_edge_attr(self, key: str) -> bool:
        return self._store.is_edge_attr(key) or (key in self.edge_attr_keys_extra)

    def is_batch(self):
        return not (self.batch is None)

    def to_networkx(self):
        return pyg.utils.to_networkx(self)

    def clone_abs(self, exclude_keys=['biparts_ls', 'part_ls', 'child_part', 'child_bipart', 'parent_graph']):
        """ to return an abstract Data graph by removing some extra features. It's used to save memory"""
        keys = set(self.keys) - set(exclude_keys)
        return Data2(**dict([(key_, self[key_]) for key_ in keys])).to(self.edge_index.device)

    def degree(self, is_weighted=False):
        _adj = self.adj(is_weighted=is_weighted)
        return _adj.sum(dim=1)

    def assign_name(self):
        if hasattr(self, 'graph_name'):
            return
        if hasattr(self, 'is_part') and hasattr(self, 'ord'):
            if self.is_part:
                self.graph_name = "O{self.ord}_L{self.level}_part{self.part_id}"
            else:
                self.graph_name = "O{self.ord}_L{self.level}"
            self._g_name = self.graph_name + ''
            self.graph_name = "{str(self.id)}_{self.graph_name}" if (hasattr(self,'id') and self.id) else self.graph_name
        return

    def set_id(self, id):
        self.id = id
        if self._g_name != '':
            self.graph_name = "{str(self.id)}_{self._g_name}"

    def get_partition_graph_of_node(self, node_idx):
        if (hasattr(self.parent_graph, 'is_root') and self.parent_graph.is_root) or self.level == 0:
            # partition graph in this case the root graph
            return self
        else:
            return self.parent_graph.child_part[self.part_id_of_node[node_idx]]

    def adj_partition_graph_of_node(self, node_idx, is_weighted=False):
        _partition_graph = self.get_partition_graph_of_node(node_idx)
        adj = pyg.utils.to_dense_adj(
            edge_index = _partition_graph.edge_index,
            max_num_nodes = _partition_graph.num_nodes,
            edge_attr = _partition_graph.edge_weight if is_weighted else None,
        )
        return adj[0].type(torch.float)

    def get_part_start_of_node(self, node_idx):
        return self.part_start_dict[self.part_id_of_node[node_idx]]

    def get_edge_feat(self, edge_feat_dim, is_weighted):
        """This is used for GRAN type GNN"""
        self.edge_feat = torch.zeros(
            self.edge_index.shape[1], edge_feat_dim,
            device=self.edge_weight.device
        )
        self.edge_feat[:, 0] = self.edge_weight if is_weighted else 1.
        self.edge_feat[:, int(edge_feat_dim/2)] = self.edge_weight if is_weighted else 1.
        return self.edge_feat

    def get_init_node_feat_GRAN(self, no_self_edge, tril_mode, is_weighted=False, padded_size=None):
        _adj = self.adj(is_weighted)
        if no_self_edge:
            _adj = _adj - torch.diag(torch.diag(_adj))
        _init_feat = tril_and_pad(_adj, padded_size=padded_size, zero_rows=0, tril_mode=tril_mode)
        return _init_feat

    def get_PESE_node_feat(self, node_feat_types, config, is_undirected=True, is_weighted=False):
        if self.get('is_root', False):
            self.node_feat_PESE_names = []
            return self

        compute_posenc_stats(self, node_feat_types, is_undirected=is_undirected, cfg=config, is_weighted=is_weighted)
        for key_ in self.node_feat_PESE_names:
            self[key_] = self[key_].to(self.edge_index.device)
        return self

    def get_parent_node_ids(self, ):
        if hasattr(self, 'is_part') and self.is_part:
            parent_node_ids = [self.part_id] * self.num_nodes
        else:
            parent_node_ids = self.part_id_of_node
        return parent_node_ids

    def adj(self, is_weighted=False):
        """To return the Adjacency matrix of the graph"""
        if self.num_nodes == 0:
            return torch.tensor([[]], dtype=torch.float, device=self.edge_index.device)
        elif self.edge_index.numel() == 0:
            return torch.zeros([self.num_nodes, self.num_nodes], dtype=torch.float, device=self.edge_index.device)
        else:
            _adj = pyg.utils.to_dense_adj(edge_index=self.edge_index,
                                          edge_attr=self.edge_weight if is_weighted else None,
                                          max_num_nodes=self.num_nodes,
                                          )[0]
            return _adj.type(torch.float)

    def get_complement_edges(self, add_self_edges=False):
        _adj = self.adj()
        _adj2 = torch.ones_like(_adj) - _adj
        _adj2 -= torch.diag(torch.diag(_adj2))
        _adj2 = _adj2.to_sparse().coalesce()
        return _adj2.indices().long()


###########################################
# To make a modified Data structure that will be used for Bipartites in HiGen
def bipartite_subgraph(subset: Union[PairTensor, Tuple[List[int], List[int]]],
                       edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                       relabel_nodes: bool = False, size: Tuple[int,
                                                                int] = None,
                       return_edge_mask: bool = False):
    # the code is adopted from torch-geometric-2.0.4

    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset[0], (list, tuple)):
        subset = (torch.tensor(subset[0], dtype=torch.long, device=device),
                  torch.tensor(subset[1], dtype=torch.long, device=device))

    if subset[0].dtype == torch.bool or subset[0].dtype == torch.uint8:
        size = subset[0].size(0), subset[1].size(0)
    else:
        if size is None:
            size = (maybe_num_nodes(edge_index[0]),
                    maybe_num_nodes(edge_index[1]))
        subset = (index_to_mask(subset[0], size=size[0]),
                  index_to_mask(subset[1], size=size[1]))

    node_mask = subset
    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long, device=device)
        node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long, device=device)
        node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(), device=device)
        node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(), device=device)
        edge_index = torch.stack([node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


class Bipart(Data):

    def __init__(self, **kwargs):
        super(Bipart, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'edge_index_new', 'edge_index_aug']:
            if hasattr(self, 'part_left') and hasattr(self, 'part_right'):
                return torch.tensor([[self.part_left.num_nodes], [self.part_right.num_nodes]], device=self[key].device)
            else:
                return torch.tensor([[self.part_left_size], [self.part_right_size]], device=self[key].device)

        elif key == 'parent_edge_index':
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['parent_edge_index', 'edge_index_new', 'edge_index_aug', 'edge_index_orig']:
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def set_id(self, id):
        self.id = id

    def adj(self, is_weighted=False):
        _adj = pyg.utils.to_dense_adj(
            edge_index = self.edge_index,
            edge_attr = self.edge_weight if is_weighted else None,
            max_num_nodes = max(self.part_left.num_nodes, self.part_right.num_nodes)
        )[0]
        return _adj[:self.part_left.num_nodes, :self.part_right.num_nodes].type(torch.float)

    def get_edge_feat(self, edge_feat_dim, is_weighted, is_aug=False):
        self.edge_feat = torch.zeros(self.edge_index.shape[1], edge_feat_dim, device=self.edge_weight.device)
        if self.edge_index.numel == 0:
            return self.edge_feat

        if is_aug:
            self.edge_feat = self.edge_feat.scatter(
                1, self.edge_index[0, :].unsqueeze(1) + 1, self.edge_weight.unsqueeze(1))
            self.edge_feat = self.edge_feat.scatter(
                1, self.edge_index[1, :].unsqueeze(1) + 1 + int(edge_feat_dim/2), self.edge_weight.unsqueeze(1))
        else:
            self.edge_feat[:, 0] = self.edge_weight if is_weighted else 1.
            self.edge_feat[:, int(edge_feat_dim / 2)] = self.edge_weight if is_weighted else 1.

        return self.edge_feat

    def t(self):
        """
        :return: the transpose of the bipartite graph so that part_left and part_right are swapped
        """
        bp = Bipart(**self.to_dict())
        bp.edge_index = self.edge_index[[1, 0], ]
        if hasattr(self, 'edge_index_aug'):
            bp.edge_index_aug = self.edge_index_aug[[1, 0], ]
        bp.part_left, bp.part_right = self.part_right, self.part_left
        return bp



###########################################
# To make a Data structure for Hierarchical Graph in HiGen

class HG(list):
    def __init__(self, g_levels_list, max_level, id=None, ord=None):
        n_levels = len(g_levels_list)
        if max_level <= n_levels:
            g_levels_list = g_levels_list[-max_level:]
        else:
            ## to add None as the levels
            g_levels_list = [None] * (max_level - n_levels) + g_levels_list
        super(HG, self).__init__(g_levels_list)
        self.n_levels = n_levels
        self.ord = ord
        self.hg_name = id if id else ''
        self.num_leafnodes = g_levels_list[-1].num_nodes
        self.sum_leafedges = torch.tril(g_levels_list[-1].adj(is_weighted=True)).sum().long()
        self.num_leafedges = g_levels_list[-1].num_edges

    @classmethod
    def make_root_graph(cls, num_leafnodes, sum_leafedges, n_levels, hg_name, *args, **kwargs):
        x = torch.zeros([1, 32], dtype=torch.float)
        ## stats of the leaf graph are used as first 3 feature of root node
        x[0, 0:3] += torch.tensor([num_leafnodes, sum_leafedges, n_levels])
        return Data2(
            *args,
            **kwargs,
            is_root=True,
            graph_name=hg_name + '_root',
            id=hg_name,
            num_nodes=1,
            level=-1,
            n_levels=n_levels,
            num_leafnodes=num_leafnodes,
            sum_leafedges=torch.tensor([sum_leafedges.long()], dtype=torch.long),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_weight=torch.tensor([sum_leafedges.long()], dtype=torch.long),
            edge_feat=torch.zeros([1, 0]),
            x=x,
        )

    def set_id(self, id):
        self.hg_name = id
        for g_level in self:
            if not g_level is None:
                g_level.set_id(id)

            if hasattr(g_level, 'part_ls'):
                for _part in g_level.part_ls:
                    if _part is None:
                        continue
                    _part.set_id(id)
            if hasattr(g_level, 'child_part'):
                for _part in g_level.child_part:
                    if _part is None:
                        continue
                    _part.set_id(id)

            if hasattr(g_level, 'child_bipart'):
                for _bipart in g_level.child_bipart:
                    if _bipart is None:
                        continue
                    _bipart.set_id(id)
            if hasattr(g_level, 'biparts_ls'):
                for _bipart in g_level.biparts_ls:
                    if _bipart is None:
                        continue
                    _bipart.set_id(id)
        return

    def leaf_graph(self):
        return self[-1]

    def get_root_graph(self):
        self.root_graph = HG.make_root_graph(
            num_leafnodes=self.num_leafnodes,
            sum_leafedges=self.sum_leafedges,
            num_leafedges=self.num_leafedges,
            n_levels=self.n_levels,
            hg_name=self.hg_name
        )
        return self.root_graph

    def to_networkx(self, level='leaf', add_node_parts=False, add_child_cluster=False):
        _node_attrs = ["child_part", "part_id_of_node"] if add_node_parts else ["part_id_of_node"]

        g_leaf = self.leaf_graph()
        if not hasattr(g_leaf, "child_part"):
            g_leaf.child_part = [None]*g_leaf.num_nodes

        if add_child_cluster:
            _node_attrs += ['child_nodes']
            for l_ , g_ in enumerate(self):
                if l_ == len(self)-1:
                    g_.child_nodes = [None]*g_.num_nodes
                    continue
                child_nodes_dic = {}
                for part in g_.child_part:
                    child_nodes_dic.update({int(part.part_id): np.arange(part.num_nodes)})
                child_nodes_ls = [None]*len(child_nodes_dic)
                _start = 0
                for part_id in range(len(child_nodes_dic)):
                    child_nodes_  =  child_nodes_dic[part_id]
                    child_nodes_ls[part_id] = child_nodes_ + _start
                    _start += len(child_nodes_)
                g_.child_nodes = child_nodes_ls

        if level == 'all':
            _out = []
            for g_ in self:
                g_nx = pyg.utils.to_networkx(g_, node_attrs=_node_attrs, edge_attrs=["edge_weight"],
                                             to_undirected=True) if (not g_ is None) else None
                _out.append(g_nx)
            return _out

        elif level == 'leaf':
            return pyg.utils.to_networkx(g_leaf, node_attrs=_node_attrs, edge_attrs=["edge_weight"], to_undirected=True)

        else:
            g_ = self[level]
            return pyg.utils.to_networkx(g_, node_attrs=_node_attrs, edge_attrs=["edge_weight"],
                                         to_undirected=True) if (not g_ is None) else None

    def clone_abs(self, exclude_keys = ['biparts_ls', 'part_ls', 'child_part', 'child_bipart', 'parent_graph']):
        """ to return an abstract HG by removing some extra features. It's used to save memory"""
        abs_HG_list = []
        for i_level in range(self.n_levels):
            abs_HG_list.append(self[i_level].clone_abs(exclude_keys = exclude_keys))

        return HG(abs_HG_list, id=self.hg_name+'_abs', max_level=self.n_levels, ord=self.ord)


def collate_partitions(level, part_ls, bipart_ls=[None], exclude_keys=[], device=None, g_id=None,
                       compact=False, add_aug_edge=False, add_dense_edge=False, add_label=False,
                       add_edge_feat=True, node_feat_names=[]):
    """
    :param level:
    :param part_ls:
    :param bipart_ls:
    :param exclude_keys:
    :param device:
    :param g_id:
    :param compact:
    :param add_aug_edge:
    :param add_dense_edge:
    :param add_label:
    :param add_edge_feat:
    :param node_feat_names:
    :return: a union of partitions
    """

    ord = part_ls[0].ord
    part_id_list = []
    part_start = 0
    part_start_dict = {}
    part_id_of_node = []
    edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
    edge_weight = torch.tensor([], dtype=torch.long, device=device)
    edge_feat = torch.tensor([], dtype=torch.float, device=device)
    edge_index_aug = torch.tensor([[], []], dtype=torch.long, device=device)
    edge_weight_aug = torch.tensor([], dtype=torch.long, device=device)
    edge_index_dense= torch.tensor([[], []], dtype=torch.long, device=device)
    edge_weight_dense = torch.tensor([], dtype=torch.long, device=device)
    label = torch.tensor([], dtype=torch.long, device=device)
    num_aug_edges = []
    node_feat_dic = dict([(key_, torch.tensor([], dtype=torch.float, device=device)) for key_ in node_feat_names])
    child_part, child_bipart = [], []
    node_id_in_part = []
    for part_ in part_ls:
        assert level == part_.level
        assert ord == part_.ord

        if part_.part_id in part_id_list:
            continue

        # To add features of partition
        child_part += part_.get('child_part', [None] * part_.num_nodes) if not compact else []

        for key_ in node_feat_names:
            node_feat_dic[key_] = torch.cat([node_feat_dic[key_] , part_[key_]], dim=0)

        edge_index = torch.cat([edge_index, part_.edge_index + part_start], dim=1)
        edge_weight = torch.cat([edge_weight, part_.edge_weight], )
        if add_edge_feat:
            edge_feat = torch.cat([edge_feat, part_.edge_feat], dim=0)
        child_bipart += part_.get('child_bipart', [None] * part_.num_edges) if not compact else []

        if add_aug_edge and hasattr(part_, 'edge_index_aug'):
            edge_index_aug = torch.cat([edge_index_aug, part_.edge_index_aug + part_start], dim=1)
            num_aug_edges.append(part_.edge_index_aug.shape[1])
        if add_dense_edge and hasattr(part_, 'edge_index_dense'):
            edge_index_dense = torch.cat([edge_index_dense, part_.edge_index_dense + part_start], dim=1)
        if add_aug_edge and hasattr(part_, 'edge_weight_aug'):
            edge_weight_aug = torch.cat([edge_weight_aug, part_.edge_weight_aug])

        if add_label and hasattr(part_, 'label'):
            label = torch.cat([label, part_.label])

        # To add extra features to characterize partitions of current graph
        part_id_list.append(part_.part_id)
        part_start_dict.update({part_.part_id: part_start})
        part_start += part_.num_nodes
        part_id_of_node += [part_.part_id.item() if torch.is_tensor(part_.part_id) else part_.part_id] * part_.num_nodes
        node_id_in_part += list(np.arange(part_.num_nodes))

    num_nodes = part_start

    for bipart in bipart_ls:
        if (bipart is None):
            continue
        id_part_left = bipart.part_left.part_id
        id_part_right = bipart.part_right.part_id

        # to add symmetric edges
        edge_idx_bp = bipart.edge_index + \
                      torch.tensor([[part_start_dict[id_part_left]],
                                    [part_start_dict[id_part_right]]], dtype=torch.long, device=device)
        edge_index = torch.cat([edge_index, edge_idx_bp], dim=1)
        edge_weight = torch.cat([edge_weight, bipart.edge_weight], dim=0)
        if add_edge_feat:
            edge_feat = torch.cat([edge_feat, bipart.edge_feat], dim=0)
        child_bipart += bipart.get('child_bipart', [None] * part_.num_edges) if not compact else []

        if add_aug_edge and hasattr(bipart, 'edge_index_aug'):
            edge_idx_aug_bp = bipart.edge_index_aug + \
                              torch.tensor([[part_start_dict[id_part_left]],
                                            [part_start_dict[id_part_right]]], dtype=torch.long, device=device)
            edge_index_aug = torch.cat([edge_index_aug, edge_idx_aug_bp], dim=1)
            num_aug_edges.append(bipart.edge_index_aug.shape[1])

        if add_dense_edge and hasattr(bipart, 'edge_index_dense'):
            edge_idx_dense_bp = bipart.edge_index_dense + \
                              torch.tensor([[part_start_dict[id_part_left]],
                                            [part_start_dict[id_part_right]]], dtype=torch.long, device=device)
            edge_index_dense = torch.cat([edge_index_dense, edge_idx_dense_bp], dim=1)

        if add_aug_edge and hasattr(bipart, 'edge_weight_aug'):
            edge_weight_aug = torch.cat([edge_weight_aug, bipart.edge_weight_aug])
        if add_label and hasattr(bipart, 'label'):
            label = torch.cat([label, bipart.label])

    out_graph = Data2(
        level=level,
        ord=ord,
        id=g_id,
        num_nodes=num_nodes,
        is_part=False,
        child_part=child_part,
        part_id_of_node=part_id_of_node,
        node_id_in_part=node_id_in_part,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_feat=edge_feat,
        child_bipart=child_bipart,
        # partition and bipart levels
        part_id_ls=part_id_list,
        part_ls=part_ls if not compact else [],
        part_start_dict=part_start_dict,
        biparts_ls=bipart_ls if not compact else [],
        **node_feat_dic
    )
    if add_aug_edge:
        out_graph.edge_index_aug = edge_index_aug
        out_graph.num_aug_edges = torch.tensor(num_aug_edges, device=device)
        out_graph.edge_weight_aug = edge_weight_aug

    if add_dense_edge:
        out_graph.edge_index_dense = edge_index_dense
    if add_label:
        out_graph.label = label

    for key in exclude_keys:
        out_graph.__delattr__(key)
    return out_graph


## to load and save graphs
def save_load_graphs(graphs, tag, config, to_load=False):
    save_path = os.path.join(
        config.dataset.data_path, config.dataset.name,
        f'{config.model.name}_{tag}_{config.dataset.node_order}_precompute'
    )

    if not to_load or not os.path.isdir(save_path) or config.dataset.is_overwrite_precompute:
        file_names = []
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        config.dataset.save_path = save_path
        for index in tqdm(range(len(graphs))):
            Hyper_Graph_ls = graphs[index]
            tmp_path = os.path.join(save_path, f'{tag}_{index}.p')
            pickle.dump(Hyper_Graph_ls, open(tmp_path, 'wb'))
            file_names += [tmp_path]
    else:
        graphs = []
        file_names = glob.glob(os.path.join(save_path, '*.p'))
        for tmp_path in file_names:
            with open(tmp_path, "rb") as f:
                graphs.append(pickle.load(f))

    return file_names, graphs


##############

class PMF_stat_HG():
    def __init__(self, HG_ls):
        self.only_nonzero_val = True
        self.const = 1 if self.only_nonzero_val else 0
        n_canonical_order = len(HG_ls[0])
        self.max_num_levels = len(HG_ls[0][0])
        n_level_pmf = np.bincount([hg[i_ord].n_levels for hg in HG_ls \
                                     for i_ord in range(n_canonical_order)])
        self.num_level_pmf = n_level_pmf / n_level_pmf.sum()
        self.num_level_pmf = self.num_level_pmf[self.const:]

        self.num_leafnodes_given_level_pmf = []
        self.sum_leafedges_given_level_pmf = []
        self.num_leafedges_given_level_pmf = []
        for i_level in range(1, self.max_num_levels + 1):
            nodes_given_level_pmf = np.bincount(
                [hg[i_ord].num_leafnodes for hg in HG_ls for i_ord in range(n_canonical_order) if hg[i_ord].n_levels == i_level]
            )
            nodes_given_level_pmf = nodes_given_level_pmf/nodes_given_level_pmf.sum()
            self.num_leafnodes_given_level_pmf.append(nodes_given_level_pmf[self.const:])

            sum_edge_given_level_pmf = np.bincount(
                [hg[i_ord].sum_leafedges for hg in HG_ls for i_ord in range(n_canonical_order) if hg[i_ord].n_levels == i_level]
            )
            sum_edge_given_level_pmf = sum_edge_given_level_pmf / sum_edge_given_level_pmf.sum()
            self.sum_leafedges_given_level_pmf.append(sum_edge_given_level_pmf[self.const:])

            num_edge_given_level_pmf = np.bincount(
                [hg[i_ord].num_leafedges for hg in HG_ls for i_ord in range(n_canonical_order) if hg[i_ord].n_levels == i_level]
            )
            num_edge_given_level_pmf = num_edge_given_level_pmf / num_edge_given_level_pmf.sum()
            self.num_leafedges_given_level_pmf.append(num_edge_given_level_pmf[self.const:])

        # to compute the marginal pmf of num_nodes per level
        self.num_nodes_pmf_train = []
        self.max_num_nodes = []
        self.num_nodes_parts_pmf_train = []
        self.max_num_nodes_part = []
        for i_level in range(self.max_num_levels):
            self.num_nodes_pmf_train += [np.bincount([hg[i_ord][i_level].num_nodes for hg in HG_ls \
                                                      for i_ord in range(n_canonical_order) if not hg[i_ord][i_level] is None])]
            self.max_num_nodes += [len(self.num_nodes_pmf_train[i_level])]

            self.num_nodes_parts_pmf_train += [
                np.bincount([part.num_nodes for hg in HG_ls for i_ord in range(n_canonical_order) if not hg[i_ord][i_level] is None
                     for part in hg[i_ord][i_level].part_ls])
            ]
            self.max_num_nodes_part += [len(self.num_nodes_parts_pmf_train[i_level])]

        self.num_nodes_pmf_train = [(self.num_nodes_pmf_train[l] / self.num_nodes_pmf_train[l].sum())[self.const:]
                                    for l in range(self.max_num_levels)]
        self.num_nodes_parts_pmf_train = [(self.num_nodes_parts_pmf_train[l] / self.num_nodes_parts_pmf_train[l].sum())[self.const:]
                                          for l in range(self.max_num_levels)]

        self.graph_stat_list = [(hg[i_ord].num_leafnodes, hg[i_ord].sum_leafedges, hg[i_ord].num_leafedges, hg[i_ord].n_levels )
                         for hg in HG_ls for i_ord in range(n_canonical_order)]

    def _multinomial_gen(self, pmf, n_sample):
        return torch.multinomial(torch.from_numpy(pmf), num_samples=n_sample, replacement=True) + self.const

    def sample(self, n_sample, leaf_edge_weight=1):
        out_samples_ls = []
        num_level_ls = self._multinomial_gen(self.num_level_pmf, n_sample)
        bin_count_levels = np.bincount(num_level_ls)[self.const:]  # bin count of levels
        for i, n_sample_level in enumerate(bin_count_levels):
            if n_sample_level == 0:
                continue
            n_level = i + self.const
            sum_leafedges_samples = self._multinomial_gen(self.sum_leafedges_given_level_pmf[i], n_sample_level)
            if leaf_edge_weight == 1:
                num_leafedges_samples = sum_leafedges_samples
            else:
                num_leafedges_samples = self._multinomial_gen(self.num_leafedges_given_level_pmf[i], n_sample_level)

            num_nodes_samples = self._multinomial_gen(self.num_leafnodes_given_level_pmf[i], n_sample_level)
            out_samples_ls += [(n_level, n_node_, sum_edge_, num_edge_)for n_node_, sum_edge_, num_edge_ in
                               zip(num_nodes_samples, sum_leafedges_samples, num_leafedges_samples)]
        return out_samples_ls

    def sample2(self, n_sample, leaf_edge_weight=1):
        sample_indices = np.random.randint(low=0, high=len(self.graph_stat_list), size=n_sample)
        graph_stat_samples = [self.graph_stat_list[ind_] for ind_ in sample_indices]
        out_samples = [(n_level_, n_node_, sum_edge_, num_edge_)
                        for n_node_, sum_edge_, num_edge_, n_level_ in graph_stat_samples]
        return out_samples

    def sample_num_nodes_parts(self, n_sample, level):
        return self._multinomial_gen(self.num_nodes_parts_pmf_train[level], n_sample).cpu().numpy()

    def compute_stat_of_degree(self, HG_ls):
        num_canonical_order = len(HG_ls[0])

        self.degree_parts_pmf_train = []
        self.min_degree_parts_pmf_train = []
        self.max_degree_parts_pmf_train = []
        for i_level in range(self.max_num_levels):
            _degree_parts_pmf_train = np.bincount(
                torch.cat([part.degree() for hg in HG_ls
                           for i_ord in range(num_canonical_order) if not hg[i_ord][i_level] is None
                           for part in hg[i_ord][i_level].part_ls
                           ]).cpu().long().numpy())
            self.degree_parts_pmf_train.append(_degree_parts_pmf_train)

            _min_degree_parts_pmf_train = np.bincount(
                [part.degree().min().long().item() for hg in HG_ls
                 for i_ord in range(num_canonical_order) if not hg[i_ord][i_level] is None
                 for part in hg[i_ord][i_level].part_ls])
            self.min_degree_parts_pmf_train.append(_min_degree_parts_pmf_train)

            _max_degree_parts_pmf_train = np.bincount(
                [part.degree().max().long().item() for hg in HG_ls
                 for i_ord in range(num_canonical_order) if not hg[i_ord][i_level] is None
                 for part in hg[i_ord][i_level].part_ls])
            self.max_degree_parts_pmf_train.append(_max_degree_parts_pmf_train)

        self.degree_parts_pmf_train = [(self.degree_parts_pmf_train[l] / self.degree_parts_pmf_train[l].sum())[self.const:]
                                       for l in range(self.max_num_levels)]

        self.min_degree_parts_pmf_train = [(self.min_degree_parts_pmf_train[l] / self.min_degree_parts_pmf_train[l].sum())[self.const:]
                                           for l in range(self.max_num_levels)]

        self.max_degree_parts_pmf_train = [(self.max_degree_parts_pmf_train[l] / self.max_degree_parts_pmf_train[l].sum())[self.const:]
                                           for l in range(self.max_num_levels)]

    def sample_min_max_degree_part(self, level, n_sample=1):
        _min_degree_part = self._multinomial_gen(self.min_degree_parts_pmf_train[level], n_sample)
        _max_degree_part = self._multinomial_gen(self.max_degree_parts_pmf_train[level], n_sample)
        if n_sample == 1:
            return (_min_degree_part.item(), _max_degree_part.item())
        else:
            return (_min_degree_part.cpu().numpy(), _max_degree_part.cpu().numpy())

