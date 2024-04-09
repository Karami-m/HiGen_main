from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Batch
from utils.graph_helper import *
from utils.higen_data import *
from utils.nn import *
from utils.loss import *
from utils.gnn_model import Node_Pre_Embedding, GATgran_module, GNN_Module

EPS = 10 * np.finfo(np.float32).eps


class Graph_Gen_MultiLevels(nn.Module):
    """To generate a hierarchichal graphs with multiple levels of hierarchy.
    """

    def __init__(self, config):
        super(Graph_Gen_MultiLevels, self).__init__()
        self.config = config
        self.logger = config.logger
        self.stat_HG_pmf = self.config.model.stat_HG_pmf
        self.device = config.device
        self.num_levels = config.model.num_levels
        self.hidden_dim = config.model.hidden_dim
        self.max_num_nodes_all = config.dataset.max_num_nodes_all
        self._verbose = config.get('verbose', 1)

        self.node_pre_emb_module = Node_Pre_Embedding(
            dimension_reduce=config.model.dimension_reduce,
            embedding_dim=config.model.embedding_dim,
            inp_dim=self.max_num_nodes_all,
        )
        self.add_module("node_pre_emb_module", self.node_pre_emb_module)

        model_list = []

        _context_parent_mdl = None
        _gnn_aug_part_mdl, _gnn_aug_bipart_mdl = None, None
        _gnn_aug_part = config.model.get('gnn_aug_part', 'GATgran')
        _gnn_aug_bipart = config.model.get('gnn_aug_bipart', 'GATgran')
        _context_parent = config.model.context_parent.split('_')[0]
        _context_part = config.model.context_part.split('_')[0]
        for l in range(self.num_levels):
            if (_context_parent_mdl is None) or ('indv' in config.model.context_parent):
                _context_parent_mdl = GNN_Module(
                    config,
                    type=_context_parent,
                ) if not _context_parent in ['aug', 'augBP'] else None

            if config.model.context_parent == 'aug':
                _context_parent_mdl = model_list[-1].get_context_aug if l > 0 else None
            if config.model.context_parent == 'augBP':
                _context_parent_mdl = model_list[-1].get_context_aug if l > 0 else None

            if (_gnn_aug_part_mdl is None) or (not 'shared' in _gnn_aug_part):
                pre_emb_layer = self.node_pre_emb_module if config.model.get('shrd_premb_part', True) \
                    else None
                _gnn_aug_part_mdl = GNN_Module(config, type=_gnn_aug_part.split('_')[0],
                                               pre_emb_layer=pre_emb_layer, is_augmented=True)
            if (_gnn_aug_bipart_mdl is None) or (not 'shared' in _gnn_aug_bipart):
                _gnn_aug_bipart_mdl = GNN_Module(config, type=_gnn_aug_bipart.split('_')[0], is_augmented=True)

            higen_model_ = Conditional_Graph_Gen(
                config, level=l,
                node_pre_emb_module=self.node_pre_emb_module,
                context_parent_mdl=_context_parent_mdl,
                gnn_aug_part=_gnn_aug_part_mdl,
                gnn_aug_bipart=_gnn_aug_bipart_mdl,
            )

            model_list.append(higen_model_)
        self.graph_gen_levels = nn.ModuleList(model_list)

    def forward(self, inp_ls):
        """
        :param inp_ls:
        :return: the loss of the generative model on HG in all levels
        """

        loss_ls = []
        loss_pbp_ls = []
        for l in range(self.num_levels):
            out = edict()
            g_batch, bp_batch, parent_batch = inp_ls[l]
            loss_, loss_p_, loss_bp_ = self.graph_gen_levels[l].forward(g_batch, bp_batch, parent_batch)
            loss_ls += [loss_] if not (loss_ is None) else []
            loss_pbp_ls.append("l={}: ({:.3f}, {:.3f})".format(l, loss_p_.item(), loss_bp_.item()) if not (loss_ is None)
                               else "(inf , inf )")

        if self._verbose > 1:
            self.logger.info(f"Loss details (Part, Bipart): {loss_pbp_ls}")
        return torch.stack(loss_ls, dim=0).mean(dim=0)

    def generate(self, g_root_batch):
        """
        :param g_root_batch:
        :return: Generated hierachical graph sample from the trained distribution
        """

        hg_batch = []
        for g_root in g_root_batch:
            hg_ls = []
            g_parent = g_root
            grandparent_graph_dict = None
            for l in range(self.num_levels - g_root.n_levels, self.num_levels):
                ## for distribution 'mix_Bernouli', we might need the following num_nodes
                g_parent.max_num_nodes_in_child_part = self.stat_HG_pmf.sample_num_nodes_parts(g_parent.num_nodes,
                                                                                               level=l)

                g_l = self.graph_gen_levels[l].generate(g_parent)
                g_l.parent_graph_name = g_parent.graph_name
                hg_ls.append(g_l)
                g_parent = g_l

            hg_batch.append(HG(hg_ls, max_level=self.num_levels))
        return hg_batch


class Conditional_Graph_Gen(nn.Module):
    """Conditional generative probability to generate the graph at the next level given its parent level graph.
    """

    def __init__(self, config, level, node_pre_emb_module, context_parent_mdl,
                 gnn_aug_part, gnn_aug_bipart):
        super(Conditional_Graph_Gen, self).__init__()
        self.config = config
        self.device = config.device
        self.level = level
        self.has_edge_weight = config.model.has_edge_weight
        self.is_leaf_level = self.level == config.dataset.num_levels - 1
        self.no_self_edge = config.dataset.no_self_edge == 'all' or (
                config.dataset.no_self_edge.lower() == 'last' and self.is_leaf_level)
        self.max_num_nodes = config.model.max_num_nodes[self.level]
        self.max_num_nodes_augbp = np.array(config.model.max_num_nodes[self.level])
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = 1
        self.num_mix_component = config.model.num_mix_component
        self.dist_part = self.dist_BP = config.model.dist.split('+')[0]
        if self.is_leaf_level:
            if '+Bern' in config.model.dist:
                self.dist_part = self.dist_BP = 'mix_Bernouli'
            if '+PartBern' in config.model.dist:
                self.dist_part = 'mix_Bernouli'
            if '+BPBern' in config.model.dist:
                self.dist_BP = 'mix_Bernouli'

        self.embedding_dim = node_pre_emb_module.embedding_dim if config.model.dimension_reduce else self.max_num_nodes

        self.gnn_aug_part = gnn_aug_part if not (gnn_aug_part is None) else \
            GATgran_module(config, type='GATgran')
        self.gnn_aug_bipart = gnn_aug_bipart
        self.gnn_parent_mdl = context_parent_mdl

        self.link_pred_part = Edge_Link_Prediction(config, level=self.level, is_part=True)
        self.add_module("link_pred_part", self.link_pred_part)
        if self.level > 0:
            self.link_pred_bp = Edge_Link_Prediction(config, level=self.level, is_part=False)
            self.add_module("link_pred_bp", self.link_pred_bp)

    def forward(self, g_batch: Batch, bp_batch: Batch, parent_batch: Batch = None):

        if (g_batch is None) and (bp_batch is None):
            return (None, None, None)

        # To get parent node futures
        if self.level > 0:
            parent_batch = self.gnn_parent_mdl(parent_batch, G_parent=None, parent_features=None)

        # To get augmented parts features and loss
        ## add parent node feature + PE/SE feature
        g_batch = self.gnn_aug_part(g_batch, G_parent=parent_batch, parent_features=None,
                                    is_aug_part=True)  # parent_features=['x'])
        loss_adj_g = self.link_pred_part.get_loss_gen_part(g_batch, parent_graph=parent_batch)

        # To get augmented biparts features and loss
        ## add parent node feature + PE/SE feature
        loss_adj_bp = torch.tensor(0., device=self.device)
        if not bp_batch is None:
            if "BiasedTransformer" in self.config.gt.layer_type:
                # BiasedTransformer requires `batch.attn_bias`.
                bp_batch.attn_bias = pyg.utils.to_dense_adj(
                    torch.cat([bp_batch.edge_index, bp_batch.edge_index_dense, bp_batch.edge_index_aug], dim=1),
                    batch=bp_batch.batch)
                mask_ = torch.eye(bp_batch.attn_bias.shape[1]).repeat(bp_batch.attn_bias.shape[0], 1, 1).bool()
                bp_batch.attn_bias[mask_] = 1.
                # bp_batch.attn_bias += torch.eye(bp_batch.attn_bias.shape[1]).unsqueeze(0)
                bp_batch.attn_bias = ~bp_batch.attn_bias.type(torch.bool)

            bp_batch = self.gnn_aug_bipart(bp_batch, G_parent=parent_batch, parent_features=['x'])
            loss_adj_bp = self.link_pred_bp.get_loss_gen_bipart(
                bp_batch,
                parent_graph=parent_batch,
                part_graph_dict=None)

            del bp_batch.attn_bias

        return (loss_adj_g + loss_adj_bp, loss_adj_g, loss_adj_bp)

    def generate(self, G_parent: Data2):
        """
        :param G_parent:
        :return: conditioned on the graph in parent level, generate graph in next leve
        """

        with torch.no_grad():
            # To get parent node futures
            if self.level > 0:
                G_parent = G_parent.get_PESE_node_feat(self.config.model.node_feat_types,
                                                       config=self.config)  # .to(self.device)
                G_parent.batch = torch.zeros([G_parent.num_nodes], device=G_parent.edge_index.device,
                                             dtype=G_parent.edge_index.dtype)
                G_parent.edge_index_batch = torch.zeros([G_parent.num_edges], device=G_parent.edge_index.device,
                                                        dtype=G_parent.edge_index.dtype)
                G_parent.ptr = torch.tensor([0, G_parent.num_nodes], device=self.device)
                G_parent.num_graphs = 1
                G_parent = self.gnn_parent_mdl(G_parent, G_parent=None, parent_features=None)

            # To Generate partition graphs of all parent nodes
            G_parent.child_part = [
                Data2(edge_index=torch.tensor([[], []], dtype=torch.long),
                      edge_weight=torch.tensor([], dtype=torch.long),
                      num_nodes=0, is_completed=False)
                for j_ in range(G_parent.num_nodes)
            ]

            for jj in range(self.max_num_nodes):
                ## it generates the partitions in AR fashion
                g_batch_set = []
                for part_id, g_aug in enumerate(G_parent.child_part):
                    if hasattr(g_aug, 'is_completed') and g_aug.is_completed:  # for early termination of generation
                        continue
                    g_aug = get_augmented_graph(
                        to_train=False,
                        adj_full_part=g_aug.adj(is_weighted=self.has_edge_weight),
                        parent_graph=G_parent,
                        node_id_in_part=jj,
                        part_start=0,
                        part_id_of_node=part_id,
                        no_self_edge=self.no_self_edge,
                        ord_=0, l_=self.level,
                        edge_feat_dim=self.config.model.edge_feat_dim,
                        max_num_nodes=self.max_num_nodes,
                        config=self.config,
                        child_bipart=[],
                        child_part=[],
                        is_completed=g_aug.is_completed,
                        bfs_feat=None,
                        sum_leafedges=G_parent.sum_leafedges,
                        num_leafedges=G_parent.num_leafedges,
                    )
                    if g_aug.edge_index_aug.numel() == 0 and self.no_self_edge:
                        continue
                    if 'mix_Bernouli' in self.dist_part:
                        # max_num_nodes_in_child_part is sampled each time, it is used to stop generation.
                        # We don't need it in multinomial case that has a mechanism for stopping the generation
                        g_aug.max_num_nodes = G_parent.max_num_nodes_in_child_part[part_id]
                    g_aug.to(self.device)
                    g_batch_set.append(g_aug)

                if len(g_batch_set) == 0:
                    continue

                g_batch = HigenData.batch_from_list_parts(
                    g_batch_set,
                    exclude_keys=['edge_attr_keys_extra', 'node_attr_keys_extra', 'child_bipart', 'child_part',
                                  'node_feat_PESE_names']
                ).to(self.device)

                # get augmented parts features and loss
                g_batch = self.gnn_aug_part(g_batch, G_parent=G_parent, parent_features=None, is_aug_part=True)
                g_batch_set = self.link_pred_part.get_loss_gen_part(g_batch, parent_graph=G_parent, to_gen=True)

                for g_aug in g_batch_set:
                    part_id = g_aug.part_id
                    del (g_aug.edge_weight_aug, g_aug.edge_index_aug)
                    G_parent.child_part[part_id] = g_aug

            ## to remove isolated nodes
            if 'mix_Bernouli' in self.dist_part and not 'Connected' in self.link_pred_part.gen_completion_part:
                for part_ in G_parent.child_part:
                    del (part_.x)  # part_.edge_feat,
                    part_.edge_index, part_.edge_weight, mask = pyg.utils.remove_isolated_nodes(
                        part_.edge_index, part_.edge_weight, num_nodes=part_.num_nodes)
                    if mask.sum().item() == 0:  # it should have one node at least
                        mask[0] = 1
                    part_.num_nodes = mask.sum().item()
                    part_.parent_node_id = part_.parent_node_id[mask]
                    part_.part_id_of_node = part_.part_id_of_node[mask]

            # BiPart generation
            G_parent.child_bipart = [None] * G_parent.num_edges
            bipart_ls = []
            visited_edges = {}
            for bp_id in range(G_parent.num_edges):
                edge_ = (G_parent.edge_index[0, bp_id].item(), G_parent.edge_index[1, bp_id].item())
                if (edge_[0] == edge_[1]):
                    continue
                if (edge_[1], edge_[0]) in visited_edges:
                    bipart_ = visited_edges[(edge_[1], edge_[0])].t()
                    bipart_ls.append(bipart_)
                    G_parent.child_bipart[bp_id] = bipart_
                    continue

                bipart_ = Bipart(
                    edge_index=torch.tensor([[], []], dtype=torch.long, device=self.device),
                    edge_weight=torch.tensor([], dtype=torch.long, device=self.device),
                    edge_feat=torch.zeros([0, self.config.model.edge_feat_dim], device=self.device),  # todo: not used
                    part_left=G_parent.child_part[G_parent.edge_index[0, bp_id]],
                    part_right=G_parent.child_part[G_parent.edge_index[1, bp_id]],
                    child_bipart=[],
                    level=self.level, ord=0,
                ).to(self.device)
                visited_edges.update({edge_: bipart_})
                bipart_ls.append(bipart_)
                G_parent.child_bipart[bp_id] = bipart_
            G_parent.bipart_ls = bipart_ls

            if G_parent.num_nodes == 1:
                g_level = collate_partitions(
                    level=self.level,
                    part_ls=G_parent.child_part,
                    bipart_ls=G_parent.bipart_ls,
                    device=self.device,
                    add_edge_feat=False
                )

            elif G_parent.num_nodes > 1:
                if self.config.model.to_joint_aug_bp.lower() == 'all':
                    bipart_aug = get_augmented_all_biparts(
                        graph_parent=G_parent,
                        to_joint_aug_bp=self.config.model.to_joint_aug_bp,
                        ord=0, level=self.level,
                        to_train=False,
                        edge_feat_dim=self.config.model.edge_feat_dim,
                        sum_leafedges=G_parent.sum_leafedges,
                        num_leafedges=G_parent.num_leafedges,
                        config=self.config,
                    )

                    bipart_aug.batch = torch.zeros([bipart_aug.num_nodes],
                                                   device=G_parent.edge_index.device, dtype=G_parent.edge_index.dtype)
                    bipart_aug.num_aug_edges_batch = torch.zeros_like(bipart_aug.num_aug_edges)
                    bipart_aug.num_graphs = 1
                    bipart_aug.ptr = torch.tensor([0, bipart_aug.num_nodes], device=self.device)
                    bipart_aug.edge_index_batch = torch.zeros([bipart_aug.num_edges],
                                                              device=bipart_aug.edge_index.device,
                                                              dtype=bipart_aug.edge_index.dtype)
                    bipart_aug.edge_index_aug_batch = torch.zeros([bipart_aug.edge_index_aug.shape[1]],
                                                                  device=bipart_aug.edge_index_aug.device,
                                                                  dtype=bipart_aug.edge_index_aug.dtype)

                    bipart_aug = self.gnn_aug_bipart(bipart_aug, G_parent=G_parent, parent_features=['x'])
                    if bipart_aug.num_aug_edges.sum() > 0:
                        bipart_aug = self.link_pred_bp.get_loss_gen_bipart(
                            bipart_aug,
                            to_gen=True,
                            parent_graph=G_parent,
                            part_graph_dict=None,
                        ).to(self.device)

                    g_level = bipart_aug
                    g_level.part_id_of_node = g_level.parent_node_id

                elif self.config.model.to_joint_aug_bp.lower() == 'ar':
                    visited_edges = set()
                    _, ord_ = pyg.utils.sort_edge_index(
                        G_parent.edge_index,
                        edge_attr=torch.arange(G_parent.num_edges, device=self.device)
                    )
                    bipart_ls = []
                    for bp_id in ord_:
                        if G_parent.edge_index[0, bp_id] <= G_parent.edge_index[1, bp_id]:
                            continue

                        edge_ = (G_parent.edge_index[0, bp_id].item(), G_parent.edge_index[1, bp_id].item())
                        if (edge_[0] == edge_[1]) or (edge_[1], edge_[0]) in visited_edges:
                            continue
                        visited_edges.add(edge_)

                        bipart_aug = get_augmented_AR_biparts(
                            bipart_sample=G_parent.child_bipart[bp_id],
                            graph_parent=G_parent,
                            to_joint_aug_bp=self.config.model.to_joint_aug_bp,
                            ord=0, level=self.level,
                            to_train=False,
                            edge_feat_dim=self.config.model.edge_feat_dim,
                            sum_leafedges=G_parent.sum_leafedges,
                            num_leafedges=G_parent.num_leafedges,
                            parent_edge_id=bp_id,
                            config=self.config,
                        )

                        bipart_aug.batch = torch.zeros([bipart_aug.num_nodes], device=G_parent.edge_index.device,
                                                       dtype=G_parent.edge_index.dtype)
                        bipart_aug.num_aug_edges_batch = torch.zeros_like(bipart_aug.num_aug_edges)
                        bipart_aug.num_graphs = 1
                        bipart_aug.ptr = torch.tensor([0, bipart_aug.num_nodes], device=self.device)
                        bipart_aug.edge_index_batch = torch.zeros([bipart_aug.num_edges],
                                                                  device=bipart_aug.edge_index.device,
                                                                  dtype=bipart_aug.edge_index.dtype)
                        bipart_aug.edge_index_aug_batch = torch.zeros([bipart_aug.edge_index_aug.shape[1]],
                                                                      device=bipart_aug.edge_index_aug.device,
                                                                      dtype=bipart_aug.edge_index_aug.dtype)
                        bipart_aug = self.gnn_aug_bipart(bipart_aug, G_parent=G_parent, parent_features=['x'])
                        if bipart_aug.num_aug_edges.sum() > 0:
                            bipart_aug = self.link_pred_bp.get_loss_gen_bipart(
                                bipart_aug,
                                to_gen=True,
                                parent_graph=G_parent,
                                part_graph_dict=None,
                            ).to(self.device)

                        G_parent.child_bipart[bp_id].edge_index = bipart_aug.edge_index_bp
                        G_parent.child_bipart[bp_id].edge_weight = bipart_aug.edge_weight_bp

                        bp_t_id = torch.where(
                            (G_parent.edge_index.t() == torch.flip(G_parent.edge_index[:, bp_id], dims=(0,))).all(
                                dim=1))[0][0]
                        G_parent.child_bipart[bp_t_id].edge_index = torch.flip(bipart_aug.edge_index_bp, dims=(0,))
                        G_parent.child_bipart[bp_t_id].edge_weight = bipart_aug.edge_weight_bp

                    g_level = collate_partitions(level=self.level,
                                                 part_ls=G_parent.child_part,
                                                 bipart_ls=[child_bp_ for child_bp_ in G_parent.child_bipart if
                                                            not child_bp_ is None],
                                                 device=self.device, add_edge_feat=False)
                    # g_level = bipart_aug

        g_level.graph_name = f"O{0}_L{self.level}"
        g_level.sum_leafedges = G_parent.sum_leafedges
        g_level.num_leafedges = G_parent.num_leafedges
        del (g_aug.edge_weight_aug, g_aug.edge_index_aug)
        return g_level


class Edge_Link_Prediction(nn.Module):
    """To compute the loss or generate the links (augmented edges) of the partitions and bipartites.
    """

    def __init__(self, config, level, is_part):
        super(Edge_Link_Prediction, self).__init__()
        self.config = config
        self.level = level
        self.is_leaf_level = self.level == config.dataset.num_levels - 1
        self.is_part = is_part
        self.dist = config.model.dist.split('+')[0]
        self._verbose = config.get('verbose', 1)
        if self.is_leaf_level:
            if '+Bern' in config.model.dist:
                self.dist = 'mix_Bernouli'
            if is_part and '+PartBern' in config.model.dist:
                self.dist = 'mix_Bernouli'
            if not is_part and '+BPBern' in config.model.dist:
                self.dist = 'mix_Bernouli'
        self.has_edge_weight = config.model.has_edge_weight
        self.NLL_avg = config.model.get('NLL_avg', 'adv')
        self.no_self_edge = config.dataset.no_self_edge == 'all' or (
                config.dataset.no_self_edge.lower() == 'last' and self.level == config.dataset.num_levels - 1)
        self.to_model_selfedge = False if self.no_self_edge else config.model.get('model_selfedge', True)
        ## is_connected_graph: is used to ensure connecetd graph generation.
        ## if is_connected_graph==1 then an edge in each cutset is reserved so that cutset_weight >= 1
        self.is_connected_graph = int(config.model.get("is_connected_graph", 1))
        self.postMixBP = config.model.get('postMixBP', False)

        self.num_mix_component = config.model.num_mix_component if 'mix' in self.dist else 1
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = 1
        self.dropout = config.model.get('dropout', 0.)
        self.batch_size_tr = config.train.batch_size
        self.batch_size_ts = config.test.batch_size

        _gen_completion = self.config.model.gen_completion.split('_')
        self.gen_completion_part = _gen_completion[0]
        self.gen_completion_bipart = _gen_completion[1] if len(_gen_completion) > 1 else None
        _gen_sampling = _gen_completion[2] if len(_gen_completion) > 2 else 'mode'
        self._probe_mode = 'mode' in _gen_sampling.lower()
        if self._probe_mode:
            self._n_sample_best = 100
        else:
            self._n_sample_best = 10 if '10' in _gen_sampling else 1

        if self.dist == 'mix_Bernouli':
            ### BCE Loss functions for mix_Bernouli
            self.BCE_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]), reduction='none')

            self.model_theta = Output_Model(
                hidden_dim=self.hidden_dim,
                out_dim=self.output_dim * self.num_mix_component,
                n_layers=3,
                gate=nn.ReLU(),
                model_type='split',
                device=self.config.device,
            )

            self.model_alpha = Output_Model(
                hidden_dim=self.hidden_dim,
                out_dim=self.num_mix_component,
                n_layers=3,
                gate=nn.ReLU(),
                model_type='simple',
                device=self.config.device,
            )

            self.add_module("output_theta", self.model_theta)
            self.add_module("output_alpha", self.model_alpha)

        elif self.dist == 'mix_multinomial':
            self.model_linkpred = Link_Prediction_Model(config, level)
            self.add_module("model_linkpred", self.model_linkpred)

        self.link_attr_type = config.model.link_attr_type
        self.link_cntx_type = config.model.link_cntx_type
        self.link_cntx_type_bp = config.model.link_cntx_type_bp

    def get_nll_avg_level(self, nll_smpl_bch, g_batch, part_bipart, post_mixture=False):
        num_parts_inlevel = g_batch.num_parts_inlevel if part_bipart == 'part' else \
            g_batch.num_biparts_inlevel * .5
        num_aug_edges = g_batch.num_aug_edges
        if self.to_model_selfedge and part_bipart == 'part':  # in this case we should count the self edge as well, so +1
            num_aug_edges += 1

        if part_bipart == 'bipart':
            num_biparts_in_level = num_parts_inlevel

            if self.NLL_avg == 'adv':
                if not post_mixture:
                    nll_smpl_bch = pyg.nn.global_add_pool(nll_smpl_bch.unsqueeze(1),
                                                          batch=g_batch.num_aug_edges_batch).squeeze()
                nll_smpl_bch = nll_smpl_bch / 2

                nll_smpl_bch = nll_smpl_bch / (EPS + g_batch.num_leafedges)

                if self.config.model.to_joint_aug_bp.lower() == 'ar':
                    nll_smpl_bch *= num_biparts_in_level / g_batch.num_samples

                NLL_bch_avgbp = torch.zeros(self.batch_size_tr, device=nll_smpl_bch.device,
                                            dtype=nll_smpl_bch.dtype).scatter_add(
                    0, index=g_batch.bch_hg, src=nll_smpl_bch)
                loss_adj_avg = NLL_bch_avgbp.sum() / (EPS + torch.count_nonzero(NLL_bch_avgbp))

        elif (self.NLL_avg == 'adv') and part_bipart == 'part':
            NLL_bch_smplsubg = nll_smpl_bch * g_batch.num_nodes_of_part * num_parts_inlevel / g_batch.num_leafedges

            num_smpl_per_hgbach = torch.zeros(
                self.batch_size_tr, device=g_batch.part_id.device, dtype=g_batch.part_id.dtype
            ).scatter_add(0, index=g_batch.bch_hg, src=torch.ones_like(g_batch.bch_hg))
            split_lengths = list(num_smpl_per_hgbach.cpu().numpy())

            NLL_bch_smplsubg = split_then_pad(
                NLL_bch_smplsubg,
                split_lengths=split_lengths,
                value=.0).t().type_as(NLL_bch_smplsubg)

            part_id = split_then_pad(
                g_batch.part_id,
                split_lengths=split_lengths,
                value=0).t().type_as(g_batch.part_id)

            mask = split_then_pad(
                torch.ones_like(g_batch.part_id),
                split_lengths=split_lengths,
                value=.0).t().type_as(g_batch.part_id)

            num_parts_max = torch.max(g_batch.part_id).item() + 1
            num_smpl_perpart = torch.zeros(
                [self.batch_size_tr, num_parts_max],
                device=g_batch.part_id.device, dtype=g_batch.part_id.dtype
            ).scatter_add(1, part_id, src=mask)

            ## Now we get average over all samples in each part
            NLL_bch_smplparts = torch.zeros(
                [self.batch_size_tr, num_parts_max], device=NLL_bch_smplsubg.device, dtype=NLL_bch_smplsubg.dtype
            ).scatter_add(1, index=part_id, src=NLL_bch_smplsubg) / (EPS + num_smpl_perpart)

            ## Now we get average over all parts
            NLL_bch_avgbp = NLL_bch_smplparts.sum(dim=1) / (EPS + torch.count_nonzero(NLL_bch_smplparts, dim=1))
            ## Now we get average over batches
            loss_adj_avg = NLL_bch_avgbp.sum() / (EPS + torch.count_nonzero(NLL_bch_avgbp))

        loss_adj_avg = torch.nan_to_num(loss_adj_avg)
        return loss_adj_avg

    def get_parent_context_of_node(self, g_batch, parent_graph):
        parent_node_id_augEdge_ = g_batch.parent_node_id[g_batch.edge_index_aug[0, :]]
        h_parent_per_augEdge = parent_graph.x[parent_node_id_augEdge_, :]

        stat_per_augEdge = torch.stack(
            [g_batch.num_nodes_aug_graph[g_batch.edge_index_aug_batch],
             g_batch.remaining_weight[g_batch.edge_index_aug_batch],
             g_batch.all_weight[g_batch.edge_index_aug_batch]], dim=1)
        stat2_per_augEdge = torch.unsqueeze(g_batch.cutset_weight[g_batch.edge_index_aug_batch], dim=1)

        h_parent_per_augGraph = parent_graph.x[g_batch.part_id, :]
        context_augGraph = torch.stack([g_batch.remaining_weight, g_batch.all_weight, g_batch.num_nodes_aug_graph],
                                       dim=1)
        h_parent_per_augGraph = torch.cat([h_parent_per_augGraph, context_augGraph], dim=1)

        return h_parent_per_augEdge, stat_per_augEdge, stat2_per_augEdge, h_parent_per_augGraph

    def get_aug_part_attributes(self, g_batch, h_aug_edges, parent_graph):

        # Get the context of the parent node
        h_parent_per_augEdge, stat_per_augEdge, stat2_per_augEdge, h_parent_per_augGraph = self.get_parent_context_of_node(
            g_batch=g_batch,
            parent_graph=parent_graph,
        )

        link_attr_bn_level = 'edge'
        alpha_level = 'graph'

        link_aug_cntx = None
        is_logp_link_attr_bn = False
        if self.link_cntx_type.lower() == 'none':
            link_aug_attr = graph_aug_attr = h_aug_edges
        elif self.link_cntx_type.lower() in [
            'cat4', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46',
            'cat6', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66',
            'cat7', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76']:

            link_aug_attr = torch.cat([h_aug_edges, h_parent_per_augEdge], dim=1) if self.level > 0 else h_aug_edges

            graph_aug_attr = None
            if self.link_cntx_type.lower() in ['cat42', 'cat62', 'cat72']:
                graph_aug_attr = pyg.nn.global_add_pool(g_batch.x, batch=g_batch.batch)
            elif self.link_cntx_type.lower() in ['cat45', 'cat65', 'cat75']:
                graph_aug_attr = pyg.nn.global_mean_pool(g_batch.x, batch=g_batch.batch)
            elif self.link_cntx_type.lower() in ['cat46', 'cat66', 'cat76']:
                ptr_ = g_batch.x_ptr
                graph_aug_attr = pyg.nn.aggr.MultiAggregation(
                    aggrs=['mean', 'max', 'min', 'std'],
                    mode='cat')(g_batch.x, ptr=ptr_)

            if self.link_cntx_type.lower() in ['cat62', 'cat63', 'cat64', 'cat65', 'cat66',
                                               'cat72', 'cat73', 'cat74', 'cat75', 'cat76']:
                # to concatenate the parent node attr
                graph_aug_attr = torch.cat([graph_aug_attr, h_parent_per_augGraph], dim=-1)

            if self.link_cntx_type.lower() in ['cat7', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76']:
                if self.link_cntx_type.lower() in ['cat75']:
                    h_graph_per_augGraph = pyg.nn.global_mean_pool(g_batch.x, batch=g_batch.batch)
                elif self.link_cntx_type.lower() in ['cat76']:
                    ptr_ = g_batch.x_ptr
                    h_graph_per_augGraph = pyg.nn.aggr.MultiAggregation(
                        aggrs=['mean', 'max', 'min', 'std'], mode='cat')(g_batch.x, ptr=ptr_)
                else:
                    h_graph_per_augGraph = pyg.nn.global_add_pool(g_batch.x, batch=g_batch.batch)
                link_aug_attr = torch.cat(
                    [link_aug_attr, stat_per_augEdge, h_graph_per_augGraph[g_batch.edge_index_aug_batch]],
                    dim=1)

            link_aug_cntx = torch.stack(
                [g_batch.remaining_weight, g_batch.all_weight, g_batch.num_nodes_aug_graph],
                dim=1)
            link_attr_bn_level = alpha_level = 'graph'
            is_logp_link_attr_bn = False
            if self.link_cntx_type.lower() in ['cat4', 'cat6', 'cat7']:
                is_logp_link_attr_bn = True
                link_attr_bn_level = alpha_level = 'edge'

        else:
            raise NotImplementedError

        return (link_aug_attr, graph_aug_attr, link_aug_cntx, is_logp_link_attr_bn, link_attr_bn_level, alpha_level)

    def get_nll_bn_mn(self, g_batch, theta_bn, theta_mn, theta_self, is_logit_bn, is_logit_mn):

        nll_binomial = get_nll_binomial(
            theta_bn,
            g_batch.cutself_weight.unsqueeze(1),
            n_trials=g_batch.remaining_weight.unsqueeze(1),
            reduction='none',
            mode=is_logit_bn
        )

        nll_self = 0.
        if self.to_model_selfedge:
            self_weight = g_batch.self_weight
            if not theta_self is None:
                nll_self = get_nll_binomial(
                    theta_self,
                    self_weight.unsqueeze(1),
                    n_trials=g_batch.cutself_weight.unsqueeze(1),
                    reduction='none',
                    mode=is_logit_bn
                )

        split_lengths = list(g_batch.num_aug_edges.cpu().numpy())
        mask = split_then_pad(
            torch.ones([g_batch.edge_index_aug_batch.shape[0]], device=theta_mn.device),
            split_lengths=split_lengths, value=.0
        ).t()
        nll_multinomial = get_nll_multinomial(
            theta_mn.permute([1, 2, 0]),
            target=split_then_pad(g_batch.label, split_lengths=split_lengths, value=.0).t().type_as(g_batch.label),
            n_trials=g_batch.cutset_weight,
            reduction='none',
            mask=mask,
            mode=is_logit_mn
        )
        return nll_binomial + nll_multinomial + nll_self

    def get_loss_gen_part(self, g_batch, parent_graph, to_gen=False):
        to_train = not to_gen

        _h_aug_edges = self.node2edge_attr(g_batch.x[g_batch.edge_index_aug[0, :], :],
                                           g_batch.x[g_batch.edge_index_aug[1, :], :], is_bp=False)

        # To get the context of the parent node and make link and graph's attributes
        link_aug_attr, graph_aug_attr, link_aug_cntx, _, link_attr_bn_level, alpha_level = \
            self.get_aug_part_attributes(
                g_batch=g_batch,
                h_aug_edges=_h_aug_edges,
                parent_graph=parent_graph,
            )

        if self.dist == 'mix_Bernouli':
            log_theta = self.model_theta(link_aug_attr, g_batch.num_aug_edges)
            log_alpha = self.model_alpha(link_aug_attr)
            log_alpha = get_reduced_attr(log_alpha, scatter_index=g_batch.edge_index_aug_batch,
                                         num_graphs=g_batch.num_aug_edges.shape[0], num_edges=g_batch.num_aug_edges,
                                         get_average=True)

            if to_train:
                split_lengths = list(g_batch.num_aug_edges.cpu().numpy())
                mask = split_then_pad(torch.ones([g_batch.edge_index_aug_batch.shape[0]], device=log_theta.device),
                                      split_lengths=split_lengths, value=.0)
                label_subgs_ = split_then_pad(g_batch.label,
                                              split_lengths=split_lengths,
                                              value=.0).t().type_as(g_batch.label)

                log_theta = torch.clamp(log_theta,
                                        min=torch.tensor(-1e6, device=log_theta.device),
                                        max=torch.tensor(1e6, device=log_theta.device))
                loss_adj = self.BCE_loss_func(log_theta,
                                              label_subgs_.t().unsqueeze(2).expand(-1, -1, self.num_mix_component))
                loss_adj = (loss_adj * mask.unsqueeze(2)).sum(dim=0)

                nll_augparts = get_mixture_nll(nlls=loss_adj, log_alpha=log_alpha,
                                               num_aug_edges=g_batch.num_aug_edges,
                                               scatter_index=g_batch.edge_index_aug_batch,
                                               alpha_level=None,
                                               split_lengths=split_lengths)
                loss_adj_avg = self.get_nll_avg_level(nll_augparts, g_batch, part_bipart='part')
                return loss_adj_avg

            else:

                g_aug_set = Batch.to_data_list(g_batch)
                for i_g, g_aug in enumerate(g_aug_set):
                    if g_aug.is_completed:
                        continue

                    _log_theta = log_theta[:, i_g, :]
                    _log_theta = _log_theta.view(-1, self.num_mix_component)

                    _log_alpha = log_alpha[i_g, :].view(-1, self.num_mix_component)
                    _prob_alpha = F.softmax(_log_alpha.mean(dim=0), -1)
                    alpha = _prob_alpha.argmax()
                    _prob_edges = torch.sigmoid(_log_theta[:, alpha])
                    if 'Connected' in self.gen_completion_part:
                        # it makes sure that at least one edge is connected
                        _prob_edges[torch.argmax(_prob_edges)] = 1.

                    if 'LeafMode' in self.gen_completion_part:
                        # this is equivalent to setting the temperature of randomness to zero
                        edge_val_pred = (_prob_edges > .5).type_as(_prob_edges)
                    else:
                        edge_val_pred = torch.bernoulli(_prob_edges)

                    Adj_pred_sp = torch.sparse_coo_tensor(
                        indices=g_aug.edge_index_aug, values=edge_val_pred,
                        size=(g_aug.num_nodes, g_aug.num_nodes)).coalesce()
                    Adj_orig_sp = torch.sparse_coo_tensor(
                        indices=g_aug.edge_index_orig, values=g_aug.edge_weight_orig,
                        size=(g_aug.num_nodes, g_aug.num_nodes)).coalesce()

                    Adj_sp = Adj_orig_sp + Adj_pred_sp
                    Adj = Adj_sp.to_dense()
                    Adj_uper = torch.tril(Adj, diagonal=-1 if self.no_self_edge else 0)
                    Adj_uper = Adj_uper.transpose(1, 0)
                    Adj_lower = torch.tril(Adj, diagonal=-1)
                    Adj = Adj_lower + Adj_uper  # to obtain a symmetric adj matrix

                    g_aug.is_completed = False
                    if 'NoNewEdge' in self.gen_completion_part:
                        if (Adj_lower[-1:, :].sum() == 0 and g_aug.num_nodes > 1):
                            g_aug.is_completed = True
                            g_aug.num_nodes -= 1
                            Adj = Adj[:-1, :-1]

                    elif 'NumEdgesNumNodes2' in self.gen_completion_part:
                        g_aug.is_completed = True if Adj_uper.sum() > g_aug.all_weight else False
                        if g_aug.num_nodes >= g_aug.max_num_nodes:
                            g_aug.is_completed = True
                            if self._verbose >= 2:
                                print(f"part generation is terminated @ {g_aug.num_nodes} nodes "
                                      f"out of {self.config.model.stat_HG_pmf.max_num_nodes_part[self.level]}, "
                                      f"max pre_sampled num_nodes: {g_aug.max_num_nodes}")

                    elif 'NumEdgesNumNodes' in self.gen_completion_part:
                        g_aug.is_completed = True if Adj_uper.sum() > g_aug.all_weight else False
                        if g_aug.num_nodes >= self.config.model.stat_HG_pmf.max_num_nodes_part[self.level]:
                            g_aug.is_completed = True
                            if self._verbose >= 2:
                                print(f"part generation is terminated @ {g_aug.num_nodes} nodes "
                                      f"out of {self.config.model.stat_HG_pmf.max_num_nodes_part[self.level]}, "
                                      f"max pre_sampled num_nodes: {g_aug.max_num_nodes}")

                    elif 'NumEdges' in self.gen_completion_part:
                        g_aug.is_completed = True if Adj_uper.sum() > g_aug.all_weight else False

                    elif 'NumNodes' in self.gen_completion_part:
                        g_aug.is_completed = True if (g_aug.num_nodes >= g_aug.max_num_nodes) else False

                    Adj_sp = Adj.to_sparse().coalesce()
                    g_aug.edge_index = Adj_sp.indices().long()
                    g_aug.edge_weight = Adj_sp.values()
                    g_aug_set[i_g] = g_aug

                return g_aug_set

        if self.dist == 'mix_multinomial':
            (theta_bn, theta_self), theta_mn, log_alpha, is_logit_bn, is_logit_mn = self.model_linkpred.forward(
                link_aug_attr, graph_aug_attr, link_aug_cntx,
                g_batch=g_batch,
                alpha_level=alpha_level,
                link_attr_bn_level=link_attr_bn_level,
                is_bipart=False,
                to_cache_out_NN=to_gen, use_cache_out_NN=False,
            )

            if to_train:
                nll_augparts_allmix = self.get_nll_bn_mn(
                    g_batch, theta_bn, theta_mn, theta_self,
                    is_logit_bn=is_logit_bn,
                    is_logit_mn=is_logit_mn
                )
                nll_augparts = get_mixture_nll(
                    nlls=nll_augparts_allmix, log_alpha=log_alpha,
                    num_aug_edges=g_batch.num_aug_edges,
                    scatter_index=g_batch.edge_index_aug_batch,
                    alpha_level=None,
                    split_lengths=list(g_batch.num_aug_edges.cpu().numpy())
                )

                loss_adj_avg = self.get_nll_avg_level(nll_augparts, g_batch, part_bipart='part')
                return loss_adj_avg

            else:

                g_aug_set = Batch.to_data_list(g_batch)
                alpha_ls = [None] * len(g_aug_set)
                # to sample the cutset + self weight
                for i_g, g_aug in enumerate(g_aug_set):
                    g_aug.is_completed = False if g_aug.remaining_weight >= 0 else True
                    if g_aug.is_completed:
                        continue

                    # sample the mixture weight
                    _log_alpha = log_alpha[i_g, :]
                    _prob_alpha = F.softmax(_log_alpha, -1)
                    alpha = _prob_alpha.argmax()
                    alpha_ls[i_g] = alpha

                    prob_cutself = theta_bn[i_g, alpha]
                    prob_cutself = torch.sigmoid(prob_cutself) if is_logit_bn == 'logit' else prob_cutself
                    g_aug.cutself_weight = binomial_sampler(total_count=g_aug.remaining_weight, probs=prob_cutself,
                                                            mode=self._probe_mode, n_sample=self._n_sample_best)
                    g_batch.cutself_weight[i_g] = g_aug.cutself_weight

                # to sample the self weight
                ## in some cases the link's output are function of cutself_weight so we need to re-calculate them.
                (_, theta_self), theta_mn, log_alpha, is_logit_bn, is_logit_mn = self.model_linkpred.forward(
                    link_aug_attr, graph_aug_attr, link_aug_cntx,
                    g_batch,
                    alpha_level=alpha_level,
                    link_attr_bn_level=link_attr_bn_level,
                    is_bipart=False,
                    to_cache_out_NN=False, use_cache_out_NN=True
                )
                Adj_pred_sp_ls = [None] * len(g_aug_set)
                for i_g, g_aug in enumerate(g_aug_set):
                    if g_aug.is_completed:
                        continue
                    alpha = alpha_ls[i_g]

                    g_aug.self_weight = torch.tensor(0., device=self.config.device)
                    if self.to_model_selfedge:
                        if g_aug.edge_index_aug.numel() == 0:
                            g_aug.self_weight = g_aug.cutself_weight
                        else:
                            prob_self = theta_self[i_g, alpha]
                            prob_self = torch.sigmoid(prob_self) if is_logit_bn == 'logit' else prob_self
                            g_aug.self_weight = binomial_sampler(total_count=g_aug.cutself_weight, probs=prob_self,
                                                                 mode=self._probe_mode, n_sample=self._n_sample_best)
                            if g_aug.num_nodes_aug_graph == 1:
                                g_aug.self_weight = g_aug.cutself_weight

                    Adj_pred_sp = torch.sparse_coo_tensor(
                        indices=torch.stack([g_aug.num_nodes_aug_graph - 1, g_aug.num_nodes_aug_graph - 1], dim=0),
                        values=g_aug.self_weight,
                        size=(g_aug.num_nodes, g_aug.num_nodes)).coalesce()
                    Adj_pred_sp_ls[i_g] = Adj_pred_sp
                    g_aug.cutset_weight = g_aug.cutself_weight - g_aug.self_weight
                    g_aug.cutset_weight += self.is_connected_graph if g_aug.num_nodes_aug_graph.item() > 1 else 0
                    g_batch.cutset_weight[i_g] = g_aug.cutset_weight
                    g_batch.self_weight[i_g] = g_aug.self_weight

                # to sample the weights of the cross edge (all edges except self edge)
                ## to re-calculate the outputs with cutset_weight
                (_, _), theta_mn, log_alpha, is_logit_bn, is_logit_mn = self.model_linkpred.forward(
                    link_aug_attr, graph_aug_attr, link_aug_cntx,
                    g_batch,
                    alpha_level=alpha_level,
                    link_attr_bn_level=link_attr_bn_level,
                    is_bipart=False,
                    to_cache_out_NN=False, use_cache_out_NN=True
                )
                for i_g, g_aug in enumerate(g_aug_set):
                    if g_aug.is_completed:
                        continue
                    alpha = alpha_ls[i_g]
                    Adj_pred_sp = Adj_pred_sp_ls[i_g]

                    if g_aug.cutset_weight > 0.:

                        theta_mn_ = theta_mn.permute([1, 0, 2])[i_g, :, alpha]
                        _prob_edges = self.model_linkpred.get_edge_prob(theta_mn_, dim=0, is_logit_mn=is_logit_mn)
                        if ('NumLeaf' in self.gen_completion_part) and self.is_leaf_level:
                            _, nonzero_ind = torch.topk(_prob_edges,
                                                        min(int(g_aug.cutset_weight.item()), len(_prob_edges)),
                                                        sorted=False)
                            edge_val_pred = torch.zeros_like(_prob_edges)
                            edge_val_pred[nonzero_ind] = 1.
                        else:
                            edge_val_pred = multinomial_sampler(total_count=g_aug.cutset_weight, probs=_prob_edges,
                                                                mode=self._probe_mode, n_sample=self._n_sample_best)
                            if ('BinLeaf' in self.gen_completion_part) and self.is_leaf_level:
                                edge_val_pred[edge_val_pred >= 1] = 1
                        Adj_pred_sp += torch.sparse_coo_tensor(indices=g_aug.edge_index_aug, values=edge_val_pred,
                                                               size=(g_aug.num_nodes, g_aug.num_nodes)).coalesce()

                    Adj_orig_sp = torch.sparse_coo_tensor(indices=g_aug.edge_index_orig, values=g_aug.edge_weight_orig,
                                                          size=(g_aug.num_nodes, g_aug.num_nodes)).coalesce()
                    Adj_sp = Adj_orig_sp + Adj_pred_sp
                    Adj = Adj_sp.to_dense()
                    Adj_uper = torch.tril(Adj, diagonal=-1)
                    Adj_uper = Adj_uper.transpose(1, 0)
                    Adj_lower = torch.tril(Adj, diagonal=-1 if self.no_self_edge else 0)
                    Adj = Adj_lower + Adj_uper  # to obtain a symmetric adj matrix

                    Adj_sp = Adj.to_sparse().coalesce()
                    g_aug.edge_index = Adj_sp.indices().long()
                    g_aug.edge_weight = Adj_sp.values()

                    g_aug.remaining_weight -= g_aug.cutself_weight
                    g_aug.is_completed = False if g_aug.remaining_weight > 0 else True
                    g_aug.remaining_weight -= self.is_connected_graph
                    g_aug_set[i_g] = g_aug

                return g_aug_set

    def node2edge_attr(self, h_src, h_dst, is_bp=False):

        if self.link_attr_type == 'diff':
            h_edge = h_src - h_dst
            if is_bp:
                h_edge = torch.abs(h_edge)
        if self.link_attr_type == 'diff2':
            h_edge = h_src - h_dst
            if is_bp:
                h_edge = h_edge ** 2.
        if self.link_attr_type == 'adiff':
            h_edge = torch.abs(h_src - h_dst)
        elif self.link_attr_type == 'sum':
            h_edge = .5 * (h_src + h_dst)
        elif self.link_attr_type == 'cat':
            h_edge = torch.cat([h_src, h_dst], dim=1)
        return h_edge

    def get_link_attributes_augBP(self, bp_batch, parent_graph):

        edge_index_aug = bp_batch.edge_index_aug
        edge_index_parent_aug = [bp_batch.parent_node_id[bp_batch.edge_index_aug[0, :]],
                                 bp_batch.parent_node_id[bp_batch.edge_index_aug[1, :]]]

        if bp_batch.get('computed_edge_feat', False):
            h_edges_aug = bp_batch.edge_feat_aug
        else:
            h_edges_aug = self.node2edge_attr(bp_batch.x[edge_index_aug[0, :], :], bp_batch.x[edge_index_aug[1, :], :],
                                              is_bp=True)

        if parent_graph.get('computed_edge_feat', False):
            temp = pyg.utils.to_dense_adj(parent_graph.edge_index,
                                          edge_attr=torch.arange(parent_graph.edge_index.shape[1],
                                                                 device=parent_graph.edge_index.device))[0]
            id_parent_edge = temp[edge_index_parent_aug[0], edge_index_parent_aug[1]]
            h_parent_edge = parent_graph.edge_feat[id_parent_edge, :]

        else:
            h_parent_edge = self.node2edge_attr(parent_graph.x[edge_index_parent_aug[0], :],
                                                parent_graph.x[edge_index_parent_aug[1], :], is_bp=True)

        h_extra_aug = bp_batch.edge_weight_aug
        h_extra_bps = bp_batch.edge_weight_aug[bp_batch.num_aug_edges.cumsum(dim=0) - 1]
        bp_batch.cutset_weight = h_extra_bps
        if not self.has_edge_weight:
            h_extra_aug = torch.zeros_like(h_extra_aug)
            h_extra_bps = torch.zeros_like(h_extra_bps)

        return h_edges_aug, h_parent_edge, torch.unsqueeze(h_extra_aug, dim=1), torch.unsqueeze(h_extra_bps, dim=1)

    def get_loss_gen_bipart(self, bp_batch, part_graph_dict, parent_graph, to_gen=False):
        to_train = not to_gen

        if bp_batch.num_graphs == 0:
            return 0.

        h_edges_aug, h_parent_edge, h_extra_aug, h_extra_bps = \
            self.get_link_attributes_augBP(bp_batch, parent_graph=parent_graph)

        alpha_level = 'edge'
        bp_aug_attr = None
        if self.link_cntx_type_bp.lower() == 'none':
            link_attr = h_edges_aug
            bp_cntx = None
        elif self.link_cntx_type_bp.lower() == 'cat':
            link_attr = torch.cat([h_edges_aug, h_parent_edge], dim=1)
            bp_cntx = h_extra_bps
        elif self.link_cntx_type_bp.lower() == 'cat2':
            link_attr = torch.cat([h_edges_aug, h_parent_edge, h_extra_aug], dim=1)
            bp_cntx = None
        elif self.link_cntx_type_bp.lower() in ['cat3', 'cat32', 'cat33']:
            link_attr = torch.cat([h_edges_aug, h_parent_edge, h_extra_aug], dim=1)
            if self.link_cntx_type_bp.lower() == 'cat32':
                batch_aug_edge_ = torch.repeat_interleave(
                    torch.arange(len(bp_batch.num_aug_edges), device=bp_batch.num_aug_edges.device,
                                 dtype=bp_batch.num_aug_edges.dtype),
                    bp_batch.num_aug_edges
                )
                bp_aug_attr = pyg.nn.global_mean_pool(h_edges_aug, batch=batch_aug_edge_)
                ptr_ = torch.cat([torch.tensor([0], device=bp_batch.num_aug_edges.device),
                                  bp_batch.num_aug_edges.cumsum(dim=0)])
                bp_aug_attr = torch.cat([bp_aug_attr, h_parent_edge[ptr_[:-1]], h_extra_bps], dim=1)
                alpha_level = 'graph'
            if self.link_cntx_type_bp.lower() == 'cat33':
                ptr_ = torch.cat(
                    [torch.tensor([0], device=bp_batch.num_aug_edges.device), bp_batch.num_aug_edges.cumsum(dim=0)])
                bp_aug_attr = pyg.nn.aggr.MultiAggregation(
                    aggrs=['mean', 'max', 'min', 'std'], mode='cat')(h_edges_aug, ptr=ptr_)
                bp_aug_attr = torch.cat([bp_aug_attr, h_parent_edge[ptr_[:-1]], h_extra_bps], dim=1)
                alpha_level = 'graph'
            bp_cntx = h_extra_bps
        else:
            raise NotImplementedError

        if self.dist == 'mix_Bernouli':
            log_theta = self.model_theta(link_attr, bp_batch.num_aug_edges)
            if alpha_level == 'edge':
                log_alpha = self.model_alpha(link_attr)
                log_alpha = get_reduced_attr(log_alpha, num_edges=bp_batch.num_aug_edges, get_average=True)
            else:
                log_alpha = self.model_alpha(bp_aug_attr)

            if self.postMixBP:
                log_alpha = pyg.nn.global_mean_pool(log_alpha, batch=bp_batch.num_aug_edges_batch)

            if to_train:
                if self.dist == 'mix_Bernouli':
                    _split_lengths = list(bp_batch.num_aug_edges.cpu().numpy())
                    _mask = split_then_pad(
                        torch.ones([bp_batch.edge_index_aug_batch.shape[0]], device=log_theta.device),
                        split_lengths=_split_lengths, value=.0)
                    label_bps = split_then_pad(bp_batch.label,
                                               split_lengths=_split_lengths, value=.0).t().type_as(bp_batch.label)
                    log_theta = torch.clamp(log_theta, min=torch.tensor(-1e6, device=log_theta.device),
                                            max=torch.tensor(1e6, device=log_theta.device))
                    loss_adj_bps = self.BCE_loss_func(log_theta,
                                                      label_bps.t().unsqueeze(2).expand(-1, -1, self.num_mix_component))
                    loss_adj_bps = (loss_adj_bps * _mask.unsqueeze(2)).sum(dim=0)

                    if not self.postMixBP:
                        nll_bps = get_mixture_nll(
                            nlls=loss_adj_bps, log_alpha=log_alpha,
                            num_aug_edges=bp_batch.num_aug_edges,
                            scatter_index=bp_batch.edge_index_aug_batch,
                            alpha_level='graph',
                            split_lengths=_split_lengths)
                    else:
                        loss_adj_bps = pyg.nn.global_add_pool(loss_adj_bps, batch=bp_batch.num_aug_edges_batch)
                        nll_bps = get_mixture_nll(
                            nlls=loss_adj_bps, log_alpha=log_alpha,
                            num_aug_edges=None,
                            scatter_index=None,
                            alpha_level='graph')
                    loss_adj_bps_avg = self.get_nll_avg_level(nll_bps, bp_batch,
                                                              part_bipart='bipart',
                                                              post_mixture=self.postMixBP)
                    return loss_adj_bps_avg

            else:
                # To generate the edges of bipart graphs
                assert self.config.test.batch_size == 1  # not implemented for larger batch_size
                bp_aug = bp_batch

                aug_edge_end_ls = bp_aug.num_aug_edges.cumsum(dim=0)
                ## second half of the edge_index_aug are transpose of the first half since the graph is undirected
                aug_edge_end_ls = aug_edge_end_ls[:len(aug_edge_end_ls) // 2]
                edge_index_ls = [bp_aug.edge_index]
                edge_weight_ls = [bp_aug.edge_weight]
                for i_bp, _end in enumerate(aug_edge_end_ls):
                    _start = aug_edge_end_ls[i_bp - 1] if i_bp > 0 else 0
                    log_theta_bp = log_theta[:bp_aug.num_aug_edges[i_bp], i_bp, :]
                    edge_index_aug_bp = bp_aug.edge_index_aug[:, _start: _end]
                    if not self.postMixBP:
                        prob_alpha = F.softmax(log_alpha[i_bp, :], -1)
                    else:
                        prob_alpha = F.softmax(log_alpha[0, :], -1)
                    alpha = prob_alpha.argmax()

                    prob_edges = torch.sigmoid(log_theta_bp[:, alpha])
                    if 'NumEdges' in self.gen_completion_bipart:
                        _, nonzero_ind = torch.topk(prob_edges, int(bp_aug.cutset_weight[i_bp].item()), sorted=False)
                        pred_edges_values = torch.zeros_like(prob_edges)
                        pred_edges_values[nonzero_ind] = 1.
                    elif 'LeafMode' in self.gen_completion_bipart:
                        pred_edges_values = (prob_edges > .5).type_as(prob_edges)
                        nonzero_ind = pred_edges_values.nonzero().squeeze(dim=1)
                    else:
                        pred_edges_values = torch.bernoulli(prob_edges)
                        nonzero_ind = pred_edges_values.nonzero().squeeze(dim=1)

                    edge_index_bp = edge_index_aug_bp[:, nonzero_ind]
                    edge_weight_bp = pred_edges_values[nonzero_ind]
                    edge_index_ls.append(edge_index_bp)
                    edge_weight_ls.append(edge_weight_bp)
                    # To add reverse edges for undirected graphs
                    edge_index_ls.append(edge_index_bp[(1, 0), :])
                    edge_weight_ls.append(edge_weight_bp)

                    if self.config.model.to_joint_aug_bp.lower() == 'ar':
                        bp_aug.edge_index_bp = edge_index_bp - bp_aug.bp_edge_bias
                        bp_aug.edge_weight_bp = edge_weight_bp

                bp_aug.edge_index = torch.cat(edge_index_ls, dim=1)
                bp_aug.edge_weight = torch.cat(edge_weight_ls, dim=0)
                delattr(bp_aug, 'edge_index_aug')
                delattr(bp_aug, 'edge_weight_aug')
                return bp_aug

        elif self.dist == 'mix_multinomial':
            _, theta_mn, log_alpha, _, is_logit_mn = self.model_linkpred.forward(
                link_attr, graph_attr=bp_aug_attr,
                g_batch=bp_batch, link_cntx=bp_cntx,
                alpha_level=alpha_level, link_attr_bn_level=None,
                is_bipart=True, cat_cntx=False,
            )
            theta_mn = theta_mn.permute([1, 2, 0])

            if self.postMixBP:
                log_alpha = pyg.nn.global_mean_pool(log_alpha, batch=bp_batch.num_aug_edges_batch)

            if to_train:
                _split_lengths = list(bp_batch.num_aug_edges.cpu().numpy())
                _mask = split_then_pad(torch.ones([bp_batch.edge_index_aug_batch.shape[0]], device=theta_mn.device),
                                       split_lengths=_split_lengths, value=.0).t()
                NLL_sampled_biparts_allmix = get_nll_multinomial(
                    theta_mn,
                    target=split_then_pad(bp_batch.label, split_lengths=_split_lengths, value=.0).t().type_as(
                        bp_batch.label),
                    n_trials=bp_batch.cutset_weight,
                    reduction='none',
                    mask=_mask,
                    # num_mix=self.num_mix_component
                )

                if not self.postMixBP:
                    NLL_sampled_biparts = get_mixture_nll(
                        nlls=NLL_sampled_biparts_allmix, log_alpha=log_alpha,
                        num_aug_edges=bp_batch.num_aug_edges,
                        scatter_index=bp_batch.edge_index_aug_batch,
                        alpha_level='graph',
                        split_lengths=_split_lengths)
                else:
                    NLL_sampled_biparts_allmix = pyg.nn.global_add_pool(NLL_sampled_biparts_allmix,
                                                                        batch=bp_batch.num_aug_edges_batch)
                    NLL_sampled_biparts = get_mixture_nll(
                        nlls=NLL_sampled_biparts_allmix, log_alpha=log_alpha,
                        num_aug_edges=None,
                        scatter_index=None,
                        alpha_level='graph')

                loss_adj_bps_avg = self.get_nll_avg_level(
                    NLL_sampled_biparts, bp_batch,
                    part_bipart='bipart', post_mixture=self.postMixBP)
                return loss_adj_bps_avg

            else:
                # To generate the edges of bipart graphs
                assert self.config.test.batch_size == 1  # not implemented for larger batch_size
                bp_aug = bp_batch

                _split_lengths = list(bp_batch.num_aug_edges.cpu().numpy())
                _mask = split_then_pad(torch.ones([bp_batch.edge_index_aug.shape[1]]),
                                       split_lengths=_split_lengths, value=0).t().type(torch.bool)

                aug_edge_end_ls = bp_aug.num_aug_edges.cumsum(dim=0)
                # second half of the edge_index_aug are transpose of the first half since the graph is undirected
                aug_edge_end_ls = aug_edge_end_ls[:len(aug_edge_end_ls) // 2]
                edge_index_ls = [bp_aug.edge_index]
                edge_weight_ls = [bp_aug.edge_weight]
                for i_bp, _end in enumerate(aug_edge_end_ls):
                    _start = aug_edge_end_ls[i_bp - 1] if i_bp > 0 else 0
                    edge_index_aug_bp = bp_aug.edge_index_aug[:, _start: _end]
                    if not self.postMixBP:
                        prob_alpha = F.softmax(log_alpha[i_bp, :], -1)
                    else:
                        prob_alpha = F.softmax(log_alpha[0, :], -1)
                    alpha = prob_alpha.argmax()
                    logit_ = theta_mn[i_bp, alpha, _mask[i_bp]]
                    prob_edges = self.model_linkpred.get_edge_prob(logit_, is_logit_mn=is_logit_mn)

                    if ('NumLeaf' in self.gen_completion_bipart) and self.is_leaf_level:
                        _, nonzero_ind = torch.topk(prob_edges,
                                                    min(int(bp_aug.cutset_weight[i_bp].item()), len(prob_edges)),
                                                    sorted=False)
                        pred_edges_values = torch.zeros_like(prob_edges)
                        pred_edges_values[nonzero_ind] = 1.
                    else:
                        pred_edges_values = multinomial_sampler(total_count=bp_aug.cutset_weight[i_bp],
                                                                probs=prob_edges,
                                                                mode=self._probe_mode, n_sample=self._n_sample_best)
                        if pred_edges_values.dim() == 0:  # to prevent the error if it is a scalar
                            pred_edges_values = pred_edges_values.unsqueeze(0)

                        nonzero_ind = pred_edges_values.nonzero().squeeze(dim=1)
                        if ('BinLeaf' in self.gen_completion_bipart) and self.is_leaf_level:
                            pred_edges_values[pred_edges_values >= 1] = 1

                    edge_index_bp = edge_index_aug_bp[:, nonzero_ind]
                    edge_weight_bp = pred_edges_values[nonzero_ind]
                    edge_index_ls.append(edge_index_bp)
                    edge_weight_ls.append(edge_weight_bp)

                    # for undirected graphs
                    edge_index_ls.append(edge_index_bp[(1, 0), :])
                    edge_weight_ls.append(edge_weight_bp)

                    if self.config.model.to_joint_aug_bp.lower() == 'ar':
                        bp_aug.edge_index_bp = edge_index_bp - bp_aug.bp_edge_bias
                        bp_aug.edge_weight_bp = edge_weight_bp

                bp_aug.edge_index = torch.cat(edge_index_ls, dim=1)
                bp_aug.edge_weight = torch.cat(edge_weight_ls, dim=0)
                delattr(bp_aug, 'edge_index_aug')
                delattr(bp_aug, 'edge_weight_aug')
                return bp_aug


class Link_Prediction_Model(nn.Module):

    def __init__(self, config, level):
        super().__init__()
        self.model_type_bn, self.model_type_mn, self.model_type_selfedge = config.model.get('LP_model',
                                                                                            'b11_m10_s1').split('_')
        self.level = level
        self.config = config
        self.is_leaf_level = self.level == config.dataset.num_levels - 1
        self.num_mix_component = config.model.num_mix_component
        self.has_edge_weight = config.model.has_edge_weight
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = 1
        self.output_dim_cntx1, self.output_dim_cntx2 = 8, 5
        self.dropout = config.model.get('dropout', 0.)
        self.is_connected_graph = int(config.model.get("is_connected_graph",
                                                       1))  # if 1 the an edge in each cutset is always reserved so that cutset_weight is greater than 1
        self.no_self_edge = config.dataset.no_self_edge == 'all' or (
                config.dataset.no_self_edge.lower() == 'last' and self.level == config.dataset.num_levels - 1)

        self.output_bn = Output_Model(
            hidden_dim=self.hidden_dim, out_dim=2 * self.num_mix_component,
            n_layers=4, gate=nn.ReLU(),
            model_type='simple', device=self.config.device, )
        self.output_selfedge = Output_Model(
            hidden_dim=self.hidden_dim, out_dim=self.num_mix_component,
            n_layers=4, gate=nn.ReLU(),
            model_type='simple', device=self.config.device, )
        self.output_mn = Output_Model(
            hidden_dim=self.hidden_dim * 4, out_dim=self.output_dim * self.num_mix_component,
            n_layers=3, gate=nn.ReLU(),
            model_type='split', device=self.config.device, )
        self.output_alpha = Output_Model(
            hidden_dim=self.hidden_dim, out_dim=self.num_mix_component,
            n_layers=4, gate=nn.ReLU(),
            model_type='simple', device=self.config.device, )
        self.output_alpha.init_weights()

        self.output_cntx = Output_Model(
            hidden_dim=64, out_dim=self.output_dim_cntx1 + self.output_dim_cntx2,
            n_layers=4, gate=nn.ELU(),
            model_type='simple', device=self.config.device, )

        self.add_module("output_bn", self.output_bn)
        self.add_module("output_mn", self.output_mn)
        self.add_module("output_alpha", self.output_alpha)
        self.add_module("output_cntx", self.output_cntx)

    def get_edge_prob(self, theta_mn, is_logit_mn='logit', dim=0):
        """
        :param theta_mn:
        :param dim:
        :param is_logit_mn:
        :return:
        """
        if is_logit_mn == 'logit':
            return torch.softmax(theta_mn, dim=dim)
        elif is_logit_mn == 'logitsigm':
            prob = torch.sigmoid(theta_mn)
            return prob / prob.sum(dim=dim, keepdim=True)
        else:
            raise ValueError('NOT A VALID CHOICE')

    def forward(self, link_attr, graph_attr, link_cntx, g_batch, alpha_level, link_attr_bn_level,
                cat_cntx=False, to_cache_out_NN=False, use_cache_out_NN=False, is_bipart=False):
        """
        :param link_attr:
        :param graph_attr:
        :param link_cntx:
        :param g_batch:
        :param alpha_level:
        :param link_attr_bn_level:
        :param cat_cntx:
        :param to_cache_out_NN:
        :param use_cache_out_NN:
        :param is_bipart:
        :return:
        """
        is_logit_bn = 'logit'
        is_logit_mn = 'logit'
        if not use_cache_out_NN:
            theta_cntx_ = self.output_cntx(link_cntx) if not link_cntx is None else None
            theta_mn_ = self.output_mn(link_attr, g_batch.num_aug_edges)
        else:
            # to use saved output to save the computation time
            theta_mn_ = self.output_mn_cache
            theta_cntx_ = self.output_cntx_cache

        if to_cache_out_NN:
            # to save the computation time
            self.output_mn_cache = theta_mn_
            self.output_cntx_cache = theta_cntx_

        if self.model_type_mn == 'm10':
            theta_mn = theta_mn_
        elif self.model_type_mn == 'm11':
            denom = (1 + torch.nn.Softplus()(theta_cntx_[:, -5]))
            theta_mn = theta_mn_ / denom.unsqueeze(1).unsqueeze(0)
        elif self.model_type_mn == 'm12':
            weight_ = (1 + torch.nn.Softplus()(theta_cntx_[:, -5])) / ((g_batch.cutset_weight + 1.).log() + 1.)
            theta_mn = theta_mn_ * weight_.unsqueeze(1).unsqueeze(0)
        elif self.model_type_mn in ['m14', 'm142']:
            if (self.model_type_mn == 'm142') and (self.is_leaf_level):
                is_logit_mn = 'logitsigm'
                theta_mn = theta_mn_
            else:
                is_logit_mn = 'logit'
                weight_ = (1 + torch.nn.Softplus()(theta_cntx_[:, -5])) / ((g_batch.cutset_weight + 1.).log() + 1.)
                theta_mn = theta_mn_ * weight_.unsqueeze(1).unsqueeze(0)

        if is_bipart:
            if alpha_level == 'edge':
                log_alpha = self.output_alpha(link_attr)
                log_alpha = get_reduced_attr(log_alpha, num_edges=g_batch.num_aug_edges, get_average=True)
            else:
                log_alpha = self.output_alpha(graph_attr)
            return None, theta_mn, log_alpha, None, is_logit_mn

        theta_self = None
        if not use_cache_out_NN:
            log_alpha = self.output_alpha(graph_attr) if 'graph' in alpha_level else self.output_alpha(link_attr)
            if alpha_level == 'edge':
                log_alpha = get_reduced_attr(log_alpha, scatter_index=g_batch.edge_index_aug_batch,
                                             num_graphs=g_batch.num_aug_edges.shape[0],
                                             num_edges=g_batch.num_aug_edges, get_average=True)

            if not graph_attr is None:
                if cat_cntx:
                    graph_attr = torch.cat([graph_attr, theta_cntx_[:, :self.output_dim_cntx1], link_cntx], dim=-1)
                else:
                    graph_attr = torch.cat([graph_attr, theta_cntx_[:, :self.output_dim_cntx1]], dim=-1)
            theta_bn_ = self.output_bn(graph_attr) if 'graph' in link_attr_bn_level else self.output_bn(link_attr)
            theta_bn_, theta_self_ = theta_bn_[:, :self.num_mix_component], theta_bn_[:, self.num_mix_component:]

            if link_attr_bn_level == 'edge':
                theta_bn_ = get_reduced_attr(theta_bn_, scatter_index=g_batch.edge_index_aug_batch,
                                             num_graphs=g_batch.num_aug_edges.shape[0], get_average=False)
                theta_self_ = get_reduced_attr(theta_self_, scatter_index=g_batch.edge_index_aug_batch,
                                               num_graphs=g_batch.num_aug_edges.shape[0],
                                               num_edges=g_batch.num_aug_edges,
                                               get_average=True)

        else:
            # to use saved output to save the computation time
            log_alpha = self.output_alpha_cache
            theta_bn_ = self.theta_bn_cache
            theta_self_ = self.theta_self_cache

        if to_cache_out_NN:
            # to save the computation time
            self.output_alpha_cache = log_alpha
            self.theta_bn_cache = theta_bn_
            self.theta_self_cache = theta_self_

        remaining_weight = torch.clamp(g_batch.remaining_weight + self.is_connected_graph,
                                       max=torch.maximum(g_batch.all_weight, torch.ones_like(g_batch.all_weight)),
                                       min=torch.ones_like(g_batch.all_weight))
        cutself_weight = g_batch.cutself_weight
        all_weight = g_batch.all_weight

        if self.model_type_bn == 'b0':
            is_logit_bn = 'logit'
            theta_bn = theta_bn_
            theta_self = theta_self_
        elif self.model_type_bn == 'b10':
            is_logit_bn = 'logit'
            theta_bn = theta_bn_ / (remaining_weight.unsqueeze(1).log() + 1.)
            theta_self = theta_self_ / (cutself_weight.unsqueeze(1).log() + 1.)

        elif self.model_type_bn == 'b13':
            is_logit_bn = 'p'
            theta_bn = torch.sigmoid((theta_bn_ - theta_cntx_[:, -1].unsqueeze(-1)) / (
                    1 + torch.nn.Softplus()(theta_cntx_[:, -2]).unsqueeze(-1)))
            theta_bn = torch.clamp(theta_bn * (all_weight / remaining_weight).unsqueeze(1),
                                   max=torch.tensor(1., device=theta_bn_.device))
            theta_self = torch.sigmoid((theta_self_ - theta_cntx_[:, -3].unsqueeze(-1)) / (
                    1 + torch.nn.Softplus()(theta_cntx_[:, -4]).unsqueeze(-1)))

        return (theta_bn, theta_self), theta_mn, log_alpha, is_logit_bn, is_logit_mn
