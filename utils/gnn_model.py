from easydict import EasyDict as edict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from utils.encoder.kernel_pos_encoder import RWSENodeEncoder, \
    HKdiagSENodeEncoder, ElstaticSENodeEncoder, CyclesNodeEncoder
from utils.encoder.laplace_pos_encoder import LapPENodeEncoder
from utils.encoder.stableLaplace_pos_encoder import StableLapPENodeEncoder
from utils.encoder.signnet_pos_encoder import SignNetNodeEncoder
from utils.encoder.linear_node_encoder import LinearNodeEncoder
from utils.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from utils.encoder.graphormer_encoder import GraphormerEncoder

from utils.layer.gps_layer import GPSLayer
from utils.layer.ppgn_layer import Powerful


# Positional Encoding node encoders.
pe_encoder_dict = {
    'LapPE': LapPENodeEncoder,
    'RWSE': RWSENodeEncoder,
    'HKdiagSE': HKdiagSENodeEncoder,
    'ElstaticSE': ElstaticSENodeEncoder,
    'SignNet': SignNetNodeEncoder,
    'EquivStableLapPE': EquivStableLapPENodeEncoder,
    'GraphormerBias': GraphormerEncoder,
    'Cycles': CyclesNodeEncoder,
    'StLapPE': StableLapPENodeEncoder
}

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, config):
        # super(FeatureEncoder, self).__init__()
        super().__init__()
        self.config = config
        # self.dim_in = config.model.embedding_dim

        node_encoder_ls = []

        for _node_feat_type in config.model.node_feat_types:
            _enc = pe_encoder_dict[_node_feat_type](config, dim_emb=None, expand_x=False)
            node_encoder_ls.append(_enc)
        self.node_encoder_ls = nn.ModuleList(node_encoder_ls)

        if config.dataset.get('node_encoder_bn', False):
            self.node_encoder_bn = BatchNorm1dNode(
                new_layer_config(config.gt.dim_inner, -1, -1, has_act=False,
                                 has_bias=False, cfg=config))
            self.add_module("node_encoder_bn", self.node_encoder_bn)

        if 'GPSModel' in config.model.gnn_aug_part:
            self.pre_emb_layer_gran_edge = nn.LazyLinear(out_features=config.gt.dim_inner)


    def forward(self, batch, is_aug_part = False):

        # to use the ghran type node and edge feat for aug_part
        if is_aug_part:
            if self.config.model.use_gran_feat == 'only':
                batch.edge_attr = self.pre_emb_layer_gran_edge(batch.edge_feat)
                batch.edge_weight_aug = None
                return batch
            elif self.config.model.use_gran_feat == 'plus':
                batch.edge_attr = self.pre_emb_layer_gran_edge(batch.edge_feat)
                batch.edge_weight_aug = None
            elif self.config.model.use_gran_feat == 'none':
                batch.edge_attr = None

                # node encoders
        if not hasattr(batch, 'x') or batch.x is None:
            batch.x = torch.zeros([batch.num_nodes, 0], device=batch.edge_index.device)

        for module in self.node_encoder_ls:
            batch = module(batch)

        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, config, pre_emb_layer):
        super().__init__()
        self.encoder = FeatureEncoder(config)

        if pre_emb_layer:
            self.pre_emb_layer = pre_emb_layer
            self.dim_in = pre_emb_layer.embedding_dim
        else:
            self.dim_in = config.gt.dim_inner
            self.pre_emb_layer = nn.LazyLinear(out_features=self.dim_in)
        self.add_module("pre_emb_layer", self.pre_emb_layer)

        if config.gt.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                self.dim_in, config.gt.dim_inner, config.gt.layers_pre_mp)
            # dim_in = config.gt.dim_inner

        if not config.gt.dim_hidden == config.gt.dim_inner == self.dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={config.gt.dim_hidden} dim_inner={config.gt.dim_inner} "
                f"dim_in={self.dim_in}"
            )

        try:
            self.local_gnn_type, self.global_model_type = config.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {config.gt.layer_type}")
        layers = []
        for _ in range(config.gt.layers):
            layers.append(GPSLayer(
                dim_h=config.gt.dim_hidden,
                local_gnn_type=self.local_gnn_type,
                global_model_type=self.global_model_type,
                num_heads=config.gt.n_heads,
                act=config.gt.act,
                pna_degrees=config.gt.pna_degrees,
                equivstable_pe = config.gt.get('posenc_EquivStableLapPE_enable', False),
                dropout = config.gt.dropout,
                attn_dropout = config.gt.attn_dropout,
                layer_norm = config.gt.layer_norm,
                batch_norm = config.gt.batch_norm,
                bigbird_cfg = config.gt.get('bigbird', edict()),
                log_attn_weights = config.train.get('mode', 'custom') == 'log-attn-weights',
            ))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        # for module in self.children():
        # if self.local_gnn_type in ['CustomGatedGCN', 'GINE']:

        batch.edge_attr = batch.edge_attr if not batch.edge_attr is None \
            else torch.unsqueeze(batch.edge_weight.float(), dim=-1)
        batch.edge_index_aug = batch.get('edge_index_aug', None)
        batch.edge_attr_aug = batch.get('edge_attr_aug', None)
        if (batch.get('edge_attr_aug', None) is None) and hasattr(batch, 'edge_weight_aug'):
            batch.edge_attr_aug = torch.unsqueeze(batch.edge_weight_aug.float(), dim=-1)

        for module in self.layers:
            batch = module(batch)
        return batch

    def pre_emb_forward(self, batch):
        batch.x = self.pre_emb_layer(batch.x)
        return batch

@register_network('PPGNModel')
class PPGNModel(torch.nn.Module):
    """
    """

    def __init__(self, config, pre_emb_layer, is_augmented=False):
        super().__init__()
        cfg = config.pgnn
        self.is_augmented = is_augmented
        self.to_get_edge_feat = True

        self.encoder = FeatureEncoder(config)

        if pre_emb_layer:
            self.pre_emb_layer = pre_emb_layer
            self.dim_in = pre_emb_layer.embedding_dim
        else:
            self.dim_in = cfg.dim_inner
            self.pre_emb_layer = nn.LazyLinear(out_features=self.dim_in)
        self.add_module("pre_emb_layer", self.pre_emb_layer)

        if cfg.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                self.dim_in, config.gt.dim_inner, config.gt.layers_pre_mp)

        if cfg.act.lower() == 'gelu':
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=.2)

        input_features = cfg.dim_inner + (2 if self.is_augmented else 1)
        self.ppgn = Powerful(
            num_layers=cfg.layers,
            input_features=input_features,
            hidden=cfg.dim_hidden,
            hidden_final=cfg.dim_hidden,
            dropout_prob=cfg.dropout,
            simplified=False,
            n_nodes=-1, # is needed for LayerNorm
            normalization=cfg.norm,
            adj_out=True,
            output_features=cfg.dim_out,
            residual=False,
            activation=activation,
            node_out=True,
            node_output_features=cfg.dim_out
        )
        self.add_module("ppgn", self.ppgn)

    def forward(self, batch):

        # prepare batched Adj and node_feat and mask
        Adj = pyg.utils.to_dense_adj(batch.edge_index, batch=batch.batch)
        if self.is_augmented:
            Adj = torch.stack([Adj, pyg.utils.to_dense_adj(batch.edge_index_aug, batch=batch.batch)], dim=3)

        x, mask_ = pyg.utils.to_dense_batch(batch.x, batch.batch)
        mask = mask_.float()
        mask = mask.unsqueeze(-1) @ mask.unsqueeze(-2)

        # apply PPGN
        Adj_out, x = self.ppgn(Adj, node_features=x, mask=mask)

        batch.x = x[mask_]

        # extract augmented edge or edge features
        Adj_out = (Adj_out + Adj_out.transpose(1, 2)) / 2
        if self.to_get_edge_feat:
            self.assign_edge_features(Adj_out, batch, is_augmented=self.is_augmented)

        return batch

    def assign_edge_features(self, e_dense, batch, is_augmented=False):
        bias = batch.ptr[batch.edge_index_batch]
        edge_index_concatenated = batch.edge_index - bias
        batch.edge_feat = e_dense[batch.edge_index_batch, edge_index_concatenated[0, :], edge_index_concatenated[1, :], :]
        batch.computed_edge_feat = True

        if is_augmented:
            bias = batch.ptr[batch.edge_index_aug_batch]
            edge_index_concatenated = batch.edge_index_aug - bias
            batch.edge_feat_aug = e_dense[batch.edge_index_aug_batch, edge_index_concatenated[0, :], edge_index_concatenated[1, :], :]
            batch.computed_edge_feat_aug = True


    def pre_emb_forward(self, batch):
        batch.x = self.pre_emb_layer(batch.x)
        return batch


class GNN_Module(nn.Module):
    def __init__(self, config, type, pre_emb_layer=None, is_augmented=False):
        super(GNN_Module, self).__init__()
        self.config = config
        self.type = type
        self.edge_feat_dim = config.model.edge_feat_dim
        # self.pre_emb_layer = pre_emb_layer if pre_emb_layer else \
        #     nn.LazyLinear(out_features=config.model.embedding_dim)
        self.pre_emb_layer = pre_emb_layer

        if type == 'GATgran':
            gnn_ = GAT_GRAN(
                msg_dim=config.model.hidden_dim,
                node_state_dim=config.model.hidden_dim,
                edge_attr_dim=config.model.edge_feat_dim,
                num_prop=config.model.num_GNN_prop,
                num_layer=config.model.num_GNN_layers,
                has_attention=config.model.has_attention)

        elif type == 'GPSModel':
            gnn_ = GPSModel(config, pre_emb_layer=pre_emb_layer)

        elif type == 'PPGN':
            gnn_ = PPGNModel(config, pre_emb_layer=pre_emb_layer, is_augmented=is_augmented)

        self.GNN_model = gnn_

    def forward(self, batch: Batch, G_parent:Batch=None, parent_features=None, is_aug_part = False):
        if self.type == 'GATgran':
            node_feat, edge_index, edge_feat = batch.x, batch.edge_index, batch.edge_feat
            if not parent_features is None:
                node_feat = self.concat_parent_feat(batch, G_parent, parent_features)
            node_feat = self.pre_emb_layer(node_feat)
            batch.x = self.GNN_model(node_feat, edge_index, edge_feat)

        elif self.type in ['GPSModel', 'PPGN']:
            # apply FeatureEncoders
            batch = self.GNN_model.encoder(batch, is_aug_part=is_aug_part)

            # append the parent features and graph
            if not parent_features is None:
                batch.x = self.concat_parent_feat(batch, G_parent, parent_features)

            batch = self.GNN_model.pre_emb_forward(batch)
            batch = self.GNN_model(batch)

        return batch

    def concat_parent_feat(self, batch, G_parent, parent_features):
        # concatenate the grandparent's node rep
        x_parent = G_parent.x[batch.parent_node_id]
        if not hasattr(batch, 'x') or batch.x is None:
            x = x_parent
        else:
            x = torch.cat([batch.x, x_parent], dim=-1)
        return x


class Node_Pre_Embedding(nn.Module):
    def __init__(self, dimension_reduce, embedding_dim, inp_dim):
        super(Node_Pre_Embedding, self).__init__()
        self.inp_dim = inp_dim
        self.to_embed = dimension_reduce

        if dimension_reduce:
            self.embedding_dim = embedding_dim
            self.node_embedding = nn.Sequential(
                nn.Linear(self.inp_dim, embedding_dim))
        else:
            self.node_embedding = nn.Identity()
            self.embedding_dim = None

    def forward(self, inp):
        if self.to_embed:
            # This is not working on apple silicon:https://github.com/pytorch/pytorch/issues/89624
            # use the following instead
            # inp = torch.cat([inp, torch.zeros(inp.shape[0], self.inp_dim - inp.shape[1]).to(inp)] , dim=-1)
            inp = F.pad(inp, (0, self.inp_dim - inp.shape[1], 0, 0),
                        mode='constant', value=0.0)
        return self.node_embedding(inp)


class GAT_GRAN(nn.Module):

    def __init__(self,
                 msg_dim,
                 node_state_dim,
                 edge_attr_dim,
                 num_prop=1,
                 num_layer=1,
                 has_attention=True,
                 att_hidden_dim=128,
                 has_residual=False,
                 has_graph_output=False,
                 output_hidden_dim=128,
                 graph_output_dim=None):
        super(GAT_GRAN, self).__init__()
        self.msg_dim = msg_dim
        self.node_state_dim = node_state_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.has_attention = has_attention
        self.has_residual = has_residual
        self.att_hidden_dim = att_hidden_dim
        self.has_graph_output = has_graph_output
        self.output_hidden_dim = output_hidden_dim
        self.graph_output_dim = graph_output_dim

        self.update_func = nn.ModuleList([
            nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
            for _ in range(self.num_layer)
        ])

        self.msg_func = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Linear(self.node_state_dim + self.edge_attr_dim,
                              self.msg_dim),
                    nn.ReLU(),
                    nn.Linear(self.msg_dim, self.msg_dim)
                ]) for _ in range(self.num_layer)
        ])

        if self.has_attention:
            self.att_head = nn.ModuleList([
                nn.Sequential(
                    *[
                        nn.Linear(self.node_state_dim + self.edge_attr_dim,
                                  self.att_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.att_hidden_dim, self.msg_dim),
                        nn.Sigmoid()
                    ]) for _ in range(self.num_layer)
            ])

        if self.has_graph_output:
            self.graph_output_head_att = nn.Sequential(*[
                nn.Linear(self.node_state_dim, self.output_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.output_hidden_dim, 1),
                nn.Sigmoid()
            ])

            self.graph_output_head = nn.Sequential(
                *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

    def _prop(self, state, edge, edge_feat, layer_idx=0):
        ### compute message
        state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        if self.edge_attr_dim > 0:
            edge_input = torch.cat([state_diff, edge_feat], dim=1)
        else:
            edge_input = state_diff

        msg = self.msg_func[layer_idx](edge_input)

        ### attention on messages
        if self.has_attention:
            att_weight = self.att_head[layer_idx](edge_input)
            msg = msg * att_weight

        ### aggregate message by sum
        state_msg = torch.zeros(state.shape[0], msg.shape[1], device=state.device, dtype=msg.dtype)
        scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
        state_msg = state_msg.scatter_add(0, scatter_idx, msg)

        ### state update
        if state_msg.device.type == 'cuda' and ((state.dtype == torch.bfloat16) or (state_msg.dtype == torch.bfloat16)):
            ### gru does not support bfloat16 on cuda
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled= True ):
                state = self.update_func[layer_idx](state_msg, state)
        else:
            state = self.update_func[layer_idx](state_msg, state)
        return state

    def forward(self, node_feat, edge_index, edge_feat, graph_idx=None):
        """
          N.B.: merge a batch of graphs as a single graph

          node_feat: N X D, node feature
          edge: 2 X M, edge indices
          edge_feat: M X D', edge feature
          graph_idx: N X 1, graph indices
        """

        edge = edge_index.t()
        state = node_feat
        prev_state = state
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)

            for jj in range(self.num_prop):
                state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

        if self.has_residual:
            state = state + prev_state

        if self.has_graph_output:
            num_graph = graph_idx.max() + 1
            node_att_weight = self.graph_output_head_att(state)
            node_output = self.graph_output_head(state)

            # weighted average
            reduce_output = torch.zeros(num_graph,
                                        node_output.shape[1], device=node_feat.device)
            reduce_output = reduce_output.scatter_add(0,
                                                      graph_idx.unsqueeze(1).expand(
                                                          -1, node_output.shape[1]),
                                                      node_output * node_att_weight)

            const = torch.zeros(num_graph, device=node_feat.device)
            const = const.scatter_add(
                0, graph_idx, torch.ones(node_output.shape[0], device=node_feat.device))

            reduce_output = reduce_output / const.view(-1, 1)

            return reduce_output
        else:
            return state

class GATgran_module(nn.Module):
    def __init__(self, config, type, edge_feat_dim=None):
        super(GATgran_module, self).__init__()

        if type == 'GATgran':
            gnn_ = GAT_GRAN(
                msg_dim         = config.model.hidden_dim,
                node_state_dim  = config.model.hidden_dim,
                edge_attr_dim   = config.model.edge_feat_dim,
                num_prop        = config.model.num_GNN_prop,
                num_layer       = config.model.num_GNN_layers,
                has_attention   = config.model.has_attention)
            self.add_module("GNN_ls", gnn_)
        else:
            raise NotImplementedError

    def forward(self, node_feat, edge_index, edge_feat):
        return self.GNN_ls(node_feat, edge_index, edge_feat)

