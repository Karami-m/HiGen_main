import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual, act='relu',
                 equivstable_pe=False, aug_model=False, **kwargs):
        super().__init__(**kwargs)
        self._in_dim = in_dim
        self.activation = register.act_dict[act]
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.LazyLinear(out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.aug_model = aug_model
        if aug_model:
            self.A2 = pyg_nn.Linear(in_dim, out_dim, bias=True)
            self.B2 = pyg_nn.Linear(in_dim, out_dim, bias=True)
            self.C2 = nn.LazyLinear(out_dim, bias=True)
            self.D2 = pyg_nn.Linear(in_dim, out_dim, bias=True)
            self.E2 = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        if self.aug_model:
            return self.forward_aug(batch)
        else:
            return self.forward_base(batch)
    def forward_base(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        # update for the augmented edges
        to_model_aug_edges = False
        if not batch.get('edge_index_aug', None) is None and \
                not batch.get('edge_attr_aug', None) is None :
            to_model_aug_edges = True
        if to_model_aug_edges:
            edge_index = torch.cat( [edge_index,  batch.edge_index_aug], dim=1)
            # e_in_0 = batch.edge_attr
            # e_in_aug = batch.edge_attr_aug
            e = torch.cat(
                [torch.cat([batch.edge_attr, torch.zeros_like(batch.edge_attr_aug)], dim=0),
                 torch.cat([torch.zeros_like(batch.edge_attr), batch.edge_attr_aug], dim=0)], dim=1)

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        # if self.residual:
        #     x_in = x
        #     e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        edge_attr = e
        if to_model_aug_edges:
            edge_attr = e[:batch.edge_index.shape[1],:]
            edge_attr_aug = e[batch.edge_index.shape[1]:,:]

        if self.residual:
            batch.x = batch.x + x
            batch.edge_attr = batch.edge_attr + edge_attr
            if to_model_aug_edges:
                batch.edge_attr_aug = batch.edge_attr_aug + edge_attr_aug
        else:
            batch.x = x
            batch.edge_attr = edge_attr
            if to_model_aug_edges:
                batch.edge_attr_aug = edge_attr_aug

        return batch

    def forward_aug(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        to_model_aug_edges = False
        if not batch.get('edge_index_aug', None) is None and \
                not batch.get('edge_attr_aug', None) is None :
            to_model_aug_edges = True

        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)


        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        if to_model_aug_edges:
            e_in_aug, edge_index_aug = batch.edge_attr_aug, batch.edge_index_aug
            Ax = self.A2(x_in)
            Bx = self.B2(x_in)
            Ce = self.C2(e_in_aug)
            Dx = self.D2(x_in)
            Ex = self.E2(x_in)
            x_aug, e_aug = self.propagate(edge_index_aug,
                                          Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                                          e=e_in_aug, Ax=Ax,
                                          PE=pe_LapPE)
            x += x_aug
            e = torch.cat([e, e_aug], dim=0)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        edge_attr = e
        if to_model_aug_edges:
            edge_attr = e[:batch.edge_index.shape[1],:]
            edge_attr_aug = e[batch.edge_index.shape[1]:,:]

        if self.residual:
            batch.x = batch.x + x
            batch.edge_attr = batch.edge_attr + edge_attr
            if to_model_aug_edges:
                batch.edge_attr_aug = batch.edge_attr_aug + edge_attr_aug
        else:
            batch.x = x
            batch.edge_attr = edge_attr
            if to_model_aug_edges:
                batch.edge_attr_aug = edge_attr_aug

        return batch


    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


@register_layer('gatedgcnconv')
class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   act=layer_config.act,
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)
