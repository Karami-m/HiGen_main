import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
import torch_geometric.nn as pyg_nn


@register_node_encoder('StableLapPE')
class StableLapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, config, dim_emb=None, expand_x=False):
        super().__init__()

        pecfg = config.posenc_StableLapPE
        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        model_type = pecfg.model  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet', 'none']:
            raise ValueError(f"Unexpected PE model {model_type}")
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE
        self.stable_GCN = MSE_GCNLayer('mean')
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if expand_x:
            if dim_emb - dim_pe < 0:  # formerly 1, but you could have zero feature size
                raise ValueError(f"LapPE size {dim_pe} is too large for "
                                 f"desired embedding size of {dim_emb}.")

            dim_in = config.share.dim_in  # Expected original input node features dim
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        elif model_type == 'DeepSet':
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        else:
            self.pe_encoder = nn.Sequential(nn.Identity())

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals_st') and hasattr(batch, 'EigVecs_st')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_StableLapPE.enable' to True")
        EigVals = batch.EigVals_st
        EigVecs = batch.EigVecs_st

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = self.stable_GCN(EigVecs, batch.edge_index)

        pos_enc = torch.cat((pos_enc.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch


class MSE_GCNLayer(pyg_nn.conv.MessagePassing):
    """
    """
    def __init__(self, aggregator='mean', **kwargs):
        super().__init__(aggr=aggregator, **kwargs)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_i, x_j):
        sigma_ij = (x_i - x_j) ** 2
        return sigma_ij


