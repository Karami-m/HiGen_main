import numpy as np
import torch
from torch import nn, Tensor
import torch_geometric as pyg

EPS = np.finfo(np.float32).eps * 10

__all__ = ['Output_Model', 'split_then_pad', 'softmax_masked', 'log_softmax_masked', 'get_reduced_attr']

class Output_Model(nn.Module):

    def __init__(self, out_dim: int, hidden_dim: int, n_layers, gate, device,
                  model_type='simple'):
        super().__init__()
        self.model_type = model_type
        self.device = device

        _layers = [nn.LazyLinear(out_features=hidden_dim, device=self.device), gate]
        for i_lay in range(1, n_layers-1):
            _layers += [nn.Linear(hidden_dim, hidden_dim), gate]
        _layers += [nn.Linear(hidden_dim, out_dim)]
        self.model = nn.Sequential(*_layers)
        if model_type == 'simple':
            self.forward = self.forward_simple
        else:
            self.forward = self.forward_split

    def init_weights(self):
        for _layers in self.model:
            if type(_layers) == nn.Linear:
                torch.nn.init.xavier_normal_(_layers.weight, gain = 5./3.)
                _layers.bias.data.fill_(0.01)

    def forward_split(self, x: Tensor, split_lengths: Tensor) -> Tensor:
        """
        :param x:
        :param split_lengths:
        :return: first split the input to the sequence then apply the model and returns the outout sequence
        """
        split_lengths = list(split_lengths.cpu().numpy())
        x = split_then_pad(x, split_lengths, value=0.)
        output = self.model(x)
        return output

    def forward_simple(self, x: Tensor) -> Tensor:
        return self.model(x)


def split_then_pad(x, split_lengths, value):
    x_ls = torch.split(x, split_lengths, dim=0)
    return nn.utils.rnn.pad_sequence(x_ls, padding_value=value)


def log_softmax_masked(logit, mask, dim):
    logit_max = logit.max(dim=dim, keepdim=True)[0]
    logit = logit - logit_max
    logit_exp = logit.exp() * mask.unsqueeze(dim)
    return logit - torch.log(logit_exp.sum(dim=dim) + EPS)

def softmax_masked(logit, mask, dim):
    logit_max = logit.max(dim=dim, keepdim=True)[0]
    logit = logit - logit_max
    logit_exp = logit.exp() * mask
    return logit_exp/logit_exp.sum(dim=dim)

def get_reduced_attr(x, scatter_index=None, num_graphs=None, num_edges=None, get_average=False):
    if scatter_index is None and num_edges is None:
        raise ValueError("one of scatter_index or num_edges should be given")
    elif scatter_index is None:
        scatter_index = torch.repeat_interleave(torch.arange(len(num_edges), device=num_edges.device), num_edges)

    if get_average:
        return pyg.nn.global_mean_pool(x, batch=scatter_index)
    else:
        return pyg.nn.global_add_pool(x, batch=scatter_index)

    # dim_feat = x.shape[1]
    # x_reduced = torch.zeros(num_graphs, dim_feat, device=x.device)
    # x_reduced = x_reduced.scatter_add(0, scatter_index.unsqueeze(1).expand(-1, dim_feat), x)
    # if get_average:
    #     return x_reduced/(num_edges.view(-1, 1) + EPS)
    # else
    #     return x_reduced


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm
