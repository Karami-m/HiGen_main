import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.nn import *
EPS = np.finfo(np.float32).eps * 10

__all__ = ['get_nll_multinomial', 'get_nll_binomial', 'multinomial_sampler', 'binomial_sampler', 'get_mixture_nll']


def multinomial_logcoef(target, n_trials, index_scatter=None):
    coef = torch.zeros_like(n_trials)
    coef = -coef.scatter_add(0, index_scatter, src=torch.lgamma(target + 1.))
    coef += torch.lgamma(n_trials + 1.)
    return coef


def get_nll_multinomial(logit, target, n_trials, reduction, mask, mode='logit'):
    alpha = torch.lgamma(n_trials + 1.) - torch.lgamma(target + 1.).sum(dim=1)
    if mode == 'logit':
        logit_max = logit.max(dim=2, keepdim=True)[0]
        logit -= logit_max
        logit_exp = logit.exp() * mask.unsqueeze(1)
        nll = n_trials.unsqueeze(1) * torch.log(logit_exp.sum(dim=2) + EPS)
        nll -= (logit * target.unsqueeze(1)).sum(dim=2) + alpha.unsqueeze(1)

    elif mode == 'logitsigm':
        sigm = torch.clamp(torch.sigmoid(logit), min=torch.tensor(EPS, device=logit.device)) * mask.unsqueeze(1)
        sum_sigm = sigm.sum(dim=2, keepdim=True)
        nll = -alpha.unsqueeze(1)
        nll -= (target.unsqueeze(1) * (logit - torch.nn.Softplus()(logit) - torch.log(sum_sigm + EPS)) * mask.unsqueeze(1)).sum(dim=2)
    else:
        raise NotImplementedError

    if reduction == 'sum':
        return nll.sum(dim=0)
    elif reduction == 'mean':
        return nll.mean(dim=0)
    elif reduction == 'none':
        return nll


def get_nll_binomial(logit, target, n_trials, reduction, mode='logit'):
    alpha = -torch.lgamma(n_trials + 1.) + torch.lgamma(target + 1.) + torch.lgamma(n_trials - target + 1.)

    if mode == 'p':
        nll = -torch.log(logit + EPS) * target-(n_trials-target) * torch.log1p(-logit + EPS) + alpha
    elif mode == 'logp':
        nll = -logit * target - (n_trials - target) * torch.log1p(-logit.exp() + EPS) + alpha
    elif mode == 'logit':
        nll = n_trials * torch.relu(logit) - logit * target + \
              n_trials * torch.log1p(torch.exp(-torch.abs(logit))) + alpha

    if reduction == 'sum':
        return nll.sum(dim=0)
    elif reduction == 'mean':
        return nll.mean(dim=0)
    elif reduction == 'none':
        return nll


n_sample_best = 5
def binomial_sampler(total_count, probs, mode=True, n_sample=n_sample_best):
    p = torch.distributions.binomial.Binomial(total_count=total_count, probs=probs)
    if mode == True:
        return p.mode
    else:
        smpl = p.sample([n_sample])
        return smpl[torch.argmax(p.log_prob(smpl))]

def multinomial_sampler(total_count, probs, mode=True, n_sample=n_sample_best):
    if probs.numel() == 1:
        return total_count

    else:
        total_count = int(total_count.item())
        if total_count > 0:
            p = torch.distributions.multinomial.Multinomial(total_count=total_count, probs=probs)
            if mode == True:
                # Mode of multinomial is approximated by finding the maximum probability of many samples
                smpl = p.sample([10 * n_sample_best])
                return smpl[torch.argmax(p.log_prob(smpl))]

            else:
                smpl = p.sample([n_sample])
                return smpl[torch.argmax(p.log_prob(smpl))]

        else:
            return torch.zeros_like(probs)


def get_mixture_nll(nlls, log_alpha: Tensor, num_aug_edges, scatter_index,
                    temperature=1., alpha_level='edge', split_lengths=None) -> Tensor:
    num_graphs = nlls.shape[0]

    log_alpha_reduced = log_alpha
    if alpha_level == 'edge0':
        log_alpha_reduced = get_reduced_attr(log_alpha, scatter_index, num_graphs)
        log_alpha_reduced = log_alpha_reduced / (num_aug_edges.view(-1, 1) + EPS)
    elif alpha_level == 'edge':
        log_alpha_reduced = split_then_pad(log_alpha, split_lengths=split_lengths, value=.0)
        log_alpha_reduced = log_alpha_reduced.sum(dim=0) / (num_aug_edges.view(-1, 1) + EPS)

    log_alpha_reduced = F.log_softmax(log_alpha_reduced / temperature, -1)
    log_prob = -nlls + log_alpha_reduced
    nll_mixture = -torch.logsumexp(log_prob, dim=1)
    return nll_mixture
