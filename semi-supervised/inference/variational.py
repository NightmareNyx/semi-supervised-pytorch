from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F

from utils import log_sum_exp, enumerate_discrete, repeat_list, batch_normalization
from inference.distributions import log_standard_categorical


class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """

    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def resample_list(self, l):
        return repeat_list(l, self.mc * self.iw)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """

    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)
        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)
        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        return torch.mean(U)


class HISVI(nn.Module):
    """
    Stochastic variational inference (SVI) for Heterogeneous and Missing data.
    Here the likelihood parameter is missing because it's incorporated in the HIDecoder.
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(HISVI, self).__init__()
        self.model = model
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None, miss_list=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)
            miss_list = miss_list.repeat(self.model.y_dim, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys).float()
        miss_list = self.sampler.resample(miss_list)

        # p(x|y,z) of observed and missing
        # samples of x from decoder
        # and their parameters
        log_p_x, log_p_x_missing, samples_x, params_x = self.model(xs, ys, miss_list)

        # Eq[log_p(x|z, y)]
        reconstruction_loss = torch.sum(log_p_x, 1, keepdim=True)
        # likelihood = -self.likelihood(samples_x, xs)  ## possibly with miss_mask?

        # p(y)
        y_prior = -log_standard_categorical(ys).unsqueeze(1)

        # Equivalent to -L(x, y)
        elbo = reconstruction_loss + y_prior - next(self.beta) * self.model.kl_divergence

        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x,a)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        return torch.mean(U)
