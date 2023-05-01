import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np


class NormalizingFlowModel(nn.Module):

    def __init__(self, dim, flows):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        bsz, _ = x.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        bsz, _ = z.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x

    
class TemporalNormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows, dim):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self.dim = dim

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x[:,0:self.dim])
        log_px = prior_logprob + log_det
        px = torch.exp(log_px)
        return z, prior_logprob, log_det, px

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples, t):
        z = self.prior.sample((n_samples,))
        time = np.ones([n_samples,1])*t
        zz = torch.Tensor(np.concatenate((z,time),axis=1))
        x, _ = self.inverse(zz)
        return x

    def sample_px(self, x, t):
        #给定时间t求px
        n_samples, _ = x.shape
        time = np.ones([n_samples,1])*t
        xx = torch.Tensor(np.concatenate((x,time),axis=1))
        # print(xx.shape)
        z, prior_logprob, log_det, px = self.forward(xx)
        return px