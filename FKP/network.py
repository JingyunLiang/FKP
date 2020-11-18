"""
network of FKP
based on NICE and RealNVP
"""

import torch
import torch.nn as nn
import torch.distributions as D
import copy


class KernelPrior(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, kernel_size=0, alpha=0, normalization=0,
                 cond_label_size=None, batch_norm=True):
        super().__init__()

        # parameters of kernel pre-processing
        self.register_buffer('kernel_size', torch.ones(1)*kernel_size)
        self.register_buffer('alpha', torch.ones(1)*alpha)
        self.register_buffer('normalization', torch.ones(1)*normalization)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask  # like permutation, though a waste of parameters in the first layer
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        # log_prob(u) is always negative, sum_log_abs_det_jacobians mostly negative -> log_prob is always negative
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return self.base_dist.log_prob(u).sum(1) + sum_log_abs_det_jacobians, u  # should all be summation

    def post_process(self, x):
        # inverse process of pre_process in dataloader
        x = x.view(x.shape[0], 1, int(self.kernel_size), int(self.kernel_size))
        x = ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha))
        x = x * self.normalization
        return x


class LinearMaskedCoupling(nn.Module):
    """ Coupling Layers """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        # stored in state_dict, but not trained & not returned by nn.parameters(); similar purpose as nn.Parameter objects
        # this is because tensors won't be saved in state_dict and won't be pushed to the device
        self.register_buffer('mask', mask)  # 0,1,0,1

        # scale function
        # for conditional version, just concat label as the input into the network (conditional way of SRMD)
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]

        self.s_net = nn.Sequential(*s_net)

        # translation function, the same structure
        self.t_net = copy.deepcopy(self.s_net)

        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        log_s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(
            -log_s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = (- (1 - self.mask) * log_s).sum(
            1)  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        log_s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))  # log of scale, log(s)
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))  # translation, t
        x = mu + (1 - self.mask) * (u * log_s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = ((1 - self.mask) * log_s).sum(1)  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = (self.log_gamma - 0.5 * torch.log(var + self.eps)).sum()

        return y, log_abs_det_jacobian

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = (0.5 * torch.log(var + self.eps) - self.log_gamma).sum()

        return x, log_abs_det_jacobian


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians
