"""
Implementation of Generalized Multi-Output Gaussian Process Censored Regression 
------------------------------------------------------

This file contains the models specifications. In particular, we implement:
(1) Variational GP (NCGP, MONCGP, CGP, MOCGP):
    Gaussian Process model able to deal with arbitrary likelihood functions through SVI.

(2) Heteroscedastic Variational GP (HCGP, HMOCGP):
    Extend the VariationalGP by introducing a second GP prior over the variance parameter.

Requirements:
- torch==1.4.0
- pyro-ppl==1.3.1
- tensorflow==2.3.0
- tensorflow_probability==0.11.0
"""

import torch
from torch.distributions import constraints
from torch.nn import Parameter
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroParam, pyro_method
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.distributions.util import eye_like


class VariationalGP(GPModel):
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6, use_cuda=False):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood
        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        f_loc = self.X.new_zeros(self.latent_shape + (N,))
        self.f_loc = Parameter(f_loc)

        identity = eye_like(self.X, N)
        f_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.f_scale_tril = PyroParam(f_scale_tril, constraints.lower_cholesky)

        self.whiten = whiten
        self._sample_latent = True
        if use_cuda:
            self.cuda()
            
    @pyro_method
    def model(self):
        self.set_mode("model")
        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()

        zero_loc = self.X.new_zeros(self.f_loc.shape)
        if self.whiten:
            identity = eye_like(self.X, N)
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=identity)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = Lff.matmul(self.f_scale_tril)
            f_loc = Lff.matmul(self.f_loc.unsqueeze(-1)).squeeze(-1)
        else:
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = self.f_scale_tril
            f_loc = self.f_loc

        f_loc = f_loc + self.mean_function(self.X)
        f_var = f_scale_tril.pow(2).sum(dim=-1)
        if self.y is None:
            return f_loc, f_var
        else:
            return self.likelihood(f_loc, f_var, self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

        pyro.sample(self._pyro_get_fullname("f"),
                    dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril)
                        .to_event(self.f_loc.dim()-1))

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        .. math:: p(f^* \mid X_{new}, X, y, k, f_{loc}, f_{scale\_tril})
            = \mathcal{N}(loc, cov).
        .. note:: Variational parameters ``f_loc``, ``f_scale_tril``, together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = conditional(Xnew, self.X, self.kernel, self.f_loc, self.f_scale_tril,
                               full_cov=full_cov, whiten=self.whiten, jitter=self.jitter)
        return loc + self.mean_function(Xnew), cov

class HeteroscedVariationalGP(GPModel):
    def __init__(self, X, y, kernel, kernel_g, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6, use_cuda=False):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood
        self.kernel_g = kernel_g
        
        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        f_loc = self.X.new_zeros(self.latent_shape + (N,))
        g_loc = self.X.new_zeros(self.latent_shape + (N,))
        self.f_loc = Parameter(f_loc)
        self.g_loc = PyroParam(g_loc)

        identity = eye_like(self.X, N)
        f_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.f_scale_tril = PyroParam(f_scale_tril, constraints.lower_cholesky)
        identity_g = eye_like(self.X, N)
        g_scale_tril = identity_g.repeat(self.latent_shape + (1, 1))
        self.g_scale_tril = PyroParam(g_scale_tril, constraints.lower_cholesky)

        self.whiten = whiten
        self._sample_latent = True
        
        if use_cuda:
            self.cuda()

    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky()
        Kgg = self.kernel_g(self.X).contiguous()
        Kgg.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lgg = Kgg.cholesky()

        zero_loc = self.X.new_zeros(self.f_loc.shape)
        if self.whiten:
            identity = eye_like(self.X, N)
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=identity)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = Lff.matmul(self.f_scale_tril)
            f_loc = Lff.matmul(self.f_loc.unsqueeze(-1)).squeeze(-1)
        else:
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = self.f_scale_tril
            f_loc = self.f_loc
            pyro.sample(self._pyro_get_fullname("g"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Lgg)
                            .to_event(zero_loc.dim() - 1))
            g_scale_tril = self.g_scale_tril
            g_loc = self.g_loc

        f_loc = f_loc + self.mean_function(self.X)
        f_var = f_scale_tril.pow(2).sum(dim=-1)
        g_var = g_scale_tril.pow(2).sum(dim=-1)
        if self.y is None:
            return f_loc, Kff, g_loc, Kgg
        else:
            return self.likelihood(f_loc, f_var, g_loc, g_var, self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

        pyro.sample(self._pyro_get_fullname("f"),
                    dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril)
                        .to_event(self.f_loc.dim()-1))
        pyro.sample(self._pyro_get_fullname("g"),
                    dist.MultivariateNormal(self.g_loc, scale_tril=self.g_scale_tril)
                        .to_event(self.g_loc.dim()-1))

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}` for both mean and variance
        parameters:
        .. math:: p(f^* \mid X_{new}, X, y, k, f_{loc}, f_{scale\_tril})
            = \mathcal{N}(loc, cov), p(g^* \mid X_{new}, X, y, k, g_{loc},
            g_{scale\_tril}) = \mathcal{N}(loc, cov).
        .. note:: Variational parameters ``f_loc``, ``f_scale_tril``, ``g_loc``, 
            ``g_scale_tril``, together with kernel's parameters have been learned 
            from a training procedure (MCMC or SVI).
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
            and :math:`p(g^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = conditional(Xnew, self.X, self.kernel, self.f_loc, self.f_scale_tril,
                               full_cov=full_cov, whiten=self.whiten, jitter=self.jitter)
        
        loc_g, cov_g = conditional(Xnew, self.X, self.kernel_g, self.g_loc, self.g_scale_tril,
                                   full_cov=full_cov, whiten=self.whiten, jitter=self.jitter)
        
        return loc + self.mean_function(Xnew), cov, loc_g, cov_g