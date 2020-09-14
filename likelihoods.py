"""
Implementation of Generalized Multi-Output Gaussian Process Censored Regression 
------------------------------------------------------

This file contains the likelihood specifications. In particular, we implement:
(1) Gaussian (NCGP, MONCGP)
(2) Censored-Gaussian with homoscedastic noise (CGP, MOCGP)
(3) Censored-Gaussian with heteroscedastic noise (HCGP, HMOCGP)
(4) Poisson (NCGP)
(5) Censored-Poisson (CGP, MOCGP)
(6) Negative Binomial (NCGP)
(7) Censored-Negative Binomial with homoscedastic noise (CGP)
(8) Censored-Negative Binomial with heteroscedastic noise (HMOCGP)

Requirements:
- torch==1.4.0
- pyro-ppl==1.3.1
- tensorflow==2.3.0
- tensorflow_probability==0.11.0
"""

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam, pyro_method

from distributions import PyroCensoredNormal
from distributions import PyroCensoredPoison

class Gaussian(Likelihood):
    def __init__(self, variance=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, f_loc, f_var, y=None):
        y_dist = dist.Normal(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var, self.variance.sqrt())
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

    
class CensoredHomoscedGaussian(Likelihood):
    
    def __init__(self, variance=None, censoring=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        self.censoring = censoring

    def forward(self, f_loc, f_var, y=None):
        y_dist = PyroCensoredNormal(loc=f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var, scale=self.variance.sqrt(), censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

    
class CensoredHeteroscedGaussian(Likelihood):
    
    def __init__(self, variance=None, censoring=None):
        super().__init__()
        
        self.censoring = censoring
        self.softplus = torch.nn.Softplus()
        
    def forward(self, f_loc, f_var, g_loc, g_var, y=None):
        scale = g_loc + torch.randn(g_loc.size(), device=g_loc.device)*g_var
        y_dist = PyroCensoredNormal(loc=f_loc + torch.randn(f_loc.size(), device=f_loc.device)*f_var, scale=self.softplus(scale), censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
    
class Poisson(Likelihood):
    
    def __init__(self, variance=None, response_function=None):
        super().__init__()
        
        self.response_function = torch.exp if response_function is None else response_function
        
    def forward(self, f_loc, f_var, y=None):
        f_res = self.response_function(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var)

        y_dist = dist.Poisson(f_res)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
    
class CensoredPoisson(Likelihood):
    
    def __init__(self, variance=None, censoring=None, response_function=None):
        super().__init__()
        
        self.response_function = torch.exp if response_function is None else response_function
        self.censoring = censoring
        
    def forward(self, f_loc, f_var, y=None):
        f_res = self.response_function(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var)

        y_dist = PyroCensoredPoison(rate=f_res, censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)
    
    
class NegBinomial(Likelihood):
    
    def __init__(self, variance=None, response_function=None):
        super().__init__()
        
        self.response_function = torch.nn.Softplus()
        
    def forward(self, f_loc, f_var, g_loc, g_var, y=None):
        mu = self.response_function(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var)
        alpha = self.response_function(g_loc + torch.randn(g_loc.dim(), device=g_loc.device)*g_var)
        y_dist = PyroNegBinomial(mu, alpha)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-mu.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

    
class CensoredNegBinomial(Likelihood):
    
    def __init__(self, variance=None, censoring=None, response_function=None):
        super().__init__()
        self.response_function = torch.nn.Softplus()
        self.censoring = censoring
        
    def forward(self, f_loc, f_var, g_loc, g_var, y=None):
        mu = self.response_function(f_loc + torch.randn(f_loc.dim(), device=f_loc.device)*f_var)
        alpha = self.response_function(g_loc + torch.randn(g_loc.dim(), device=g_loc.device)*g_var)
        y_dist = PyroCensoredNegBinomial(mu, alpha, censoring=self.censoring)
        self.y_dist = y_dist
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-mu.dim()]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)