"""
Implementation of Generalized Multi-Output Gaussian Process Censored Regression 
------------------------------------------------------

This file contains the distributions specifications. In particular, we implement:
(1) Censored-Gaussian (Section 2)
(2) Censored-Poisson (Section 3)
(3) Censored-Negative Binomial (Section 3) 

Requirements:
- torch==1.4.0
- pyro-ppl==1.3.1
- tensorflow==2.3.0
- tensorflow_probability==0.11.0
"""

import torch
import pyro
from torch.distributions.exp_family import ExponentialFamily
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions import constraints
from numbers import Number
import math
import tensorflow as tf
import tensorflow_probability as tfp

class PyroCensoredNormal(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, censoring, validate_args=None):
        
        self.loc, self.scale, self.censoring = broadcast_all(loc, scale, censoring)
        
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(censoring, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(PyroCensoredNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroCensoredNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.censoring = self.censoring
        super(PyroCensoredNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        self.threshold = self.censoring==1
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        log_prob = -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

        log_prob[self.threshold] = math.log(1 - self.cdf(value)[self.threshold] + 0.01) if isinstance(1 - self.cdf(value)[self.threshold] + 1e-6,
                                                                                                                       Number) else (1 - self.cdf(value)[self.threshold] + 1e-6).log()
        return log_prob

    def cdf(self, value):
        
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))
    
    
class PyroCensoredPoison(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'rate': constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def __init__(self, rate, censoring, validate_args=None):
        self.rate, self.censoring = broadcast_all(rate, censoring)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super(PyroCensoredPoison, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroCensoredPoison, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        new.censoring = self.censoring
        super(PyroCensoredPoison, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        self.threshold = self.censoring==1
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        value = value.float()
        log_prob = (rate.log() * value) - rate - (value + 1).lgamma()
        
        v_tf = tf.convert_to_tensor(value.numpy())
        r_tf = tf.convert_to_tensor(rate.detach().numpy())
        poison = tfp.distributions.Poisson(rate=r_tf)
        cdf_val = torch.tensor(poison.cdf(v_tf).numpy())
        log_prob[self.threshold] = math.log(1 - cdf_val[self.threshold] + 0.01) if isinstance(1 - cdf_val[self.threshold] + 1e-6, Number) else (1 - cdf_val[self.threshold] + 1e-6).log()
        return log_prob


class PyroNegBinomial(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'mu': constraints.positive, 'alpha': constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.alpha

    def __init__(self, mu, alpha, validate_args=None):
        self.mu, self.alpha = broadcast_all(mu, alpha)
        if isinstance(mu, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(PyroNegBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroNegBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        super(PyroNegBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.negative_binomial(self.mu.expand(shape), self.alpha.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, alpha, value = broadcast_all(self.mu, self.alpha, value)
        value = value.float()
        
        d = torch.distributions.negative_binomial.NegativeBinomial(total_count=1/alpha, logits=alpha*mu)
        return d.log_prob(value)

    
class PyroCensoredNegBinomial(ExponentialFamily, TorchDistributionMixin):
    
    arg_constraints = {'mu': constraints.positive, 'alpha': constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.alpha

    def __init__(self, mu, alpha, censoring=None, validate_args=None):
        self.mu, self.alpha, self.censoring = broadcast_all(mu, alpha, censoring)
        if isinstance(mu, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(PyroCensoredNegBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PyroCensoredNegBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.censoring = self.censoring
        super(PyroCensoredNegBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.negative_binomial(self.mu.expand(shape), self.alpha.expand(shape))

    def log_prob(self, value):
        self.threshold = self.censoring==1
        if self._validate_args:
            self._validate_sample(value)
        mu, alpha, value = broadcast_all(self.mu, self.alpha, value)
        value = value.float()
        log_prob = torch.distributions.negative_binomial.NegativeBinomial(total_count=1/alpha, logits=alpha*mu).log_prob(value)
        
        v_tf = tf.convert_to_tensor(value.numpy())
        mu_tf = tf.convert_to_tensor(mu.detach().numpy())
        alpha_tf = tf.convert_to_tensor(alpha.detach().numpy())
        negbin = tfp.distributions.NegativeBinomial(total_count=1/alpha_tf, logits=alpha_tf*mu_tf)
        cdf_val = torch.tensor(negbin.cdf(v_tf).numpy())
        log_prob[self.threshold] = math.log(1 - cdf_val[self.threshold] + 0.01) if isinstance(1 - cdf_val[self.threshold] + 1e-6, Number) else (1 - cdf_val[self.threshold] + 1e-6).log()
        return log_prob