# Copyright (c) 2015 Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
from GPy import Parameterized
from nose.tools import assert_list_equal
import numpy as np
from GPy.core.parameterization.observable_array import ObsAr
from var_dtc_fixed_cov import VarDTCFixedCov

class BGPLVM_PANAMA(BayesianGPLVMMiniBatch):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, fixed_covariates, fixed_cov_kernels, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, likelihood=None,
                 name='bayesian gplvm with fixed covariates', normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1):

        assert type(fixed_covariates) == dict, "Need a dictionary with (name, array) pairs, so that the covariates can be used as parameters and be shown in the printing."
        assert type(fixed_cov_kernels) == dict, "Need a dictionary with (name, kernel) pairs to identify the kernel for each covariate"
        assert_list_equal(sorted(fixed_cov_kernels.keys()), sorted(fixed_covariates.keys()))

        super(BGPLVM_PANAMA, self).__init__(Y, input_dim, X=X, X_variance=X_variance, init=init, num_inducing=num_inducing,
                 Z=Z, kernel=kernel, inference_method=VarDTCFixedCov(), likelihood=likelihood,
                 name=name, normalizer=normalizer,
                 missing_data=missing_data, stochastic=stochastic, batchsize=batchsize)

        self._log_marginal_likelihood = 0

        self.cov_kernels = fixed_cov_kernels
        self.fixed_covariates = fixed_covariates
        for i,[name, p] in zip(range(4, 4+len(self.fixed_covariates)), self.cov_kernels.iteritems()):
            self.link_parameter(Parameterized(name, [p]), i)
        self.covs_kernels = dict([(name,
                                   [ObsAr(self.fixed_covariates[name]), self.cov_kernels[name]])
                                  for name in self.cov_kernels.iterkeys()])
        #self.fixed_cov_trKs = {name:np.trace(k.K(self.fixed_covariates[name])) for name, k in self.cov_kernels}

    def _inner_parameters_changed(self, kern, X, Z, likelihood, Y, Y_metadata, Lm=None, dL_dKmm=None, subset_indices=None, fixed_covs_kerns=None, **kw):
        return super(BGPLVM_PANAMA, self)._inner_parameters_changed(kern, X, Z, likelihood, Y, Y_metadata, Lm=Lm, dL_dKmm=dL_dKmm, subset_indices=subset_indices, fixed_covs_kerns=self.covs_kernels, **kw)

    def parameters_changed(self):
        super(BGPLVM_PANAMA, self).parameters_changed()

        for name, [cov, k] in self.covs_kernels.iteritems():
            k.update_gradients_full(self.grad_dict['dL_dcovs'], cov)

    def log_likelihood(self):
        return self._log_marginal_likelihood


if __name__ == "__main__":
    from bgplvmpanama import *
    import GPy, itertools
    # simple test for bgplvm panama:
    n = 100
    d = 50
    q = 8

    X = np.random.normal(0, 1, (n, q))

    m = 20
    s = 500
    S = np.random.binomial(1, .2, (n, s))
    M = np.random.binomial(1, .8, (n, m))

    Ws = [np.random.normal(0,1,(q,d)),
          np.random.normal(0,1,(s,d)),
          np.random.normal(0,1,(m,d)),
          ]

    Y = sum(itertools.starmap(np.dot, zip([X,S,M], Ws)))

    covs = dict(SNPs=S,
                Meth=M,
                )
    kerns = dict(SNPs=GPy.kern.Linear(s),
                Meth=GPy.kern.Linear(m)
                 )

    m = BGPLVM_PANAMA(Y, q, covs, kerns, num_inducing=20, kernel=GPy.kern.Linear(q, ARD=True))


