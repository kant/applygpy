'''
Created on 29 Sep 2015

@author: maxz
'''

import GPy, numpy as np

class PredictionModel(GPy.core.SparseGP):
    
    def __init__(self, Z, kernel, posterior, name):
        self.Z = Z
        self.posterior = posterior
        super(PredictionModel, self).__init__(np.array([[]]), np.array[[]], Z, kernel=kernel)
    
    def log_likelihood(self):
        return self._log_likelihood
    
    def parameters_changed(self):
        print "Immutable, not changing anything"