'''
Created on 29 Sep 2015

@author: maxz
'''

import GPy, numpy as np

class PredictionModelSparse(GPy.core.SparseGP):
    
    def __init__(self, model):
        super(PredictionModelSparse, self).__init__(np.array([[]]), np.array([[]]), likelihood=model.likelihood.copy(), kernel=model.kern.copy(), Z=model.Z.copy(), inference_method=model.inference_method)
        self.posterior = model.posterior
        self._log_likelihood = model.log_likelihood()
        
    def log_likelihood(self):
        return self._log_likelihood
    
    def parameters_changed(self):
        # print "Immutable, not changing anything"
        pass
        
class PredictionModel(GPy.core.GP):
    
    def __init__(self, model):
        super(PredictionModel, self).__init__(model.X.copy(), np.zeros((model.Y.shape[0], 0)), likelihood=model.likelihood.copy(), kernel=model.kern.copy())
        self.posterior = model.posterior
        self._log_likelihood = model.log_likelihood()
        
    def log_likelihood(self):
        return self._log_likelihood
    
    def parameters_changed(self):
        # "Immutable, not changing anything"
        pass