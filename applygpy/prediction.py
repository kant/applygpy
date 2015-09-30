'''
Created on 29 Sep 2015

@author: maxz
'''

import GPy, numpy as np
import copy
from GPy.core.parameterization.observable_array import ObsAr

class ArrayPlaceholder(ObsAr):
    def __init__(self, array):
        self.shape = array.shape
        self._min, self._max = array.min(), array.max()
        self.ndim = array.ndim
        
        
    def min(self):
        return self._min
    
    def max(self):
        return self._max

class PredictionModelSparse(GPy.core.SparseGP):
    
    def __init__(self, model):
        super(PredictionModelSparse, self).__init__(ArrayPlaceholder(model.X), ArrayPlaceholder(model.Y), 
                                                    likelihood=model.likelihood.copy(), 
                                                    kernel=model.kern.copy(), Z=model.Z.copy(), 
                                                    inference_method=copy.deepcopy(model.inference_method),
                                                    mean_function=(model.mean_function.copy() if model.mean_function is not None else None),
                                                    )
        self.posterior = model.posterior
        self._log_likelihood = model.log_likelihood()
        
    def log_likelihood(self):
        return self._log_likelihood
    
    def parameters_changed(self):
        # print "Immutable, not changing anything"
        pass

    def plot(self, plot_limits=None, which_data_rows='all', 
        which_data_ycols='all', fixed_inputs=[], 
        levels=20, samples=0, fignum=None, ax=None, resolution=None, 
        plot_raw=False, linecol=None, fillcol=None, Y_metadata=None, 
        data_symbol='kx', predict_kw=None, plot_training_data=False, samples_y=0, apply_link=False):
        if plot_training_data:
            plot_training_data = False
            print("Training data not saved, continuing without data plotting.")
        return super(PredictionModelSparse, self).plot(plot_limits=plot_limits, which_data_rows=which_data_rows, which_data_ycols=which_data_ycols, fixed_inputs=fixed_inputs, levels=levels, samples=samples, fignum=fignum, ax=ax, resolution=resolution, plot_raw=plot_raw, linecol=linecol, fillcol=fillcol, Y_metadata=Y_metadata, data_symbol=data_symbol, predict_kw=predict_kw, plot_training_data=plot_training_data, samples_y=samples_y, apply_link=apply_link)

    def plot_data(self, which_data_rows='all', 
        which_data_ycols='all', visible_dims=None, 
        fignum=None, ax=None, data_symbol='kx'):
        print("Data has been deleted, not plotting training data")

class PredictionModel(GPy.core.GP):

    def __init__(self, model):
        super(PredictionModel, self).__init__(model.X.copy(), ArrayPlaceholder(model.Y), 
                                              kernel=model.kern.copy(), likelihood=model.likelihood.copy(), 
                                              mean_function=(model.mean_function.copy() if model.mean_function is not None else None),
                                              inference_method=copy.deepcopy(model.inference_method)
                                              )
        self.posterior = model.posterior
        self._log_likelihood = model.log_likelihood()
        self.name = model.name
        
    def log_likelihood(self):
        return self._log_likelihood
    
    def parameters_changed(self):
        # "Immutable, not changing anything"
        pass
    
    def plot_data(self, which_data_rows='all', 
        which_data_ycols='all', visible_dims=None, 
        fignum=None, ax=None, data_symbol='kx'):
        print("Data has been deleted, not plotting training data")
        
    def plot(self, plot_limits=None, which_data_rows='all', 
        which_data_ycols='all', fixed_inputs=[], 
        levels=20, samples=0, fignum=None, ax=None, resolution=None, 
        plot_raw=False, linecol=None, fillcol=None, Y_metadata=None,
        data_symbol='kx', predict_kw=None, plot_training_data=False, samples_y=0, apply_link=False):
        if plot_training_data:
            plot_training_data = False
            print("Training data not saved, continuing without data plotting.")
        return super(PredictionModel, self).plot(plot_limits=plot_limits, which_data_rows=which_data_rows, which_data_ycols=which_data_ycols, fixed_inputs=fixed_inputs, levels=levels, samples=samples, fignum=fignum, ax=ax, resolution=resolution, plot_raw=plot_raw, linecol=linecol, fillcol=fillcol, Y_metadata=Y_metadata, data_symbol=data_symbol, predict_kw=predict_kw, plot_training_data=plot_training_data, samples_y=samples_y, apply_link=apply_link)