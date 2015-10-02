#===============================================================================
# Copyright (c) 2015, Max Zwiessele
#
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of paramax nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

import GPy
import copy
from GPy.core.parameterization.observable_array import ObsAr

class ArrayPlaceholder(ObsAr):
    def __init__(self, array):
        self._shape = array.shape
        self._min, self._max = array.min(), array.max()
    
    @property
    def shape(self):
        return self._shape
        
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
                                                    name=model.name,
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
                                              inference_method=copy.deepcopy(model.inference_method),
                                              name=model.name,
                                              )
        self.posterior = model.posterior
        self._log_likelihood = model.log_likelihood()
        
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