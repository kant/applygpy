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

'''
Created on 29 Sep 2015

Script for running model selection for a few models using
k-fold cross-validation. 

@author: Max Zwiessele
'''
from __future__ import print_function
from numpy.linalg.linalg import LinAlgError
import numpy as np, scipy as sp

def log_likelihood_multivariate(Y_pred, Y_var, Y_test):
    from scipy import stats  # @UnresolvedImport
    return stats.multivariate_normal.logpdf(Y_test.flatten(), Y_pred.flatten(), Y_var).sum()

def log_likelihood_univariate(mu, var, Y_test):
    from scipy import stats  # @UnresolvedImport
    return stats.norm.logpdf(Y_test.flatten(), mu.flatten(), var).sum()

def RMSE(Y_pred, Y_test):
    return np.sqrt(((Y_pred - Y_test)**2).mean())

def standard_models(X):
    """
    Return kernels for model selection
    """
    from GPy import kern
    return [
            ['Mat+Lin', kern.Matern32(X.shape[1]) + kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])], 
            ['Exp+Lin', kern.Exponential(X.shape[1]) + kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])], 
            ['RBF+Lin', kern.RBF(X.shape[1]) + kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])], 
            ['Lin', kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])],
            ]

def run_model(model_builder, X, Y, model, verbose=True):
    """
    Optimize a model class with its model
    """
    m = model_builder(X, Y, model)
    num_opt_rounds = 3
    for i in range(num_opt_rounds):
        old_p = m.param_array.copy()
        try:
            if verbose:
                print('Optimization Round: {}/{}'.format(i+1, num_opt_rounds))
            if hasattr(m, 'Z'):
                m.Z.randomize()
            m.optimize(messages=verbose)
        except LinAlgError:
            m[:] = old_p
    return m


def make_model_builder(Y, num_inducing=None, name=None, sparse=False):
    """
    This is a helper for creating several models. It
    returns a function which takes data and makes a GPy
    model out of it, with the specified properties.
    
    Y: the data to make the model builder for (for testing for classification or not)
    num_inducing: The number of inducing inputs to use, standard is min(500, Y.shape[0]/4)
    name: name of the model
    sparse: if the model is sparse or not, it will be sparse if there is more then a 1000 data points
    """
    from GPy.models import GPClassification, GPRegression, SparseGPClassification, SparseGPRegression
    if num_inducing is None:
        num_inducing = min(500, int(Y.shape[0]/4))
    if np.all(np.unique(Y) == [0, 1]):
        def model_builder(X, Y, kernel):
            if X.shape[0] < 1000 and not sparse:
                tmp = GPClassification(X, Y, kernel=kernel)
            else:
                tmp = SparseGPClassification(X, Y, num_inducing=num_inducing, kernel=kernel)
            if name is not None:
                tmp.name = name
            return tmp
    else:
        def model_builder(X, Y, kernel):
            if X.shape[0] < 1000 and not sparse:
                tmp = GPRegression(X, Y, kernel=kernel)
            else:
                tmp = SparseGPRegression(X, Y, num_inducing=num_inducing, kernel=kernel)
            if name is not None:
                tmp.name = name
            return tmp
    return model_builder


def _add_error(l, idx, test_size, name, errname, err):
    l.append([name, idx, test_size, errname, err]) 

def cross_validate(X, Y, num_inducing=None, sparse=False, k=5, kernels_models=None, verbose=True, model_builder=None):
    """
    Run a k fold cross validation on the data with 
    the models defined in kernels_models. Standard 
    is the models from standard_models(X). The 
    kernels_models need to be supplied by 
    a list of [name, GPy.kern.Kern] pairs for the 
    model selection to work.
    
    X: inputs
    Y: outputs
    num_inducing: if sparse, how many inducing inputs to use, standard is min(500, Y.shape[0]/4)
    sparse: use sparse gp?
    k: k-fold crossvalidation
    kernels_models: list of [name, GPy.kern.Kern] pairs, models to test
    verbose: print messages of cross validation progess
    """
    import pandas as pd  # @UnresolvedImport
    if np.all(np.unique(Y) == [0, 1]):
        from sklearn.cross_validation import StratifiedKFold  # @UnresolvedImport
        assert Y.shape[1] == 1, 'classification only in one dimension for now'
        kval = StratifiedKFold(Y.flatten(), k, shuffle=True)
    else:
        from sklearn.cross_validation import KFold  # @UnresolvedImport
        kval = KFold(Y.shape[0], k, shuffle=True)
    res = []
    if kernels_models is None:
        kernels_models = standard_models(X)
    for model in kernels_models:
        name, kernel = model[0], model[1]
        if verbose:
            print("Running model:", name)
        if model_builder is None:
            model_builder = make_model_builder(Y, num_inducing, name, sparse)
        _i = 0
        for idx in kval:
            if verbose:
                print("Fold", _i+1)
            train_index, test_index = idx[0], idx[1]
            if test_index.size == 0:
                break
            if sp.sparse.issparse(X):  # @UndefinedVariable
                X_train, Y_train = X[train_index].toarray(), Y[train_index]
                X_test = X[test_index].toarray()
            else:
                X_train, Y_train = X[train_index], Y[train_index]
                X_test = X[test_index]
            Y_test = Y[test_index]
            m = run_model(model_builder, X_train, Y_train, kernel, verbose=verbose)
            try:
                test_mu, test_var = m.predict(X_test, full_cov=True)
                _add_error(res, _i, test_index.size, name, 'log likelihood multivariate', log_likelihood_multivariate(test_mu, test_var, Y_test))
                _add_error(res, _i, test_index.size, name, 'log_likelihood univariate', log_likelihood_univariate(test_mu, test_var, Y_test))
            except MemoryError:
                test_mu, test_var = m.predict(X_test, full_cov=False)
                _add_error(res, _i, test_index.size, name, 'log_likelihood univariate', log_likelihood_univariate(test_mu, test_var, Y_test))
            except:
                _add_error(res, _i, test_index.size, name, 'log_likelihood univariate', np.nan)
            try:
                _add_error(res, _i, test_index.size, name, 'RMSE', RMSE(test_mu, Y_test))
            except:
                _add_error(res, _i, test_index.size, name, 'RMSE', np.nan)
            _i += 1
    res = pd.DataFrame(res, columns=['model_name', 'fold', 'test_size', 'error_name', 'error']).set_index(['model_name', 'error_name', 'fold']).unstack(0)
    
    return res