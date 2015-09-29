'''
Created on 29 Sep 2015

Script for running model selection for a few models using
k-fold cross-validation. 

@author: Max Zwiessele
'''
from numpy.linalg.linalg import LinAlgError
import numpy as np

def log_likelihood_multivariate(Y_pred, Y_var, Y_test):
    from scipy import stats  # @UnresolvedImport
    return stats.multivariate_normal.logpdf(Y_test.flatten(), Y_pred.flatten(), Y_var).sum()

def log_likelihood_univariate(mu, var, Y_test):
    from scipy import stats  # @UnresolvedImport
    return stats.norm.logpdf(Y_test.flatten(), mu.flatten(), var).sum()

def RMSE(Y_pred, Y_test):
    return np.sqrt(((Y_pred - Y_test)**2).mean())

def kernel_models(X):
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

def run_model(model_builder, X, Y, model):
    """
    Optimize a model class with its model
    """
    m = model_builder(X, Y, model)
    for _ in range(3):
        old_p = m.param_array.copy()
        try:
            if hasattr(m, 'Z'):
                m.Z.randomize()
            m.optimize(messages=1)
        except LinAlgError:
            m[:] = old_p
    return m


def make_model_builder(Y, num_inducing=None, name=None):
    from GPy.models import GPClassification, GPRegression, SparseGPClassification, SparseGPRegression
    if num_inducing is None:
        num_inducing = min(500, int(Y.shape[0]/4))
    if np.all(np.unique(Y) == [0, 1]):
        def model_builder(X, Y, kernel):
            if X.shape[0] < 1000:
                tmp = GPClassification(X, Y, kernel=kernel)
            else:
                tmp = SparseGPClassification(X, Y, num_inducing=num_inducing, kernel=kernel)
            if name is not None:
                tmp.name = name
            return tmp
    else:
        def model_builder(X, Y, kernel):
            if X.shape[0] < 1000:
                tmp = GPRegression(X, Y, kernel=kernel)
            else:
                tmp = SparseGPRegression(X, Y, num_inducing=num_inducing, kernel=kernel)
            if name is not None:
                tmp.name = name
            return tmp
    return model_builder


def _add_error(l, idx, test_size, name, errname, err):
    l.append([name, idx, test_size, errname, err]) 

def cross_validate(X, Y, num_inducing=None):
    import pandas as pd  # @UnresolvedImport
    if np.all(np.unique(Y) == [0, 1]):
        from sklearn.cross_validation import StratifiedKFold  # @UnresolvedImport
        kval = StratifiedKFold(Y, 5)
    else:
        from sklearn.cross_validation import KFold  # @UnresolvedImport
        kval = KFold(Y.shape[0], 5)
    res = []
    for model in kernel_models(X):
        name, kernel = model[0], model[1]
        model_builder = make_model_builder(Y, num_inducing, name)
        _i = 0
        for idx in kval:
            train_index, test_index = idx[0], idx[1]
            if test_index.size == 0:
                break
            X_train, Y_train = X[train_index].toarray(), Y[train_index][:,None]
            Y_test = Y[test_index][:,None]
            m = run_model(model_builder, X_train, Y_train, kernel)
            try:
                test_mu, test_var = m.predict(X[test_index].toarray(), full_cov=True)
                _add_error(res, _i, test_index.size, name, 'log likelihood multivariate', log_likelihood_multivariate(test_mu, test_var, Y_test))
                _add_error(res, _i, test_index.size, name, 'RMSE', RMSE(test_mu, Y_test))
            except MemoryError:
                
                test_mu, test_var = m.predict(X[test_index].toarray(), full_cov=False)
                _add_error(res, _i, test_index.size, name, 'log_likelihood', log_likelihood_univariate(test_mu, test_var, Y_test))
                _add_error(res, _i, test_index.size, name, 'RMSE', RMSE(test_mu, Y_test))
            except:
                _add_error(res, _i, test_index.size, name, 'log_likelihood', np.nan)
                _add_error(res, _i, test_index.size, name, 'RMSE', np.nan)
            _i += 1
    res = pd.DataFrame(res, columns=['model_name', 'fold', 'test_size', 'error_name', 'error']).set_index(['model_name', 'error_name', 'fold']).unstack(0)
    
    return res