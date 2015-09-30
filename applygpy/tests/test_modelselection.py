'''
Created on 30 Sep 2015

@author: Max Zwiessele
'''
import unittest, numpy as np, pandas as pd  # @UnresolvedImport
import GPy, GPy.kern as kern
from applygpy.model_selection import cross_validate
from GPy.models.sparse_gp_regression import SparseGPRegression
from GPy.models.sparse_gp_classification import SparseGPClassification
from GPy.core.gp import GP
from GPy.likelihoods.gaussian import Gaussian
from GPy.inference.latent_function_inference.exact_gaussian_inference import ExactGaussianInference

class Test(unittest.TestCase):


    def setUp(self):
        np.random.seed(11111)
        self.X = np.linspace(-1, 1, 20)[:,None]
        k = GPy.kern.Matern32(1, lengthscale=1, variance=1)
        self.sim_model = 'Mat+Lin'
        self.mf = GPy.mappings.Linear(1, 1)
        self.mf[:] = .01
        self.mu = self.mf.f(self.X)
        self.Y = np.random.multivariate_normal(np.zeros(self.X.shape[0]), k.K(self.X))[:,None]
        self.mf.randomize()
        self.test_models = [
                            ['Mat+Lin', kern.Matern32(self.X.shape[1]) + kern.Linear(self.X.shape[1], variances=.01) + kern.Bias(self.X.shape[1])], 
                            ['Lin', kern.Linear(self.X.shape[1], variances=.01) + kern.Bias(self.X.shape[1])],
                            ] 
        self.verbose = True

    def testCrossval(self):
        def model_builder(X, Y, kernel):
            return GP(X, Y, kernel=kernel, likelihood=Gaussian(), mean_function=self.mf.copy(), inference_method=ExactGaussianInference())
        res = cross_validate(self.X, self.Y+self.mu, verbose=self.verbose)#, kernels_models=self.test_models)#, model_builder=model_builder)
        tmp = (res['error'] / res['test_size'])
        self.assertEqual(tmp.loc['RMSE'].mean().argmin(), self.sim_model)
        self.assertEqual(tmp.loc['log likelihood multivariate'].mean().argmax(), self.sim_model)

    def testCrossvalSparse(self):
        def model_builder(X, Y, kernel):
            m = SparseGPRegression(X, Y, kernel=kernel)
            m.Z.fix()
            return m
        res = cross_validate(self.X, self.Y, sparse=True, verbose=self.verbose, 
                             kernels_models=self.test_models, 
                             k=2,
                             #model_builder=model_builder
                             )
        tmp = (res['error'] / res['test_size'])
        self.assertEqual(tmp.loc['RMSE'].mean().argmin(), self.sim_model)
        self.assertEqual(tmp.loc['log likelihood multivariate'].mean().argmax(), self.sim_model)

    def testCrossvalClass(self):
        res = cross_validate(self.X, self.Y>self.Y.mean(), verbose=self.verbose, 
                             kernels_models=self.test_models,
                             #, model_builder=model_builder
                             k=2,
                             )
        tmp = (res['error'] / res['test_size'])
        self.assertEqual(tmp.loc['RMSE'].mean().argmin(), self.sim_model)
        
    def testCrossvalSparseClass(self):
        res = cross_validate(self.X, self.Y>self.Y.mean(), sparse=True, verbose=self.verbose, 
                             kernels_models=self.test_models, 
                             #model_builder=model_builder,
                             k=2,
                             )
        tmp = (res['error'] / res['test_size'])
        self.assertEqual(tmp.loc['RMSE'].mean().argmin(), self.sim_model)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCrossval']
    unittest.main()