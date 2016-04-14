'''
Created on 30 Sep 2015

@author: Max Zwiessele
'''
import unittest, numpy as np, GPy
from applygpy.prediction import PredictionModel, PredictionModelSparse,\
    ArrayPlaceholder
try:
    import cPickle as pickle
except ImportError:
    import pickle

class Test(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = np.random.normal(0, 1, (10, 2)), np.random.normal(0, 1, (10, 1))
        pass

    def testArrayPlaceholder(self):
        pl = ArrayPlaceholder(self.X)
        self.assertTupleEqual(pl.shape, self.X.shape)
        self.assertTupleEqual((pl.min(), pl.max()), (self.X.min(), self.X.max()))
        self.assertEqual(pl.ndim, self.X.ndim)

    def testFullGP(self):
        m = GPy.models.GPRegression(self.X, self.Y) 
        p = PredictionModel(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 3)
        self.assertEqual(p.size, 3)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 3)

        Xpred = np.random.normal(0, 1, (40, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)
        
        Xpred = np.random.normal(0, 1, (2, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)
    
    def testSparseGP(self):
        m = GPy.models.SparseGPRegression(self.X, self.Y) 
        p = PredictionModelSparse(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 23)
        self.assertEqual(p.size, 23)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 23)
        Xpred = np.random.normal(0, 1, (40, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)
        
        Xpred = np.random.normal(0, 1, (2, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

    def testSparseGPClassification(self):
        m = GPy.models.SparseGPClassification(self.X, self.Y<0) 
        p = PredictionModelSparse(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 22)
        self.assertEqual(p.size, 22)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 22)
        Xpred = np.random.normal(0, 1, (40, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

        Xpred = np.random.normal(0, 1, (2, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

    def testFullGPClassification(self):
        m = GPy.models.GPClassification(self.X, self.Y<0) 
        p = PredictionModel(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 2)
        self.assertEqual(p.size, 2)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 2)
        Xpred = np.random.normal(0, 1, (40, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

        Xpred = np.random.normal(0, 1, (2, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

    def testGPmeanf(self):
        m = GPy.models.GPRegression(self.X, self.Y, mean_function=GPy.mappings.Linear(2, 1)) 
        p = PredictionModel(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 5)
        self.assertEqual(p.size, 5)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 5)
        mu1, var1 = p.predict(m.X)
        mu2, var2 = pt.predict(m.X)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(m.X, full_cov=True)
        mu2, var2 = pt.predict(m.X, full_cov=True)
        np.testing.assert_allclose(var1, var2)

        Xpred = np.random.normal(0, 1, (2, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)

    def testPredPrint(self):
        m = GPy.models.GPClassification(self.X, self.Y<0, mean_function=GPy.mappings.Linear(2, 1)) 
        p = PredictionModel(m)
#===============================================================================
#         self.assertEqual('\n'.join(m.__str__(VT100=False).split('\n')[3:]), """Number of Parameters              : 4
# Number of Optimization Parameters : 4
# Updates                           : True
# Parameters:
#   gp_classification.  |  Value   |  Constraint  |  Prior  |  Tied to
#   linmap.A            |  (2, 1)  |              |         |         
#   rbf.variance        |     1.0  |     +ve      |         |         
#   rbf.lengthscale     |     1.0  |     +ve      |         |         """)
#===============================================================================
        self.assertEqual(m.__str__(VT100=False), p.__str__(VT100=False))
        self.assertTupleEqual(p.X.shape, self.X.shape)
        self.assertTupleEqual((p.X.min(),p.X.max()), (self.X.min(), self.X.max()))
        self.assertEqual(p.X.ndim, self.X.ndim)
        m = GPy.models.SparseGPClassification(self.X, self.Y<0) 
        p = PredictionModelSparse(m)
#===============================================================================
#         self.assertEqual('\n'.join(m.__str__(VT100=False).split('\n')[3:]), """Number of Parameters              : 22
# Number of Optimization Parameters : 22
# Updates                           : True
# Parameters:
#   SparseGPClassification.  |   Value   |  Constraint  |  Prior  |  Tied to
#   inducing inputs          |  (10, 2)  |              |         |         
#   rbf.variance             |      1.0  |     +ve      |         |         
#   rbf.lengthscale          |      1.0  |     +ve      |         |         """)
#===============================================================================
        self.assertEqual(m.__str__(VT100=False), p.__str__(VT100=False))
        self.assertTupleEqual(p.X.shape, self.X.shape)
        self.assertTupleEqual((p.X.min(),p.X.max()), (self.X.min(), self.X.max()))
        self.assertEqual(p.X.ndim, self.X.ndim)

    def testGPClassmeanf(self):
        m = GPy.models.GPClassification(self.X, self.Y<0, mean_function=GPy.mappings.Linear(2, 1)) 
        p = PredictionModel(m)
        self.assert_(m.checkgrad())
        self.assertEqual(m.size, 4)
        self.assertEqual(p.size, 4)
        ptdata = pickle.dumps(p)
        pt = pickle.loads(ptdata)
        self.assertEqual(pt.size, 4)
        Xpred = np.random.normal(0, 1, (40, 2))
        mu1, var1 = p.predict(Xpred)
        mu2, var2 = pt.predict(Xpred)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        mu1, var1 = p.predict(Xpred, full_cov=True)
        mu2, var2 = pt.predict(Xpred, full_cov=True)
        np.testing.assert_allclose(var1, var2)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
