'''
Created on 30 Sep 2015

@author: Max Zwiessele
'''
import matplotlib
from GPy.testing.plotting_tests import flatten_axis as fl, compare_axis_dicts as cm
matplotlib.use('agg')
import matplotlib.pyplot as plt  # @UnresolvedImport

import GPy, numpy as np
from applygpy.prediction import PredictionModelSparse, PredictionModel
from io import StringIO
import unittest

class Test(unittest.TestCase):

    def setUp(self):
        self.X, self.Y = np.random.normal(0, 1, (10, 1)), np.random.normal(0, 1, (10, 1))
        pass

    def tearDown(self):
        plt.close('all')

    def testPlotting(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        p = PredictionModel(m)
        fig, ax1 = plt.subplots()
        m.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        #self.assertEqual(i1.read(), i2.read())
        cm(fl(ax1), fl(ax2))

    def testPlottingSparse(self):
        m = GPy.models.SparseGPRegression(self.X, self.Y)
        p = PredictionModelSparse(m)
        fig, ax1 = plt.subplots()
        m.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        #self.assertEqual(i1.read(), i2.read())
        cm(fl(ax1), fl(ax2))

    def testPlottingClass(self):
        m = GPy.models.GPClassification(self.X, self.Y<0)
        p = PredictionModel(m)
        fig, ax1 = plt.subplots()
        m.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        #self.assertEqual(i1.read(), i2.read())
        cm(fl(ax1), fl(ax2))

    def testPlottingSparseClass(self):
        m = GPy.models.SparseGPClassification(self.X, self.Y<0)
        p = PredictionModelSparse(m)
        fig, ax1 = plt.subplots()
        m.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        #self.assertEqual(i1.read(), i2.read())
        cm(fl(ax1), fl(ax2))

    def testPlottingDataNotShow(self):
        m = GPy.models.SparseGPRegression(self.X, self.Y)
        p = PredictionModelSparse(m)
        p.plot_data()

        fig, ax1 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=True, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        cm(fl(ax1), fl(ax2))

        m = GPy.models.GPRegression(self.X, self.Y)
        p = PredictionModel(m)
        p.plot_data()

        fig, ax1 = plt.subplots()
        p.plot(plot_training_data=False, ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-2, 2)
        #i1 = StringIO()
        #fig.savefig(i1, format='svg')
        #i1.seek(0)

        fig, ax2 = plt.subplots()
        p.plot(plot_training_data=True, ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-2, 2)
        #i2 = StringIO()
        #fig.savefig(i2, format='svg')
        #i2.seek(0)

        cm(fl(ax1), fl(ax2))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPlotting']
    unittest.main()
