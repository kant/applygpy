'''
Created on 29 Sep 2015

@author: Max Zwiessele
'''
try:
    import matplotlib  # @UnresolvedImport
    matplotlib.use('Agg')
except:
    pass
import data
import model_selection
import prediction