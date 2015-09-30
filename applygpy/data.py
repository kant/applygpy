'''
Created on 29 Sep 2015

@author: Max Zwiessele
'''

def load_libsvm(path):
    from sklearn.datasets import load_svmlight_file  # @UnresolvedImport
    X, Y = load_svmlight_file(path)
    return X, Y