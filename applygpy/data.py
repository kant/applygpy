'''
Created on 29 Sep 2015

@author: Max Zwiessele
'''

def load_libsvm(path):
    from sklearn.datasets import load_svmlight_file  # @UnresolvedImport
    X, Y = load_svmlight_file(path)
    return X, Y


if __name__ == '__main__':
    path = '../azDatasets/dataset1.train'
    X, Y = load_libsvm(path)