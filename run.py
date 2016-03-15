#!/usr/bin/env python
# -*- coding: utf-8 -*-u

"""
Notes : Training a model from scratch for the kernel MVA course
"""
#########################################################
# Import packages and helpers
#########################################################

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial



from utils import *
# import matplotlib.pyplot as plt


#########################################################
# Load Data
#########################################################

X_train = np.genfromtxt("Xtr.csv", delimiter=',')
Y_train = np.genfromtxt("Ytr.csv", delimiter=',')
Y_train = Y_train[1:, :]
X_test = np.genfromtxt("Xte.csv", delimiter=',')


#########################################################
# Data Augmentation
#########################################################

# Add new training set by doing 2 pixels translation
train_new = get_translate_train_all(X_train, Y_train)
X_train_new, Y_train_new = train_new

#reduce nb of samples
nb_new_train_samples = 8000
# random selection
index_to_select = np.random.choice(range(X_train.shape[0]), nb_new_train_samples)
X_train_new, Y_train_new = X_train_new[
    index_to_select, :], Y_train_new[index_to_select, :]

# Adding basic training samples
X_train_new = np.vstack((X_train, X_train_new))
Y_train_new = np.vstack((Y_train, Y_train_new))


#########################################################
# SVM
#########################################################

class SVM2(object):

    def __init__(self, kernel=polynomial_kernel_vec):
        self.kernel = kernel

    def fit(self, X, y, C=0.1, **kwargs):
        """ Training the model

        Parameters
        ----------
        X: numpy.ndarray
            array containing all the training features
        y: """

        # store Xtrain for the test kernel
        self.Xtrain = X

        # store **kwargs for test kernel
        self.kernel_params = kwargs
        print(self.kernel_params)

        n_samples, n_features = X.shape

        # Get Gram matrix with our vectorized kernels
        K = self.kernel(X, X, **kwargs)

        # cvxopt solves min(1/2 x^T P x + q^T x)
        # s.t. : Gx <= h, Ax = b
        print("shape equality : {}".format(K.shape[0] == y.shape[0]))
        d = y.astype(np.float64)
        P = cvxopt.matrix(K)
        q = cvxopt.matrix(-d, tc='d')
#         A = cvxopt.matrix(y, (1,n_samples))
#         b = cvxopt.matrix(0.0)

        #  0 <= y_i * a_i
        G_1 = np.diag(-1 * d)
        h_1 = np.zeros((n_samples, 1), dtype='float64')

        # y_i * a_i <= C
        G_2 = np.diag(d)
        h_2 = C * np.ones((n_samples, 1), dtype='float64')

        # stacking arrays
        G = cvxopt.matrix(np.vstack((G_1, G_2)))
        h = cvxopt.matrix(np.vstack((h_1, h_2)))

        # solve QP problem
        cvxopt.solvers.options['show_progress'] = True
        solution = cvxopt.solvers.qp(P, q, G, h)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = (np.abs(a) > 1e-15)
        self.ind_sv = np.arange(len(a))[sv]
        self.a = a[sv]
#       self.sv = X[sv]
#       self.sv_y = y[sv]
        print ("{} support vectors out of {} points").format(
            len(self.a), n_samples)


#         # Intercept
#         self.b = 0
#         for n in range(len(self.a)):
#             self.b += self.sv_y[n]
#             self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
#         self.b /= len(self.a)

    def project(self, X):
        K_test = self.kernel(X, self.Xtrain, **self.kernel_params)
        return np.dot(K_test[:, self.ind_sv], self.a)

#########################################################
#  Machine learning helpers
#########################################################

def create_train_test(X, Y, perc=0.5):
    """ Split data in train test , returns tuple with (train,test) """
    nrows = X.shape[0]
    rows_test = int(nrows * perc)
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[rows_test:], indices[:rows_test]
    return X[training_idx, :], Y[training_idx, :], X[test_idx, :], Y[test_idx, :]





def helper_predicition_p(i, X_train, Y_train, X_test, *args, **kwargs):
    Y_train_i = np.where(Y_train[:, 1] == i, 1, -1)
    svm = SVM2(kernel=polynomial_kernel_vec)
    svm.fit(X_train, Y_train_i,*args, **kwargs)
    preds = svm.project(X_test)
    return preds


def multi_classification(X_train, Y_train, X_test, n_jobs=None, *args, **kwargs):
    res = np.zeros((X_test.shape[0], 10))
    if n_jobs is None:
        for i in range(10):
            Y_train_i = np.where(Y_train[:, 1] == i, 1, -1)
            svm = SVM2(kernel=polynomial_kernel_vec)
            svm.fit(X_train, Y_train_i, *args, **kwargs)
            preds = svm.project(X_test)
            res[:, i] = preds
        return res.argmax(axis=1)
    else:
        # res = Parallel(n_jobs=n_jobs, verbose=100)(
        #     delayed(helper_predicition_p)(i, X_train, Y_train, X_test) for i in range(10))
        partial_h = partial(helper_predicition_p,X_train=X_train, Y_train=Y_train, X_test=X_test)
        pool = Pool(n_jobs)
        res = pool.map(partial_h,range(10))
        pool.close()
        print(len(res))
        print(res[0].shape)
        return np.hstack(res)


def multi_classification_accuracy(X_train, Y_train, X_test, Y_test, *args, **kwargs):
    predictions = multi_classification(
        X_train, Y_train, X_test, *args, **kwargs)
    return float(np.equal(predictions, Y_test[:, 1]).sum()) / len(Y_test)

#########################################################
# Cross Validation
#########################################################
#
# X_train_1, Y_train_1, X_test_1, Y_test_1 = create_train_test(X_train, Y_train, perc=0.75)
# multi_classification_accuracy(X_train_1, Y_train_1, X_test_1,Y_test_1, n_jobs=None)
#########################################################
# Fit model and predict test
#########################################################
res_test = multi_classification(X_train_new, Y_train_new, X_test, C=1, p=3, biais=1)
df_test = pd.DataFrame(range(1,10001), columns=['Id'])
df_test.loc[:,'Prediction'] = res_test
df_test.to_csv('Yte.csv', index=False)
