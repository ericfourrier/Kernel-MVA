#!/usr/bin/env python
# -*- coding: utf-8 -*-u

"""
Notes : helper for the run.py file
"""
import numpy as np
# import matplotlib.pyplot as plt
#
#
# def plot_digit(row_data, reshape=True):
#     """ Take a row of data as input (1D array that will be transform to 2D array for picture representation) """
#     if reshape:
#         row_data = row_data.reshape(28, 28)
#     plt.imshow(row_data, cmap=plt.cm.gray)
#     plt.show()


def get_ytrain_dummy(Y, nb_digits=10):
    """ returns a matrix of dummy variables (1, -1) coding the Y_train 10 digits
    useful for one vs all strategy """
    res = np.zeros(((len(Y)), nb_digits))
    for i in range(nb_digits):
        Y_i = np.where(Y[:, 1] == i, 1, -1)
        res[:, i] = Y_i
    return res
#########################################################
#  Data Augmentation functions
#########################################################
def check_margin(digit_arr, shift, direction='right', reshape=True):
    """ Check if we can use roll by looking at the margin of the image """
    if reshape:
        digit_arr = digit_arr.reshape(28, 28)
    first_elements = range(shift)
    nrows = digit_arr.shape[0]
    last_elements = range(nrows - shift,nrows)
    if direction == 'right':
        return np.equal(digit_arr[:,last_elements],0).all()
    if direction == 'left':
        return np.equal(digit_arr[:,first_elements],0).all()
    if direction == 'up':
        return np.equal(digit_arr[first_elements,:],0).all()
    if direction == 'down':
        return np.equal(digit_arr[last_elements,:],0).all()

def translate_digit(digit_arr, direction, shift=2, reshape=True):
    if reshape:
        digit_arr = digit_arr.reshape(28, 28)
    if check_margin(digit_arr, shift, direction):
        if direction == 'right':
            res = np.roll(digit_arr, shift, axis=1)
        elif direction == 'left':
            res = np.roll(digit_arr, -shift, axis=1)
        elif direction == 'down':
            res = np.roll(digit_arr, shift, axis=0)
        elif direction == 'up':
            res = np.roll(digit_arr, -shift, axis=0)
        else:
            raise ValueError("wrong direction")

    else:
        return ValueError("check margin should be correct")
    return res.flatten()


def get_translate_train(X_train, Y_train, direction, shift=2, reshape=True):
    """ Return a dataset of translated train in one direction """
    indices_to_translate = np.apply_along_axis(check_margin, axis=1,
                                               arr=X_train, shift=shift, direction=direction, reshape=reshape)
    X_train_i = X_train[indices_to_translate, :]
    Y_train_i = Y_train[indices_to_translate, :]
    return np.apply_along_axis(translate_digit, axis=1, arr=X_train_i,
                               shift=shift, direction=direction, reshape=reshape), Y_train_i


def get_translate_train_all(X_train, Y_train, shift=2, reshape=True):
    """Returns a merged dataset of translated train in every direction """
    directions = ['right', 'left', 'up', 'down']
    list_array = [get_translate_train(
        X_train, Y_train, direction=d, shift=shift, reshape=reshape) for d in directions]
    X_new = np.vstack([t[0] for t in list_array])
    Y_new = np.vstack([t[1] for t in list_array])
    print(len(X_new))
    print(len(Y_new))
#     X_new =  np.vstack((X_train, X_new))
#     Y_new =  np.vstack((Y_train, Y_new))
    return X_new, Y_new


#########################################################
#  Kernels functions
#########################################################

def polynomial_kernel(X, Y, p=2, biais=3.0):
    """ Vectorized polynomial kernel """
    return (biais + np.dot(X, Y)) ** p


def linear_kernel(X, Y):
    """ Linear kernel vectorized """
    return np.dot(X, Y)


def gaussian_kernel(X, Y, sigma=3.0):
    """ Vectorized gaussian kernel """
    return np.exp(-np.linalg.norm(X - Y) ** 2 / (2 * (sigma ** 2)))


def gaussian_kernel_vec(X, sigma=5.0):
    km = matrix_distance_vec(X)
    km /= (2 * (sigma ** 2))
    km = np.exp(-km)
    return km


def gaussian_kernel_nvec(X, Y, sigma):
    """Non vectorized version of kernels """
    n_samples = X.shape[0]
    km = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            km[i, j] = gaussian_kernel(X[i], Y[j], sigma=sigma)
    return km


def polynomial_kernel_vec(X, Y, p=2, biais=3.0):
    km = np.dot(X, Y.T)
    return ((km + biais) ** p)
