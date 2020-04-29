#!/usr/bin/env python
# coding: utf-8
# Project: Optimizing protein sequence encodings for improved machine learning
# Author: Sarah Fahlberg
# Details: This project examines two different encoding methods for machine learning of directed evolution projects
#    by generating machine learning curves for each method

import pickle
import numpy as np
import random


#One-hot-encoding functions

def oneHotEncodeOptimized(seqs, parent_seqs, standardize = False):
    ''' Encodes a list of sequences based on the parents.
    First creates a matrix called pos_coding of all the possible positions
    Ex: If parents are ['ATR', 'AYR', 'RQR'] with interval=1, then pos_coding =
    |[A, R]   |
    |[T, Y, Q]|
    |[R]      |
    Args:
        seqs: list variable containing strings that represent each of the sequences to be encoded
        parent_seqs: list variable containting strings of the sequences that all of the sequences are based on
        interval: number of characters that separate each individual position. Ex: For DNA codons interval = 3
        standardize: if implementing ridge regression standardize must be set to true
    Returns:
        seqs_encoded: a list containing a list for each one-hot encoded sequence
        return pos_coding: a list variable used for decoding sequences
    '''
    # Each new code will be appended to the list of codes for that position
    pos_coding = []
    
    for AA_list in zip(*parent_seqs):
        pos = []
        for AA in AA_list:
            if AA not in pos:
                pos.append(AA)
        pos_coding.append(pos[1:])
    
    # Encoding algorithm
    seqs_encoded = []
    
    for seq in seqs:
        individual_encoded = []
        if standardize is False:
            individual_encoded.append(1) #Adds the normalizing 1
        for AA, pos in zip(seq, pos_coding):
            for pos_AA in pos:
                if AA == pos_AA:
                    individual_encoded.append(1)
                else:
                    individual_encoded.append(0)
        seqs_encoded.append(individual_encoded)
        
    return seqs_encoded, pos_coding


def decodeOneHotEncodeOptimized(seqs_encoded, pos_coding, parent_seqs, standardize = False):
    '''This function decodes a list of sequences that have been one-hot encoded by the oneHotEncodeOptimizedv2 
    function
    Args:
        seqs_encoded: a list/matrix of one hot-encoded sequences
        pos_coding: a list variable obtained when one-hot encoding the original sequences
        interval: the number of characters that separate each individual position. Ex: For DNA codons interval = 3
        standardize: if implementing ridge regression standardize must be set to true
    Returns:
        seqs_decoded - a list variable containting strings of the decoded sequences'''
    
    seqs_decoded = []
    for individual_encoded in seqs_encoded:
        individual_encoded = individual_encoded.copy()
        if not standardize:
            individual_encoded.remove(1) #Removes the normalizing 1
        individual_decoded = ''
        for i, pos_AA in enumerate(pos_coding):
            AA_encoded = individual_encoded[:len(pos_AA)]
            individual_encoded = individual_encoded[len(pos_AA):]
            if 1 not in AA_encoded:
                individual_decoded += parent_seqs[0][i]
            else:
                individual_decoded += pos_AA[AA_encoded.index(1)]
        seqs_decoded.append(individual_decoded)
    return seqs_decoded


def checkEncoding(seqs, parent_seqs, standardize = False):
    '''This function checks that the encoding and decoding methods work as expected on the dataset given
    Args:
        seqs_encoded: a list/matrix of one hot-encoded sequences
        pos_coding: a list variable obtained when one-hot encoding the original sequences
        interval: the number of characters that separate each individual position. Ex: For DNA codons interval = 3
        standardize: if implementing ridge regression standardize must be set to true
    Errors:
        AssertionError: thrown if encoding or decoding method is not working properly
        '''
    seqs_encoded, pos_coding = oneHotEncodeOptimized(seqs, parent_seqs, standardize = standardize)
    assert seqs == decodeOneHotEncodeOptimized(seqs_encoded, pos_coding, parent_seqs, standardize = standardize)
    

#Ridge Regression Functions

def standardize_X(X, X_test = []):
    '''This function standardizes X by calculating the mean of X and dividing by the stdev
    Arg:
        X: a list variable to be standardized
        X_train = []: optionally the X_train values can be standardized as well during this time
    Returns:
        X_std: numpy array representing X that has been standardized each xij is replaced by xij-xj_bar 
        mean_X: numpy array representing the mean of the xjth column
        X_train_std: numpy array representing the standardized X_train set, if no train set is provided then
            an empy list is returned'''
    mean_X = []
    for xj in zip(*X):
        xj = np.array(xj)
        mean_xj = sum(xj)/len(xj)
        mean_X.append(mean_xj)
        
    #Creates X_std
    X_std = [[] for i in range(len(X))]
    for i, xi in enumerate(X):
        for j, xij in enumerate(xi):
            X_std[i].append(xij-mean_X[j])
            
    #Creates X_test_std
    X_test_std = [[] for i in range(len(X_test))]
    for i, xi in enumerate(X_test):
        for j, xij in enumerate(xi):
            X_test_std[i].append(xij-mean_X[j])
    return np.array(X_std), np.array(mean_X), np.array(X_test_std)


def estimate_B0(y):
    '''Estimates the B0 by finding the mean of y
    Args:
        y: list variable consisting of y-train
    Returns:
        mean_y: estimated B0'''
    
    mean_y = sum(np.array(y))/len(y)
    return mean_y


def RidgeRegressionV2(X, y, l=0):
    '''Ridge regression with standardized X
    Args:
        X: a numpy array representing the standardized independent variable/sequence
        y: a numpy array representing the dependent variable
        l=0: lamba used in ridge regression
    Returns: 
        beta_hat_R: such that y=y_bar + BX where beta_hat reduces the least squares error and the coefficients 
            of beta_hat_R '''
    beta_hat_R = np.linalg.pinv(X.transpose().dot(X)+l*np.identity(len(X[0]))).dot(X.transpose()).dot(y)
    return beta_hat_R


def leaveOneOutCV_Ridge_Std(X, y, l=0):
    '''Leave-one-out cross validation function for ridge regression with standardized variables
    Args:
        X: a list representing the independent variable/sequence
        y: a list representing the dependent variable
        model: a function that runs some form of regression
    Returns:
        y_hats: the predicted value of X[i] is represented as y_hats[i]
        r: the correlation coefficient for this model
        mserr: the mean square error for this model
        mae: the mean absolute error for this model'''
    y_hats = []
    mserr = 0
    mae = 0
    
    for i in range(len(X)):
        #Create train set
        train_X = X.copy()
        train_y = y.copy()
        train_X.pop(i)
        train_y.pop(i)
        
        #Fit model
        std_X, X_bar, std_X_test = standardize_X(train_X, [X[i]])
        beta = RidgeRegressionV2(std_X, np.array(train_y), l)
        y_bar = estimate_B0(train_y)
        
        #Predict y_hats
        y_hats.append(y_bar + beta.dot(std_X_test[0]))
        mserr += (y[i]-y_hats[i])**2
        mae += abs(y[i]-y_hats[i])
        
    #Calculate error measures
    mserr = mserr/len(X)
    mae = mae/len(X)
    r = np.corrcoef(np.array(y),np.array(y_hats))[0][1]
    
    return y_hats, r, mserr, mae
    

def selectLambda(X, y, min_log_l = -10, max_log_l = 1.5):
    '''Tests lambdas over a logspace to get best lambda
    Args:
        X: a list representing the independent variable/sequence
        y: a list representing the dependent variable
        min_log_l = -10: The minimum log of lambda that will be prospected
        max_log_l = -10: The maximum log of lambda that will be prospected
    Returns:
        lambdas: list of lambdas prospected
        mserrs: list of MSEs for the ridge regression model at each lambda calculated from leave-one-out CV
        maes: list of MAEs for the ridge regression model at each lambda calculated from leave-one-out CV
        '''
    lambdas = np.logspace(min_log_l, max_log_l)
    mserrs = []
    maes = []

    for l in lambdas:
        Y_hats, r, mserr, mae = leaveOneOutCV_Ridge_Std(X, y, l)
        mserrs.append(mserr)
        maes.append(mae)
    
    return lambdas, mserrs, maes


#Generate Machine Learning Curves
def predictTestSet(train_X, train_y, test_X, test_y, l=0):
    '''This function, assess how well a model works for a certain train and test set
    Args:
        train_X, train_y: lists containing the encoded data and the function data
        test_X, test_y: lists containing the test set
    Returns:
        y_hats: predicted y values for the test set
        r: correlation coefficient for this model
        mserr: mean square error for this model
        mae: mean absolute error for this model
    '''
    #Convert train and test sets to numpy arrays
    test_y = np.array(test_y)
    std_X, X_bar, std_X_test = standardize_X(train_X, test_X)
    
    #Generate model with train set
    beta = RidgeRegressionV2(std_X, np.array(train_y), l)
    
    #Use the model to predict y_hat for each test sequence in test X and evaluate the mean-squared error 
    #and correlation coeefficent
    y_hats = []
    mserr = 0
    mae = 0
    y_bar = estimate_B0(train_y)
    
    for i in range(len(test_X)):
        y_hat = y_bar + beta.dot(std_X_test[i])
        y_hats.append(y_hat)
        mserr += (y_hat-test_y[i])**2
        mae += abs(y_hat-test_y[i])
    mserr /= len(test_y)
    mae /= len(test_y)
    r = np.corrcoef(test_y, np.array(y_hats))[0][1]

    return y_hats, r, mserr, mae

def generateMLCurve(X, y, std_l, seed = 10, train_size_min = 1, train_size_max = 100, num_sims = 100):
    '''Generates a ML curve running num_sims number of trial at each training size
    Args:
        X: indepenedent variable
        y: dependent variable
        std_l: standard lambda calculated from 
        seed = 10: for reproducibility
        train_size_min = 1: must be greater than 0
        train_size_max = 100: must be smaller than size of train set
        num_sims = 100: number of simulations to run
    Returns:
        train_sizes: sizes surveyed
        r_ML_curve: averaged r values for each train size
        mserr_ML_curve: averaged MSE for each train size
        mae_ML_curve: averaged MAE for each train size
    '''
    #Set seed to ensure reproducibility
    random.seed(seed)
    
    train_sizes = [i for i in range(train_size_min, train_size_max)]
    
    #Initialize variables used to evaluate ML curve
    r_values = [[] for i in train_sizes]
    mserr_values = [[] for i in train_sizes]
    mae_values = [[] for i in train_sizes]
    

    #Each k represents a simulation
    for k in range(num_sims):
        #Set aside 10% of the data, this will be used to assess the error of the model created for 
            #each training size
        test_set_X = []
        test_set_y = []
        total_train_set_X = []
        total_train_set_y = []
        total = [i for i in range(len(X))] # List of all possible indexes of data
    
        for i in range(len(X)):
            rand_ind = random.randint(0,len(total)-1)
            ind = total.pop(rand_ind)
            if (i%10 == 0):
                test_set_X.append(X[ind])
                test_set_y.append(y[ind])
            else:
                total_train_set_X.append(X[ind])
                total_train_set_y.append(y[ind])

        #Creates a model for each training size and evaluates the model
        for i in range(len(train_sizes)):
            #creates train set containg the current train_size number of sequences
            total_train_set_X_c = total_train_set_X.copy()
            total_train_set_y_c = total_train_set_y.copy()
            train_X = []
            train_y = []
            for j in range(train_sizes[i]):
                rand_ind = random.randint(0, len(total_train_set_X_c) - 1)
                train_X.append(total_train_set_X_c.pop(rand_ind))
                train_y.append(total_train_set_y_c.pop(rand_ind))
            #creates model and predicts y_hats for each sequence in the original test set
            y_hats, r, mserr, mae = predictTestSet(train_X, train_y, test_set_X, test_set_y, std_l)
            mserr_values[i].append(mserr)
            mae_values[i].append(mae)
            r_values[i].append(r)
                
            
    #Averages r, mserr, mae
    r_ML_curve = []
    mserr_ML_curve = []
    mae_ML_curve = []
    for i in range(len(r_values)):
        r_ML_curve.append(np.average(np.array(r_values[i])))
        mserr_ML_curve.append(np.average(np.array(mserr_values[i])))
        mae_ML_curve.append(np.average(np.array(mae_values[i])))
    
    return train_sizes, r_ML_curve, mserr_ML_curve, mae_ML_curve



