# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:09:29 2023

@author: Fang Cheng
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:36:25 2023

@author: Fang Cheng
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.distributions import Normal
from scipy.stats import norm
import sdeint

def W_in(ressize, insize, win_a, seed):
    np.random.seed(seed)
    win = (np.random.rand(ressize, 1 + insize) - 0.5) * win_a
    return win



def W_res(degree, ressize, eig_rho, seed = 13):
    G = nx.erdos_renyi_graph(ressize, degree/ressize, seed=seed)
    plt.figure(0).clear()
    nx.draw(G, node_size=50)
    
    W = nx.to_numpy_matrix(G)
    print('Computing spectral radius...', end='')
    rhoW = max(abs(linalg.eig(W)[0]))
    print(rhoW,'done.')
    W *= eig_rho / rhoW
    return W


def create_ESN(ESN_par, data, W_in, W_res):
    ressize = ESN_par[0]
    a = ESN_par[1]
    reg = ESN_par[2]
    trainlen = ESN_par[3]
#     testlen = ESN_par[4]
    initlen = ESN_par[4]
    insize = ESN_par[5]
    trajectory_n = ESN_par[6]
    
#     global X, y_train
    # allocated memory for the design (collected states) matrix
    X = np.zeros((ressize, (trainlen - initlen) * trajectory_n)) #reservoir collection
    # set the corresponding target matrix directly
    Yt = data[:, initlen + 1:trainlen + 1] # (insize*trajectory_n) * (len(tran-init))
    Yt_flatten = np.zeros((insize, (trainlen - initlen) * trajectory_n)) # insize , (len(tran-init)*trajectory_n)
    x_last = np.zeros((ressize, trajectory_n))
    print('Reservoir states shape: {}\nTrain data flatten shape:{}\nLast reservoir shape: {}'.format(X.shape, Yt_flatten.shape, x_last.shape))
    # run the reservoir with the data and collect X
    
    for n in range(trajectory_n):
        Yt_flatten[:, n*(trainlen-initlen):(n+1)*(trainlen-initlen)] = Yt[n*insize:(n+1)*insize, :]
    
    u_train = data[:, :trainlen]
    x = np.zeros((ressize, 1*trajectory_n))
    for t in range(trainlen):
        u = u_train[:, t].reshape(insize, -1, order='F')#insize*trajectory_N
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
        if t >= initlen:
            X[:, t - initlen::(trainlen-initlen)] = x #X: (ressize+isnize+1) * (len(tran-init)*trajectory_n)
    x_last = x
    print('**Reservoir states collection finished**')
    
    # train the output
    X_T = X.T #X_T: (len(tran-init)*trajectory_n) * (ressize+isnize+1) 
    W_out = np.dot(np.dot(Yt_flatten, X_T), linalg.inv(np.dot(X, X_T) 
                                                       + reg * np.eye(ressize)))
    y_train = np.matmul(W_out, X)
    mse_train = np.mean(np.square(Yt_flatten - y_train))
    print('MSE in training = ' + str(mse_train))
    
    return W_out, x_last #x is last reservior state

def error_cal(data, trajectory_n, last_res, outsize, insize, trainlen, errorlen, W_out, W_res, W_in, a):
    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outsize * trajectory_n, errorlen))
    x_last = np.zeros(last_res.shape)
    
 
    x = last_res
    for t in range(errorlen):
        u = data[:, trainlen+t].reshape(insize,-1,order='F')
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
        y = np.dot(W_out, x)
        Y[:, t] = y.reshape(-1,order='F')

    x_last = x
        
    mse = np.mean(np.square(data[:, trainlen + 1:trainlen +
                      errorlen + 1] - Y[:, 0:errorlen]))
    if mse > 1e+5:
        mse = 1e+5
    print('MSE = ' + str(mse))
    return mse, x_last ##x is last reservior state used for test


def error_cal_was(data, trajectory_n, last_res, outsize, insize, trainlen, errorlen, W_out, W_res, W_in, a):
    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outsize * trajectory_n, errorlen))
    x_last = np.zeros(last_res.shape) 
    
    x = last_res
    for t in range(errorlen):
        u = data[:, trainlen+t].reshape(insize,-1,order='F')
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
        y = np.dot(W_out, x)
        Y[:, t] = y.reshape(-1,order='F')
        # generative mode:
#             u = y
    x_last
    
#     Y_was = np.concatenate((Y.reshape(-1,1),t_column),axis=1)
#     data_was = np.concatenate((data[:, trainlen + 1:trainlen + errorlen + 1].reshape(-1,1),t_column),axis=1)
    
    x = torch.tensor(data[:, trainlen + 1:trainlen + errorlen + 1].T, dtype=torch.float).unsqueeze(-1) #errorlen*trajectory_N*1
    y = torch.tensor(Y.T, dtype=torch.float).unsqueeze(-1)

    sinkhorn = SinkhornDistance(eps=0.01, max_iter=100, reduction=None)
    dist, P, C = sinkhorn(x, y)
    dist_mean = dist.mean()
    print("Sinkhorn distance: {:.6f}".format(dist_mean.item()))
    
#     mse = np.mean(np.square(data[:, trainlen + 1:trainlen +
#                       errorlen + 1] - Y[:, 0:errorlen]))
#     if mse > 1e+5:
#         mse = 1e+5
#     print('MSE = ' + str(mse))
    return dist_mean.item(), x_last ##x is last reservior state used for test


def prediction(data, trajectory_n, last_res, outsize, insize, trainlen, testlen, W_out, W_res, W_in, a):          

    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outsize * trajectory_n, testlen))
    

    u_pred = data[:, trainlen].reshape(insize,-1,order='F')
    x = last_res
    for t in range(testlen):
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u_pred))) + np.dot(W_res, x))
        y = np.dot(W_out, x)
        Y[:, t] = y.reshape(-1,order = 'F')
        # generative mode:
        u_pred = y

    return Y


#non-rolling error boxplot
def errortrain_NR(ESN_par, data, W_in, W_res, W_out):
    ressize = ESN_par[0]
    a = ESN_par[1]
    reg = ESN_par[2]
    trainlen = ESN_par[3]
#     testlen = ESN_par[4]
    initlen = ESN_par[4]
    insize = ESN_par[5]
    trajectory_n = ESN_par[6]
    
    # allocated memory for the design (collected states) matrix
    XX = np.zeros((ressize, (trainlen - initlen) * trajectory_n)) #reservoir collection
    # set the corresponding target matrix directly
    Yt = data[:, initlen + 1:trainlen + 1] # (insize*trajectory_n) * (len(tran-init))
    print('Reservoir states shape: {}\n'.format(XX.shape))
    # run the reservoir with the data and collect X

    u_train = data[:, :trainlen]
    x = np.zeros((ressize, trajectory_n))
    for t in range(trainlen):
        u = u_train[:, t].reshape(insize, -1, order='F')
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
        if t >= initlen:
            XX[:, t - initlen::(trainlen-initlen)] = x #X: (ressize+isnize+1) * (len(tran-init)*trajectory_n)
    print('**Reservoir states collection finished**')

    y_train_flatten = np.matmul(W_out, XX)
    y_train = np.zeros((insize*trajectory_n, trainlen-initlen))
    for n in range(trajectory_n):
        y_train[n*insize:(n+1)*insize,:] = y_train_flatten[:,n*(trainlen-initlen):(n+1)*(trainlen-initlen)]    
    
    return y_train #x is last reservior state


#rolling error boxplot
def errortrain_R(ESN_par, trajectory_n, trajectory_all, W_out, W_res, W_in):          
    ressize = ESN_par[0]
    a = ESN_par[1]
    reg = ESN_par[2]
    trainlen = ESN_par[3]
#     testlen = ESN_par[4]
    initlen = ESN_par[4]
    insize = ESN_par[5]
    trajectory_n = ESN_par[6]
    
    X = np.zeros((ressize, (trainlen - initlen) * trajectory_n)) #reservoir collection
    Y = np.zeros((insize, (trainlen-initlen) * trajectory_n))

    u_train = trajectory_all[:, :initlen + 1]
    x = np.zeros((ressize, trajectory_n))

    for t in range(initlen):
        u = u_train[:, t].reshape(insize, -1,order='F')
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
    u = u_train[:, -1].reshape(insize, -1,order='F')
    for t in range(0,trainlen-initlen):
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
        y = np.dot(W_out, x)
        Y[:, t::(trainlen - initlen)] = y
        u = y  
    #rolling train finished
    y_train_approx = np.zeros((insize*trajectory_n, trainlen-initlen))
    for n in range(trajectory_n):
        y_train_approx[n*insize:(n+1)*insize,:] = Y[:,n*(trainlen-initlen):(n+1)*(trainlen-initlen)]

    y_train = trajectory_all[:, initlen + 1:trainlen + 1]

    errortrain_r = y_train_approx - y_train
    return errortrain_r


def predictionError_NR(data, trajectory_n, last_res, outsize, insize, trainlen, testlen, W_out, W_res, W_in, a, errorsamples):          

    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outsize * trajectory_n, testlen))
    error_samples = errorsamples.T #tensor shape(samples * insize)    

    u_pred = data[:, trainlen].reshape(insize,-1,order='F')
    x = last_res
    for t in range(testlen):
        x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u_pred))) + np.dot(W_res, x))
        y = np.dot(W_out, x) - error_samples[:, t*trajectory_n:(t+1)*trajectory_n]         
        # generative mode:
        u_pred = y
        Y[:, t] = y.reshape(-1,order='F')

    return Y


def universality(ESN_par, data, W_in, W_res, W_out, errordata): #data for warm up
    ressize = ESN_par[0]
    a = ESN_par[1]
    reg = ESN_par[2]
    trainlen = data.shape[1] + errordata.shape[1] #ESN_par[3]
#     testlen = ESN_par[4]
    initlen = data.shape[1]
    insize = ESN_par[5]
    trajectory_n = int(data.shape[0]/insize) #ESN_par[6]
    
    # allocated memory for the design (collected states) matrix
    Y = np.zeros((insize * trajectory_n, trainlen - initlen))
    # set the corresponding target matrix directly
    x_last = np.zeros((ressize, trajectory_n))

    u_train = data
    
    x = np.zeros((ressize, trajectory_n))
    for t in range(trainlen):
        if t < initlen - 1:
            u = u_train[:, t].reshape(insize, -1,order='F')
            x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
            y = np.dot(W_out, x)
        elif t == initlen - 1:
            u = u_train[:, t].reshape(insize, -1,order='F')
            x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
            y = np.dot(W_out, x) - errordata[:, 0].reshape(insize,-1,order='F')
            u = y
        else:                
            x = (1 - a) * x + a * np.tanh(np.dot(W_in, np.vstack((np.ones(trajectory_n), u))) + np.dot(W_res, x))
            y = np.dot(W_out, x) - errordata[:, t-initlen].reshape(insize,-1,order='F')
            Y[:, t - initlen] = y.reshape(-1,order='F')
            u = y
    
    return Y #x is last reservior state


