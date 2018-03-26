#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:03:50 2018

@author: aralhekimoglu
"""
import matplotlib.pyplot as plt
import numpy as np
from math import pi as pi

## generate sinusoidal data with noise option
def generate_sinusodial_data(amount=100,add_noise=True):
    x=np.arange(-2*pi,2*pi,4*pi/amount)
    y=[np.sin(i) for i in x]
    #add noise
    if(add_noise):
        x=[np.random.normal(i,0.03) for i in x]
        y=[np.random.normal(i,0.07) for i in y]
    return x,y


#gaussian basis function to have nonlinear feature vector
def gaussian_basis_function(x,sigma=pi,xmin=-2*pi,xmax=2*pi,numFeatures=20):
    return np.append(1,np.exp(-(x - np.arange(xmin,xmax, (xmax-xmin)/float(numFeatures))) ** 2 / (2 * sigma * sigma)))


#gaussian features
def gaussian_features(x,numFeatures=20):
    sigma = np.ptp(x)/4 
    xmax= np.amax(x)
    xmin= np.amin(x)
    return np.array([gaussian_basis_function(i,sigma,xmin,xmax,numFeatures) for i in x])

x,y= generate_sinusodial_data(100,add_noise=True)    
plt.scatter(x,y)
gaus_features=gaussian_features(x)                   

#Bayesion Regression
num_training_set=gaus_features.shape[0]
num_features=gaus_features.shape[1]
beta=100.0 # noise parameter

#Prior as Gaussian
#S0 can be any num_featuresxnum_features matrix
#m0 can be any num_featuresx1 matrix
alpha=0.001
Sigma_0=(1/float(alpha))*np.identity(num_features)
mean_0=np.zeros((num_features,1))

#Calculate Posterior, again as Gaussian
Sigma_0_inv=np.linalg.inv(Sigma_0)         
Sigma_N=np.linalg.inv(Sigma_0_inv+beta*gaus_features.T.dot(gaus_features))
mean_N=beta*Sigma_N.dot(gaus_features.T).dot(y) # mean of posterior also w in wTx for regression

# output of bayesian regressor given input x, basis function included
def output_regression(x,w):
    return np.dot(w, gaussian_basis_function(x))

x_pred = np.arange(-2*pi, 2*pi, pi/1000.0)
y_pred = [output_regression(i,mean_N) for i in x_pred] 

plt.plot(x_pred,y_pred,'r')