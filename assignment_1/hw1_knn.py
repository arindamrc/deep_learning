# -*- coding: utf-8 -*-
"""
Created on  

@author: arindam roychoudhury
"""
 
import numpy as np 
import scipy.stats as stats
from matplotlib.pyplot import axis
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    # euclidean distance is sqrt(X^2 + Y^2 - 2*X*Y)
    X_sqr =  (X**2).sum(axis=1, keepdims=True)
    Y_sqr =  (Y**2).sum(axis=1) 
    XY = X.dot(Y.T)
    distances = np.sqrt(X_sqr + Y_sqr - 2*XY)
    return distances
 

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    # sort the distances matrix to get the nearest examples
    sorted_indices = np.argsort(dists, axis = 1)[:,:k] #take first k indices
    expanded_labels = np.tile(labels,[len(sorted_indices), 1])
    rows = np.arange(len(expanded_labels)).reshape(len(expanded_labels),1)
    chosen_labels = expanded_labels[rows, sorted_indices]
    mode = stats.mode(chosen_labels, axis = 1)[0]
    return mode
    
    
     