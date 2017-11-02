# -*- coding: utf-8 -*-
"""
Created on  

@author: arindam roychoudhury
"""

import numpy as np 
import scipy.special as spl
from matplotlib.pyplot import axis
 

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(X.dot(W) + b)
    
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return spl.expit(a)
    

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
    """
    predicted = predict(X, W, b)
    loss = (y - predicted)
    predicted = np.reshape(predicted, (len(predicted), 1))
    loss = np.reshape(loss, (len(loss), 1))
    sum_sqr_loss = np.sum(loss ** 2)
    dw = -2.0 * X * loss * predicted * (1 - predicted)
    db = -2.0 * loss * predicted * (1 - predicted)
    sum_dw = np.sum(dw, axis = 0) 
    sum_db = np.sum(db, axis = 0)
    return sum_sqr_loss, sum_dw, sum_db
    


def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
    """
    precision = 0.00001
    cur_W = W
    cur_b = b
    for i in np.arange(num_iters):
        loss, dw, db = l2loss(X, y, cur_W, cur_b)
        if loss < precision:
            break 
        prev_W = cur_W
        cur_W += -eta * dw
        prev_b = cur_b
        cur_b += -eta * db
    return cur_W, cur_b
            
        


 