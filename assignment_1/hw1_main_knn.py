# -*- coding: utf-8 -*-
"""
Created on 

@author: arindam roychoudhury
"""

 
from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
import sklearn.metrics as metrics
import time
import itertools
import matplotlib.pyplot as plt  

def load_data():
    # Load data - ALL CLASSES
    X_train, y_train = load_mnist('training')
    X_test, y_test = load_mnist('testing')
    
    # Reshape the image data into rows  
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    
    return X_train, y_train, X_test, y_test

def make_1000(X_train, y_train):
    combined = np.c_[X_train, y_train]
    label_col = len(combined[0]) - 1
    
    combined_1000 = []
    
    for digit_class in np.arange(10):
        selected_digit = combined[combined[:,label_col] == digit_class]
        rand_idx = np.random.randint(len(selected_digit), size = 100)
        selected_100 = selected_digit[rand_idx, :]
        if len(combined_1000) == 0:
            combined_1000 = selected_100
        else:
            combined_1000 = np.vstack((combined_1000, selected_100))
    
    np.random.shuffle(combined_1000) #arrange the data randomly
            
    X_1000 = combined_1000[:,:-1]
    y_1000 = combined_1000[:,-1]
    return X_1000, y_1000

def test_1000(X_1000, y_1000, X_test, y_test):
    C1 = test_data(X_1000, y_1000, X_test, y_test, 1)
    C5 = test_data(X_1000, y_1000, X_test, y_test, 5)
    return C1, C5

def test_data(X_train, y_train, X_test, y_test, k = 1):
    dists =  mlBasics.compute_euclidean_distances(X_test, X_train)
        
    #2) Run the code below and predict labels: 
    y_test_pred = mlBasics.predict_labels(dists, y_train, k)
    
    y_test_pred = y_test_pred.flatten()
    
    C = metrics.confusion_matrix(y_test, y_test_pred)
    
    #3) Report results
    print 'for k= {0}, {1:0.02f}'.format(k,np.mean(y_test_pred==y_test)*100), "of test examples classified correctly."
    
    return C, np.mean(y_test_pred==y_test)
    
def make_folds(X_train, y_train, fold = 5):
    X_fold = []
    y_fold = []
    fold_size = (X_train.shape[0] / fold) 
    begin = 0
    end = begin + fold_size
    for part in np.arange(5):
        X_fold.append(X_train[begin:end,:])
        y_fold.append(y_train[begin:end])
        begin = end
        end = (end + fold_size)
    return np.array(X_fold), np.array(y_fold)

def test_fold(X_fold, y_fold, k = 1):
    max_accuracy = 0
    for test_idx in np.arange(len(X_fold)):
        X_test = X_fold[test_idx]
        y_test = y_fold[test_idx]
        
        #everything except the test fold
        X_train = X_fold[np.arange(len(X_fold)) != test_idx]
        y_train = y_fold[np.arange(len(y_fold)) != test_idx]
        
        #combine to form the training data
        rowcount =  len(X_train) * X_train[0].shape[0]
        colcount = X_train[0].shape[1]
        X_train = np.concatenate(X_train).ravel()
        X_train = X_train.reshape(rowcount, colcount)
        y_train = y_train.flatten()
        
        #test the fold
        dists = mlBasics.compute_euclidean_distances(X_test, X_train)
        y_test_pred = mlBasics.predict_labels(dists, y_train, k)
        y_test_pred = y_test_pred.flatten()
        
        # keep track of max accuracy
        max_accuracy = max([max_accuracy, np.mean(y_test_pred==y_test)])
    return max_accuracy

def plot_accuracies(k_acc, k_labels):
    # plot the trend line with error bars that correspond to standard deviation
    plt.plot(k_labels,k_acc)
    plt.title('Best k')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

def find_best_k(X_fold, y_fold, Ks):
    max_acc = 0
    best_k = Ks[0]
    k_acc = []
    for k in Ks:
        acc = test_fold(X_fold, y_fold, k)
        k_acc.append(acc)
        if acc > max_acc:
            max_acc = acc
            best_k = k
    plot_accuracies(k_acc, Ks)
    return best_k, max_acc


#taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def visualize_nn(X_test, X_train, y_train, k):
    dists =  mlBasics.compute_euclidean_distances(X_test, X_train)
    y_test_pred = mlBasics.predict_labels(dists, y_train, k)

if __name__ == '__main__':
    #load all data
    X_train, y_train, X_test, y_test = load_data()
    
    #part b
    X_1000, y_1000 = make_1000(X_train, y_train)
    C1, C5 = test_1000(X_1000, y_1000, X_test, y_test)
    plot_confusion_matrix(C1[0], np.arange(10), normalize=False, title = "C1")
    plot_confusion_matrix(C5[0], np.arange(10), normalize=False, title = "C5")
    #end of part b
    
    X_fold_1000, y_fold_1000 = make_folds(X_1000, y_1000, 5)
      
    k_1000, acc_1000 = find_best_k(X_fold_1000, y_fold_1000, np.arange(1,16))
      
    print "best k in 1000 examples:"
    print k_1000, acc_1000 * 100.0
      
    t1 = time.time()
    C_a, acc = test_data(X_train, y_train, X_test, y_test, k_1000)
    t2 = time.time()
    C_b, acc_1 = test_data(X_train, y_train, X_test, y_test, 1)
    t3 = time.time()
       
    print "all examples:"
    print "for best k:"
    print k, acc, (t2-t1) * 1000.0
    print "for k=1:"
    print k_1, acc_1, (t3-t2) * 1000.0
    