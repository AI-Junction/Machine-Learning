import numpy as np
#import sys
#from __future__ import print_function

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
print(X)
print(syn0)
print(syn1)
print(np.dot(X,syn0))
print(1/(1+np.exp(-0.2542245)))
print(l1)

l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
print(l2)
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
    
print(l2)
print(2*np.random.random((3,4)))
print(2*np.random.random((3,4))-1)
print(2*np.random.random((4,1)) - 1)
print(X)
print(syn0)
print(np.dot(X,syn0))
l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
print(l1.shape)
print(l2.shape)
print(l1_delta)
print(l2_delta)

print(syn0)
print(syn1)
print(l2_delta)

# -*- coding: utf-8 -*-

