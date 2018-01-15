# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:23:40 2017

@author: echtpar
"""

import numpy as np
import range

X = [1,2,3,4,5,6,7,8,9]

Y = np.array(X)

print (Y)
print (X)

A = [1,2,3]
B = np.array(A)
print(A)
print(B)


Z = np.convolve(B, X, 'full')
print(Z)


def fc_forward(x,w,b):
    """
    computes the forward pass for an affine layer (fully connected layer)
    
    Inputs:
    - x: Input Tensor (N, d_1,....d_k)
    - w: Weights (D, M)
    - b: Bias(M,)
    
    N: Mini-batch size
    M: Number of outputs of fully connected layer
    D: Input Dimension
    
    Returns a tuple of
    - out: output, of shape(N, M)
    - cache: (x, w, b)
    """
    
    out = None
    
    # get batch size (first dimension)
    N = x.shape[0]
    
    # Reshape activations to [N X (d_1),...., d_k], which will be a 2d matrix 
    # [NXD]
    
    reshaped_input = x.reshape(N, -1)
    
    #calculate output
    out = np.dot(reshaped_input, w) + b.T
    
    # Save inputs for backward propagation
    cache = (x, w, b)
    return out, cache
    
def fc_backward(dout, cache):
    
    """
    Inputs:
        - dout: Layer partial derivative w.r.t loss of shape (N,M) (same as output)
        - cache: (x,w,b) inputs from previous forwards computation
        
        N: Mini batch size
        M: Number of outputs of fully connected layer
        D: Input dimension
        d_1,...., d_k: Single input dimension
        
        Returns a tuple of:
            - dx: gradient w.r.t x, of shape (N, d_1,...., d_k)
            - dw: gradient w.r.t w of shape (D, M)
            - db: gradient w.r.t b of shape (M,)
            
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    
    # Get batch size (first dimension)
    
    N = x.shape[0]
    
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    
    # Get dW (same format as w)
    # reshape activations to [NX(d_1,....d_k)], which will be a 2d matrix
    # [NXD]

    reshaped_input = x.reshape(N, -1)
    
    # Transpose then dot product with dout
    dw = reshaped_input.T.dot(dout)
    
    # Get dB (Same format as b)
    db = np.sum(dout, axis = 0)
    
    # Return outputs
    dx, dw, db
        

def relu_forward(x):        
    """
    Computes the forward pass for ReLU
    Input:
        - x: Inputs, of any shape
    
    Returns a tuple of: (out, cache)
    The shape on the output is the same as the input
    """
    
    out = None
    
    # Create a function that receives x and returns x if x is bigger than
    # zero, or zero if x is negative
    
    relu = lambda x: x*(x > 0).astype(float)
    out = relu(x)
    
    
    # Cache input and return outputs
    cache = x
    return out, cache
    
    
def relu_backward (dout, cache):
    """
    Computes the backward pass for ReLU
    Input:
        - dout: Upstream derivates, of any shape
        - cache: Previous input (used on forward pass)
        
    Returns:
        - dx: gradient w.r.t x
    """
    
    # Initialize dx with None and x with cache
    dx, x = None, cache
    
    # Make all positive elements in x equal to dout while all the other elements 
    # become zero
    dx = dout *(x>=0)
    
    # Return dx (gradient with respect to x)
    return dx
    
#experiment with drop out algorigthm - uncomment below to understand
#x = [[1,2,3,4,5,6], [7,8,9,10,11,12]]
#x = np.array(x)
#mask1 = np.random.rand(*x.shape)
#mask = (mask1 < 0.5) / 0.5
#
##mask2 = np.random.rand(x.shape)
##print (mask2)
#print (*x.shape)
#print(x)
#print (mask1)
#print (mask)    
#print (x.shape)    
#out = x*mask    
#print(out)


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys: (p, test/train, seed)
    
    Ouputs:
        (out, cache)
    """
    
#    Get the current dropout mode, p, and seed
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    
#    Initialization of outputs and mask
    mask = None
    out = None
    
    if mode == 'train':
        #   create and apply mask (normally p = 0.5for half of neurons), we scale all
        #   by p to avoid having to multiply by p on backpropagation, this is called 
        #   inverted dropout

        mask = (np.random.rand(*x.shape) < p) / p
        
        #   Apply mask
        out = x * mask
    elif mode == 'test':
        # during prediction no mask is used
        mask = None
        out = x
        
    #    Save mask and dropout parameters for backprop
    cache = (dropout_param, mask)
    
    #   convert "out" type and return ouput and cache
    out = out.astype(x.dtype, copy = False)
    return out, cache
        
    
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    Inputs:
        - dout: upstream derivates, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
    """
    
    #   recover dropout parameters (p, mask, mode) from cache
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    # Back propagate (Dropout laer has not parameters just input X)
    
    if mode == 'train':
        # just back propagate dout from teh neurons that were used during dropout
        dx = dout*mask
    elif mode =='test':
        # disable droput dring prediction / test
        dx = dout
        
    # return dx
    return dx


#experiment with padding functionality of numpy
#x=[[[7,8,9],[9,10,11], [1,2,3]]]
#P = 3
#y = np.lib.pad(x,((2,1),(0,1),(1,1)), 'constant', constant_values = 0)
#print (y)

    
def conv_forward_naive(x, w, b, conv_params):
    """
    computes the forward pass for the convolution layer (naive)
    Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A Dictionary with the following keys:
            - 'stride': How much pixels the sliding window will travel
            - 'pad': The number of pixels that will be used to zero pad the input
            
        N: Mini-batch size
        C: Input depth (i.e. 3 for RGB images)
        H/W: Image height / width
        F: number of filters on convolution layer (will be the output depth)
        HH/WW: Kernel Height / Width
        
        Returns a tuple of:
            - out: output data, of shape (N, F, H', W') where H' and w'are given by
                H' = 1 + (H + 2 * pad - HH) / stride
                W' = 1 + (W + 2 * pad - WW) / stride
            - cache: (x, w, b, conv_param)
            
    """
    
    out = None
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    # Get parameters
    
    P = conv_params["pad"]
    S = conv_params["stride"]

    #Calculate output size and initialize output volume
    
    H_R = 1 + (H + 2 * P - HH) / S
    W_R = 1 + (W + 2 * P - WW) / S

    out = np.zeros(N,F,H_R, W_R)
    
    #pad images with zeroes on the border (used to keep spatial information)
    
    x_pad = np.lib.pad(x((0,0),(0,0),(P,P),(P,P)), 'constant', constant_values = 0)
    
#    Apply the convolution
    for n in xrange(N): # for each element on batch
        for depth in xrange(F): # for each input depth
            for r in xrange(0,H,S): # slide vertically taking stride into account
                for c in xrange(0,W,S): # slide horizontally taking stride into account
                    out[n, depth, r/S, c/S] = np.sum (x_pad[n, :, r:r+HH, c:c+WW] * W[depth, :, :,:]) + b[depth]
    
    # cache parameters and input for backpropagation and return output volume
    cache = (x,w,b,conv_param)
    return out,cache
    
def conv_backward_naive(dout, cache):
    
    """
    computes the backward pass for the convolution layer. (naive)
    Inputs:
        - dout: upstream derivatives.
        - cache: a tuple of (x,w,b,conv_param)
    
    Returns a tuple of (dw, dx, db) gradients
    """
    
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    N, F, H_R, W_R = dout.shape
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    P = conv_param["pad"]
    S = conv_param["stride"]

    # Do zero padding on x_pad
    
    x_pad = np.lib.pad(x,((0,0),(0,0),(P,P),(P,P)), 'constant', constant_values=0)
    
    # initiaalise outputs
    dx = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    # Calculate dx with 2 extra col/row that will be deleted
    for n in xrange(N): # for each element on batch
        for depth in xrange(F): # for each filter
            for r in xrange(0,H,S): # slide vertically taking stride into account
                for c in xrange(0,W,S): # slide horizontally taking stride into account
                    dx[n, :, r:r+HH, c:c+WW] += dout[n,depth,r/S,c/S] * w[depth,:,:,:]

    #deleting padded rows to match real dx
    delete_rows = range(P) + range(H+P, H+2*p,1)
    delete_columns = range(P) + range(W+P, W+2*P,1)
    dx = np.delete(dx, delete_rows, axis=2) #height
    dx = np.delete(dx, delete_columns, axis=3) #width
    
    # Calculate dw
    for n in xrange(N): # for each element on batch
        for depth in xrange(F): # for each filter
            for r in xrange(H_R): #slide vertically taking stride into account
                for c in xrange(W_R): # slide horizontally taking stride into account
                    dw[depth,:,:,:] += dout[n, depth, r,c] * x_pad[n,:,r*S:r*S+HH, c*S:c*S+WW]

    # Calculate db, 1 scalar bias per filter, so its just a matter of summing
    for depth in range(F):
        db[depth] = np.sum(dout[:depth, :, :])
    
    return dx, dw, db
    

                    
                
            
        
                    
                
            
        
    