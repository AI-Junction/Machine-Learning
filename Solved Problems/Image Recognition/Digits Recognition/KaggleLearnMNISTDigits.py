# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:19:37 2017

@author: Chandrakant Pattekar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:54:11 2017

@author: Chandrakant Pattekar
"""
#%%
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
plt.ion()

#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test_pd = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllDigitsData\\test.csv')

print(type(X_train))

X_train = X_train.reshape(60000, 784)
print(X_train.shape)
X_test_k = np.array(X_test_pd).reshape(28000, 784)
X_test = np.array(X_test).reshape(10000, 784)
print(X_test_k.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_test_k = X_test_k.astype('float32')

X_train /= 255
X_test_k /= 255
X_test /= 255


print(X_train.shape[0], 'Train samples')
print(X_test_k.shape[0], 'Test samples')
print(X_test.shape[0], 'Validation samples')

#%%
plt.pcolor(X_test_k[5].reshape(28,28))

#z = X_train[5].reshape(28,28)

#print(z.shape)
#print(z)


#plt.scatter(z[0], z[1], color = 'r', size = 10, lineswidth = 15)
#plt.show()

#convert class vectors to binary class matrices

#%%

print(y_test)
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(y_train.shape)
print(Y_train.shape)

print(Y_train[0])
print(y_train[0])

model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.summary()
model.add(Activation('sigmoid'))

#suppose I accidentally add an activation layer in the model. Can I remove it
#model.add(Activation('sigmoid'))
model.summary()
#yes I can remove the layer accidentally added with below line of code!
#model.layers = model.layers[:-1]
#model.summary()
model.add(Dense(10))
model.add(Activation('softmax'))
sgd = SGD()
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')
h=model.fit(X_train, Y_train, batch_size = 128, nb_epoch = 5, 
          validation_data = (X_test, Y_test), verbose=1, show_accuracy = True)
model.summary()
print(h.history)

score = model.evaluate(X_test, Y_test, verbose=1, show_accuracy = True)

print(score)
plt.pcolor(X_test[0].reshape(28,28))
print(Y_test[0])
print(X_test[0].shape)

y_pred = model.predict_classes(X_test_k)

print(y_pred[:25])
#print(y_test[0:25])


ImageId = [x+1 for x in range(len(X_test_k))]
           
result = pd.DataFrame({
        "ImageId": ImageId,
        "Label": np.array(y_pred)[:]
    })
result.to_csv('mnist_result.csv', index=False)

#%%


fig, axes = plt.subplots(12,12, figsize = (28,28))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

print(axes.shape)


# Plot the impages starting from i = 1
for i, ax in enumerate(axes.flat):
    a = i
    im = np.reshape(X_test_k[a], (28,28))
    ax.imshow(im, cmap = 'binary')
    ax.text(0.95, 0.05, 'n={0}'.format(model.predict_classes(X_test_k[i:i+1])), ha='right', 
            transform = ax.transAxes, color = 'blue')
    
    ax.set_xticks([])
    ax.set_yticks([])



wts = np.array(model.get_weights())
#print(wts)
print(wts.shape)
#print(wts[0].shape)
#print(wts[1].shape)
#print(wts[2].shape)
#print(wts[3].shape)
#print(wts[2])

W1,b1,W2,b2 = model.get_weights()
print(W1, b1, W2, b2)
print(W1.shape)
print(b1.shape)
print(W2.shape)
print(b2.shape)

ax, ay = (16,8)

#f, con = plt.subplots(ax, ay, sharex = 'col', sharey = 'row')
f, con = plt.subplots(ax, ay)
print(type(con))
print(con.shape)
print(type(f))

print (W1[:,1])
W1Reshape = np.reshape(W1[:,1], (28,28))

print(W1Reshape)

for xx in range(ax):
    for yy in range(ay):
        print(8*xx + yy)
        con[xx, yy].pcolormesh(W1 [:, 8*xx + yy].reshape(28,28))
