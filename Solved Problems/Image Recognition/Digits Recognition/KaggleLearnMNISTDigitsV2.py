# -*- coding: utf-8 -*-

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
'''

#df = pd.read_csv('../input/train.csv')
#df2 = pd.read_csv('../input/test.csv')


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


batch_size = 128
nb_classes = 10
nb_epoch = 40
#nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

df = pd.DataFrame()

df = X_train

X_test_pd = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllDigitsData\\test.csv')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)



print(len(X_train), len(X_test))
print(K.image_dim_ordering())

X_test_k = np.array(X_test_pd).reshape(28000, 784)


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test_k = X_test_k.reshape(X_test_k.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_test_k = X_test_k.reshape(X_test_k.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_test_k = X_test_k.astype('float32')
X_train /= 255
X_test /= 255
X_test_k /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'validation samples')
print(X_test_k.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict_classes(X_test_k)

print(y_pred[:25])
#print(y_test[0:25])


ImageId = [x+1 for x in range(len(X_test_k))]
           
result = pd.DataFrame({
        "ImageId": ImageId,
        "Label": np.array(y_pred)[:]
    })
result.to_csv('2017-06-12-mnist_result.csv', index=False)


fig, axes = plt.subplots(12,12, figsize = (28,28))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)

print(axes.shape)


# Plot the impages starting from i = 1
for i, ax in enumerate(axes.flat):
    a = i+100
    im = np.reshape(X_test_k[a], (28,28))
    ax.imshow(im, cmap = 'binary')
    ax.text(0.95, 0.05, '{0}'.format(y_pred[a]), ha='right', 
            transform = ax.transAxes, color = 'blue', size=20)
    
    ax.set_xticks([])
    ax.set_yticks([])

W1 = model.get_weights()

print(W1)
print(np.array(W1[4]).shape)


n_components = 30
pca = PCA(n_components=n_components).fit(X_test_pd.values)

eigenvalues = pca.components_.reshape(n_components, 28, 28)

# Extracting the PCA components ( eignevalues )
#eigenvalues = pca.components_.reshape(n_components, 28, 28)
eigenvalues = pca.components_

n_row = 4
n_col = 7

# Plot the first 8 eignenvalues
plt.figure(figsize=(8,7))
for i in list(range(n_row * n_col)):
#     for offset in [10, 30,0]:
#     plt.subplot(n_row, n_col, i + 1)
    offset =0
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')
    title_text = 'Eigenvalue ' + str(i + 1)
    plt.title(title_text, size=6.5)
    plt.xticks(())
    plt.yticks(())
plt.show()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 6))
print(X_test_pd.shape)

plt.title('Correlation plot of a 100 columns in the MNIST dataset')
# Draw the heatmap using seaborn
sns.heatmap(X_test_pd.ix[:,0:200].astype(float).corr(),linewidths=0, square=True, cmap="viridis", xticklabels=False, yticklabels= False, annot=True)


"""
************

Second kernel from Bruno


************
"""

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id

y_train = train["price_doc"] * .9691 + 10.08
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)    


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'alpha' : 0.9,
    'lambda' : 10,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('xgbSub.csv', index=False)

