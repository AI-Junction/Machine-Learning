# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:34:50 2017

@author: echtpar
"""

################################


#Peter Giannakopoulos
#1st try with Keras - 0.918 LB
#Peter Giannakopoulos
#Planet: Understanding the Amazon from Space
#voters
#last run 19 days ago · Python script · 252 views
#using data from Planet: Understanding the Amazon from Space ·
#Public

################################

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import cv2
from tqdm import tqdm

from sklearn.metrics import fbeta_score

from keras.applications.inception_v3 import InceptionV3

# Params
input_size = 64
input_channels = 3

epochs = 15
batch_size = 128
learning_rate = 0.001
lr_decay = 1e-4

valid_data_size = 5000  # Samples to withhold for validation




train_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train-jpg\\train-jpg"
test_path_additional = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg-additional\\test-jpg-additional"
test_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg\\test-jpg"
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train_v2.csv\\train_v2.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\sample_submission_v2\\sample_submission_v2.csv")



model = Sequential()
model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
#model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, nb_row=2, nb_col=2, activation='relu'))
#model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(32, nb_row=2, nb_col=2, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, nb_row=2, nb_col=2, activation='relu'))

#model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(64, nb_row=2, nb_col=2, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, nb_row=2, nb_col=2, activation='relu'))

#model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(128, nb_row=2, nb_col=2, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, nb_row=2, nb_col=2, activation='relu'))

#model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(256, nb_row=2, nb_col=2, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

#df_train_data = pd.read_csv('../input/train_v2.csv')
df_train_data = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train_v2.csv\\train_v2.csv')


flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

x_valid = []
y_valid = []

df_valid = df_train_data[(len(df_train_data) - valid_data_size):]

print(train_path + "\\"+ '{}.jpg'.format(f))
                         
for f, tags in tqdm(df_valid.values, miniters=100):
    img = cv2.resize(cv2.imread(train_path + "\\"+ '{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_valid.append(img)
    y_valid.append(targets)

print(img.shape)    
y_valid = np.array(y_valid, np.uint8)
x_valid = np.array(x_valid, np.float32)

x_train = []
y_train = []

df_train = df_train_data[:(len(df_train_data) - valid_data_size)]

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.resize(cv2.imread(train_path + "\\"+ '{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 0)  # flip vertically
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 1)  # flip horizontally
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 0)  # flip vertically
    x_train.append(img)
    y_train.append(targets)

    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)

#df_test_data = pd.read_csv('../input/sample_submission_v2.csv')
df_test_data = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\sample_submission_v2\\sample_submission_v2.csv')
print(df_test_data.shape)


x_test = []

print(test_path+"\\"+'{}.jpg'.format(f))
img = cv2.resize(cv2.imread(test_path+"\\"+'{}.jpg'.format(f)), (input_size, input_size))
plt.imshow(img)
print(type(img))
x_test.append(img)

for f, tags in tqdm(df_test_data.values, miniters=1000):
#    img = cv2.resize(cv2.imread('../input/test-jpg/{}.jpg'.format(f)), (input_size, input_size))
    img = cv2.resize(cv2.imread(test_path+"\\"+'{}.jpg'.format(f)), (input_size, input_size))

    x_test.append(img)

#x_test = np.array(x_test, np.float32)

x_test = np.array(x_test)

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint('weights.h5',
                             save_best_only=True)]

opt = Adam(lr=learning_rate, decay=lr_decay)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          nb_epoch=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_data=(x_valid, y_valid))

p_valid = model.predict(x_valid, batch_size=batch_size)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

y_test = []

p_test = model.predict(x_test, batch_size=batch_size, verbose=2)
y_test.append(p_test)

result = np.array(y_test[0])
result = pd.DataFrame(result, columns=labels)

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)

# 0.918




################################
#Kele XU
#Pretrained ResNet Feature+ XGB
#Kele XU
#Planet: Understanding the Amazon from Space
#voters
#last run 6 days ago · Python script · 443 views
#using data from Planet: Understanding the Amazon from Space ·
#Public
###################################





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input

import scipy
from sklearn.metrics import fbeta_score

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 17

train_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train-jpg\\train-jpg"
test_path_additional = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg-additional\\test-jpg-additional"
test_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg\\test-jpg"
train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train_v2.csv\\train_v2.csv")
test = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\sample_submission_v2\\sample_submission_v2.csv")


#train_path = "../input/train-jpg"
#test_path = "../input/train-jpg/test-jpg"
#train = pd.read_csv("../input/train_v2.csv")
#test = pd.read_csv("../input/sample_submission_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
#print(list(set(l.split(' ') for l in train['tags'][:10].values)))
#print(list(set(flatten([l.split(' ') for l in train['tags'][:10].values]))))

labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
print(len(labels))
#print([x for x in flatten])

label_map = {l: i for i, l in enumerate(labels)}
print(label_map)
             
inv_label_map = {i: l for l, i in label_map.items()}
print(inv_label_map)
                  
# use ResNet50 model extract feature from fc1 layer
#base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
base_model = ResNet50(weights='imagenet', include_top = False)
print(base_model.name)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
print(base_model.summary())
x = Flatten()(x)
print(type(x))
#model = Model(inputs=input, outputs=x)
model = Model(input, x)
print(type(x))

X_train = []
y_train = []

base_model.summary()

#################
#################
#  trial section


#targets = np.zeros(n_classes)
#print(targets)
#print(tags)
#for t in tags.split(' '):
#    targets[label_map[t]] = 1
#print(targets)
#print(label_map['primary'])
#y_train.append(targets)

#img_path = train_path + "\\" + "train_1.jpg"
#img = image.load_img(img_path, target_size=(224, 224))
#plt.imshow(img)
#print(type(img))
#x = image.img_to_array(img)
#print(type(x))
#print(x.shape)
#x = np.expand_dims(x, axis=0)
#
#print(type(x))
#print(x.shape)
#
#x = preprocess_input(x)
#
#print(type(x))
#print(x.shape)
#print(x[0])
#
#
#features = model.predict(x)
#print(features.shape)
#features_reduce =  features.squeeze()
#print(features.shape)
#print(features[0].shape)
#X_train.append(features_reduce)

#################
#################


#print(train[:10])

for f, tags in tqdm(train.values, miniters=1000):
    # preprocess input image
    img_path = train_path + "\\" + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce =  features.squeeze()
    X_train.append(features_reduce)

    # generate one hot vector for label
    targets = np.zeros(n_classes)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)

    
#print(features.shape)    
#print(features_reduce.shape)    
#print(targets.shape)
#print(len(y_train))


X = np.array(X_train)
y = np.array(y_train, np.uint8)

#import matplotlib.pyplot as plt
#f = plt.figure(figsize=(10,10))
#ax = plt.subplot(1,1,1)
#ax.imshow(img)
#plt.show()


#print(img.shape)
#print(img.format)
#print(x.shape)
#print(features.shape)
#print(type(features))
#print(features[:][:1])
#print(features_reduce.shape)
#print(len(X_train))
#print(np.max(list(np.round(X_train[0], decimals=1, out=None))))
#print(X_train[:2][:2])


#print(len(test.values))
#
#print(test.values[:10])

#################
#################
#  trial section

#print(test.values[:10])
#img_path = test_path + "\\"+ "test_1.jpg"
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
## generate feature [4096]
#features = model.predict(x)
#features_reduce = features.squeeze()
#X_test.append(features_reduce)
#z = model.predict_proba(X_test)[:, 1]
#
#print(len(features_reduce))


#X_test.append(features_reduce)

#################
#################

X_test = []


for f, tags in tqdm(test.values, miniters=1000):
    img_path = test_path + "\\"+ "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # generate feature [4096]
    features = model.predict(x)
    features_reduce = features.squeeze()
    X_test.append(features_reduce)


    
    
    
#################
#################
#  trial section
    
#print(len(X_test))
#
#print(X_test[:,1])
#
#y_pred = []    
#
#z = model.predict_proba(X_test)
#
#print(z[:,0])
#print(y[:, class_i])  
#
#print(y[:,2])  
#print(y.shape)
#print(type(y))
#print(len(z))
#model.predict_proba(X_test)[:, 1]

#################
#################


y_pred = {}
print('Training and making predictions')
for class_i in tqdm(range(n_classes), miniters=1):
    model = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=200, \
                              objective='binary:logistic', nthread=-1, \
                              subsample=0.7, colsample_bytree=0.7, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
#    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]
    y_pred[class_i] = model.predict_proba(X_test)[:, 1]
#    y_pred[:, class_i]=z


print(type(y_pred))
#d = {'a':[1,2], 'b':[2,3], 'c':[3,4]}
#print(type(d))    
#
#df_d= pd.DataFrame(d)

df_y_pred = pd.DataFrame(y_pred)
df_y_pred.to_csv("predictedClasses.csv")

#print(df_d)

df_y_pred = pd.DataFrame(y_pred)

print(df_y_pred.shape)

preds = []
scores = []
#for y_pred_row in y_pred:
#    result = []
#    full_result = []
#    for i, value in enumerate(y_pred_row):
#        full_result.append(str(i))
#        full_result.append(str(value))
#        if value > 0.2:
#            result.append(labels[i])
#    preds.append(" ".join(result))
#    scores.append(" ".join(full_result))

#print (df_y_pred.columns)
#for row in df_y_pred.iterrows():
#    print(row[0], row[1])
#
#print(np.arange(0,10,2))
#print([x for x in range(0,10,2)])

np_y_pred = np.array(df_y_pred)

for j, row in enumerate(np_y_pred):
    result = []
    full_result = []
    for i, value in enumerate(row):
#        print(element)
        full_result.append(str(i))
        full_result.append(str(value))
        if value > 0.2:
            result.append(labels[i])
    preds.append(" ".join(result))
    scores.append(" ".join(full_result))
            

#print(preds)
#            
#print(full_result)            
#print(result)            
#
#for i, row in df_y_pred:
#    print(row)
#    
#for i in range(0, len(df_y_pred)):
#    print(df_y_pred.iloc[i][0], df_y_pred.iloc[i][1], df_y_pred.iloc[i][2])
#    
#print(labels)    

#for y_pred_row in df_y_pred:
#    result = []
#    full_result = []
#    for i, value in enumerate(y_pred_row):
#        full_result.append(str(i))
#        full_result.append(str(value))
#        if value > 0.2:
#            result.append(labels[i])
#    preds.append(" ".join(result))
#    scores.append(" ".join(full_result))
#
#    
#print(y_pred_row)
    
orginin = pd.DataFrame()
orginin['image_name'] = test.image_name.values
#orginin['tags'] = scores
#print(list(preds))
orginin['tags'] = preds
print(orginin)
orginin.to_csv('ResNet_XGB_result.csv', index=False)





##############################
#
#
#    Learn ResNet50 - Start
#
#
##############################


#from resnet50 import ResNet50
#from keras.preprocessing import image
#from imagenet_utils import preprocess_input, decode_predictions
#import numpy as np

#model = ResNet50(weights='imagenet')

#img_path = 'ele.jpg'

img_path1 = 'C:\\Users\\Public\\Pictures\\Sample Pictures\\Penguins.jpg'

img1 = image.load_img(img_path1, target_size=(224, 224))
x = image.img_to_array(img1)
#print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)
print(x[0,:2,:2,:3])

x = preprocess_input(x)
print(x.shape)
print(x[0,:2,:2,:3])


preds = model.predict(x)
print(preds.shape)
preds_array = pd.DataFrame()
preds_array = pd.DataFrame(preds)
print(type(preds), type(preds_array))
preds_array = pd.concat([preds_array, pd.DataFrame(preds)], axis = 0)
print(preds_array.shape)
#preds =  preds.squeeze()
print('Predicted:', decode_predictions(preds_array))
print(preds)

##############################
#
#
#    Learn ResNet50 - End
#
#
##############################







#############################
#Fan Fei Chong
#Keras Pre-trained Inception_v3 in Notebook
#L
#forked from Keras Starter Code in Notebook by Fan Fei Chong (+0/-213/~1)
#Fan Fei Chong
#Planet: Understanding the Amazon from Space
#voters
#last run 14 days ago · Python notebook · 329 views
#using data from Planet: Understanding the Amazon from Space ·
#Public
#############################


#Model: Inception_v3 pre-trained

#Function: Find Best Threshold

#Output:

 #   A sample submission using simple Keras CNN 2)
 #   Predicted probability and truth labels for use in Best Threshold finding

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pickle as pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
import datetime as dt
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Using TensorFlow backend.

#sample_submission_v2.csv
#test-jpg-v2
#test-tif-v3
#test_v2_file_mapping.csv
#train-jpg
#train-tif-v2
#train_v2.csv

# Import Model Specific packages
from keras.preprocessing.image import img_to_array, load_img

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# import packages for InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

from multiprocessing import Pool, cpu_count

# callback for saving models, early stopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# for plotting model training history
import matplotlib.pyplot as plt

# os.chdir('C:/deep_learning') # This is where the input dataset is stored
# os.getcwd()
print("------Fan Fei's Imports Complete-----")

#------Fan Fei's Imports Complete-----

random_seed = 987654321
random.seed(random_seed)
np.random.seed(random_seed)
input_dim = 299

x_train0 = []
x_test0 = []
y_train0 = []

df_train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\train.csv")[:256]
print(df_train.shape)

#df_train = pd.read_csv('../input/train_v2.csv')[:256]        # just get the first 256 images

labels = df_train['tags'].str.get_dummies(sep=' ').columns

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

#img = cv2.imread('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train-jpg\\train-jpg\\train_0.jpg')                 

                 
for f, tags in tqdm(df_train.values, miniters=1000):

#C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train-jpg\\train-jpg\\

#    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
    img = cv2.imread('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\train-jpg\\train-jpg\\{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train0.append(cv2.resize(img, (input_dim, input_dim)))
    y_train0.append(targets)
    
y_train0 = np.array(y_train0, np.uint8)
x_train0 = np.array(x_train0, np.float16) / 255.

print(x_train0.shape)
print(y_train0.shape)

#100%|██████████| 256/256 [00:00<00:00, 403.97it/s]

(256, 299, 299, 3)
(256, 17)

#C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\sample_submission.csv\\sample_submission.csv

#df_test = pd.read_csv('../input/sample_submission_v2.csv')[:256]        # just get the first 256 images
df_test = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\sample_submission.csv\\sample_submission.csv')[:256]        # just get the first 256 images

#C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg\\test-jpg

for f, tags in tqdm(df_test.values, miniters=1000):
#    img = cv2.imread('../input/test-jpg-v2/{}.jpg'.format(f))
    img = cv2.imread('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\All Data\\test-jpg\\test-jpg\\{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_test0.append(cv2.resize(img, (input_dim, input_dim)))
    
x_test0 = np.array(x_test0, np.float16) / 255.

#100%|██████████| 256/256 [00:00<00:00, 393.08it/s]

split = 192
# split = 35000
x_train, x_valid, y_train, y_valid = x_train0[:split], x_train0[split:], y_train0[:split], y_train0[split:]

# create the base pre-trained model
#base_model = InceptionV3(weights=None, include_top=False, input_shape=(input_dim,input_dim,3))
# no weight initialization because Kaggle kernel is isolated from the internet, cannot download
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(input_dim,input_dim,3))

# add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(17, activation='sigmoid')(x)

# let's visualize layer names and layer indices to see how many layers 
# we should freeze
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

#0 input_1
#1 conv2d_1
#2 batch_normalization_1
#3 activation_1
#4 conv2d_2
#5 batch_normalization_2
#6 activation_2
#7 conv2d_3
#8 batch_normalization_3
#9 activation_3
#10 max_pooling2d_1
#11 conv2d_4
#12 batch_normalization_4
#13 activation_4
#14 conv2d_5
#15 batch_normalization_5
#16 activation_5
#17 max_pooling2d_2
#18 conv2d_9
#19 batch_normalization_9
#20 activation_9
#21 conv2d_7
#22 conv2d_10
#23 batch_normalization_7
#24 batch_normalization_10
#25 activation_7
#26 activation_10
#27 average_pooling2d_1
#28 conv2d_6
#29 conv2d_8
#30 conv2d_11
#31 conv2d_12
#32 batch_normalization_6
#33 batch_normalization_8
#34 batch_normalization_11
#35 batch_normalization_12
#36 activation_6
#37 activation_8
#38 activation_11
#39 activation_12
#40 mixed0
#41 conv2d_16
#42 batch_normalization_16
#43 activation_16
#44 conv2d_14
#45 conv2d_17
#46 batch_normalization_14
#47 batch_normalization_17
#48 activation_14
#49 activation_17
#50 average_pooling2d_2
#51 conv2d_13
#52 conv2d_15
#53 conv2d_18
#54 conv2d_19
#55 batch_normalization_13
#56 batch_normalization_15
#57 batch_normalization_18
#58 batch_normalization_19
#59 activation_13
#60 activation_15
#61 activation_18
#62 activation_19
#63 mixed1
#64 conv2d_23
#65 batch_normalization_23
#66 activation_23
#67 conv2d_21
#68 conv2d_24
#69 batch_normalization_21
#70 batch_normalization_24
#71 activation_21
#72 activation_24
#73 average_pooling2d_3
#74 conv2d_20
#75 conv2d_22
#76 conv2d_25
#77 conv2d_26
#78 batch_normalization_20
#79 batch_normalization_22
#80 batch_normalization_25
#81 batch_normalization_26
#82 activation_20
#83 activation_22
#84 activation_25
#85 activation_26
#86 mixed2
#87 conv2d_28
#88 batch_normalization_28
#89 activation_28
#90 conv2d_29
#91 batch_normalization_29
#92 activation_29
#93 conv2d_27
#94 conv2d_30
#95 batch_normalization_27
#96 batch_normalization_30
#97 activation_27
#98 activation_30
#99 max_pooling2d_3
#100 mixed3
#101 conv2d_35
#102 batch_normalization_35
#103 activation_35
#104 conv2d_36
#105 batch_normalization_36
#106 activation_36
#107 conv2d_32
#108 conv2d_37
#109 batch_normalization_32
#110 batch_normalization_37
#111 activation_32
#112 activation_37
#113 conv2d_33
#114 conv2d_38
#115 batch_normalization_33
#116 batch_normalization_38
#117 activation_33
#118 activation_38
#119 average_pooling2d_4
#120 conv2d_31
#121 conv2d_34
#122 conv2d_39
#123 conv2d_40
#124 batch_normalization_31
#125 batch_normalization_34
#126 batch_normalization_39
#127 batch_normalization_40
#128 activation_31
#129 activation_34
#130 activation_39
#131 activation_40
#132 mixed4
#133 conv2d_45
#134 batch_normalization_45
#135 activation_45
#136 conv2d_46
#137 batch_normalization_46
#138 activation_46
#139 conv2d_42
#140 conv2d_47
#141 batch_normalization_42
#142 batch_normalization_47
#143 activation_42
#144 activation_47
#145 conv2d_43
#146 conv2d_48
#147 batch_normalization_43
#148 batch_normalization_48
#149 activation_43
#150 activation_48
#151 average_pooling2d_5
#152 conv2d_41
#153 conv2d_44
#154 conv2d_49
#155 conv2d_50
#156 batch_normalization_41
#157 batch_normalization_44
#158 batch_normalization_49
#159 batch_normalization_50
#160 activation_41
#161 activation_44
#162 activation_49
#163 activation_50
#164 mixed5
#165 conv2d_55
#166 batch_normalization_55
#167 activation_55
#168 conv2d_56
#169 batch_normalization_56
#170 activation_56
#171 conv2d_52
#172 conv2d_57
#173 batch_normalization_52
#174 batch_normalization_57
#175 activation_52
#176 activation_57
#177 conv2d_53
#178 conv2d_58
#179 batch_normalization_53
#180 batch_normalization_58
#181 activation_53
#182 activation_58
#183 average_pooling2d_6
#184 conv2d_51
#185 conv2d_54
#186 conv2d_59
#187 conv2d_60
#188 batch_normalization_51
#189 batch_normalization_54
#190 batch_normalization_59
#191 batch_normalization_60
#192 activation_51
#193 activation_54
#194 activation_59
#195 activation_60
#196 mixed6
#197 conv2d_65
#198 batch_normalization_65
#199 activation_65
#200 conv2d_66
#201 batch_normalization_66
#202 activation_66
#203 conv2d_62
#204 conv2d_67
#205 batch_normalization_62
#206 batch_normalization_67
#207 activation_62
#208 activation_67
#209 conv2d_63
#210 conv2d_68
#211 batch_normalization_63
#212 batch_normalization_68
#213 activation_63
#214 activation_68
#215 average_pooling2d_7
#216 conv2d_61
#217 conv2d_64
#218 conv2d_69
#219 conv2d_70
#220 batch_normalization_61
#221 batch_normalization_64
#222 batch_normalization_69
#223 batch_normalization_70
#224 activation_61
#225 activation_64
#226 activation_69
#227 activation_70
#228 mixed7
#229 conv2d_73
#230 batch_normalization_73
#231 activation_73
#232 conv2d_74
#233 batch_normalization_74
#234 activation_74
#235 conv2d_71
#236 conv2d_75
#237 batch_normalization_71
#238 batch_normalization_75
#239 activation_71
#240 activation_75
#241 conv2d_72
#242 conv2d_76
#243 batch_normalization_72
#244 batch_normalization_76
#245 activation_72
#246 activation_76
#247 max_pooling2d_4
#248 mixed8
#249 conv2d_81
#250 batch_normalization_81
#251 activation_81
#252 conv2d_78
#253 conv2d_82
#254 batch_normalization_78
#255 batch_normalization_82
#256 activation_78
#257 activation_82
#258 conv2d_79
#259 conv2d_80
#260 conv2d_83
#261 conv2d_84
#262 average_pooling2d_8
#263 conv2d_77
#264 batch_normalization_79
#265 batch_normalization_80
#266 batch_normalization_83
#267 batch_normalization_84
#268 conv2d_85
#269 batch_normalization_77
#270 activation_79
#271 activation_80
#272 activation_83
#273 activation_84
#274 batch_normalization_85
#275 activation_77
#276 mixed9_0
#277 concatenate_1
#278 activation_85
#279 mixed9
#280 conv2d_90
#281 batch_normalization_90
#282 activation_90
#283 conv2d_87
#284 conv2d_91
#285 batch_normalization_87
#286 batch_normalization_91
#287 activation_87
#288 activation_91
#289 conv2d_88
#290 conv2d_89
#291 conv2d_92
#292 conv2d_93
#293 average_pooling2d_9
#294 conv2d_86
#295 batch_normalization_88
#296 batch_normalization_89
#297 batch_normalization_92
#298 batch_normalization_93
#299 conv2d_94
#300 batch_normalization_86
#301 activation_88
#302 activation_89
#303 activation_92
#304 activation_93
#305 batch_normalization_94
#306 activation_86
#307 mixed9_1
#308 concatenate_2
#309 activation_94
#310 mixed10

# this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
model = Model(base_model.input, predictions)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# Read in pre-trained weights - fast
# model.load_weights(obj_save_path + "weights_incv3.best.hdf5")

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=SGD(lr=0.01, momentum=0.9))

# Incorporate Callback features
# Checkpointing 
filepath= "weights_incv3.best.hdf5"
# filepath= obj_save_path + "weights_incv3.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

# Early Stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=5, verbose=0, mode='auto') 

callbacks_list = [checkpoint, earlystop]

# the explicit split approach was taken so as to allow for a local validation
# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)

# Fit the model (Add history so that the history may be saved)
history = model.fit(x_train, y_train,
          batch_size=32,
          nb_epoch=1,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_valid, y_valid))

from sklearn.metrics import fbeta_score

p_train = model.predict(x_train0, batch_size=32,verbose=2)
p_test = model.predict(x_test0, batch_size=32,verbose=2)

#Train on 192 samples, validate on 64 samples
#Epoch 1/1
#160/192 [========================>.....] - ETA: 40s - loss: 0.5246Epoch 00000: val_loss improved from inf to 0.57693, saving model to weights_incv3.best.hdf5
#192/192 [==============================] - 306s - loss: 0.5370 - val_loss: 0.5769

def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def find_f2score_threshold(p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0.1,0.4,0.025) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True: 
        print('Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best

print(fbeta_score(y_train0, np.array(p_train) > 0.2, beta=2, average='samples'))
best_threshold = find_f2score_threshold(p_train, y_train0, try_all=True, verbose=True)

#0.482486116396

#/opt/conda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1122: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in samples with no predicted labels.
#  'precision', 'predicted', average, warn_for)

#Best score:  0.68219  @ threshold = 0.468698

# Saving predicted probability and ground truth for Train Dataset
# Compute the best threshold externally
print(labels)
chk_output = pd.DataFrame()
for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
    chk_output['class %d' % index] = p_train[:,index-1]
chk_output.to_csv('predicted_probability.csv', index=False)
for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
    chk_output['class %d' % index] = y_train0[:,index-1]
chk_output.to_csv('true_label.csv', index=False)

Index(['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
       'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation',
       'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging',
       'slash_burn', 'water'],
      dtype='object')

values_test = (p_test > .222222)*1.0        # before multiplying by 1.0, this appears as an array of True and False
values_test = np.array(values_test, np.uint8)

print(values_test)
# Build Submission, using label outputted from long time ago
test_labels = []
for row in range(values_test.shape[0]):
    test_labels.append(' '.join(labels[values_test[row,:]==1]))
Submission_PDFModel = df_test.copy()
Submission_PDFModel.drop('tags', axis = 1)
Submission_PDFModel['tags'] = test_labels
Submission_PDFModel.to_csv('sub_pretrained_inception_v3_online.csv', index = False)






################################

#Charles Jansen
#U-Net in Keras
#Charles Jansen
#Planet: Understanding the Amazon from Space
#voters
#last run 20 hours ago · Python script · 156 views
#using data from Planet: Understanding the Amazon from Space ·
#Public

#################################






import numpy as np 
import pandas as pd 
import os
import cv2
from tqdm import tqdm

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.merge import concatenate

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split


x_train = []
x_test = []
y_train = []

path = ""
name = "Unet"
weights_path = path + name + '.h5'

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

img_size = 64
channels = 4 #4 for tiff, 3 for jpeg

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('../input/test-tif-v2/{}.tif'.format(f), -1)
    x_test.append(cv2.resize(img, (img_size, img_size)))
x_test  = np.array(x_test, np.float32)/255. 

for f, tags in tqdm(df_train.values, miniters=1000):
    #https://stackoverflow.com/questions/37512119/resize-transparent-image-in-opencv-python-cv2
    #If you load a 4 channel image, the flag -1 indicates that the image is loaded unchanged, so you can load and split all 4 channels directly.
    img = cv2.imread('../input/train-tif-v2/{}.tif'.format(f), -1)#0-1 voir au dessus les 2 comments
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)/255.


print(x_train.shape)
print(y_train.shape)

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)   

print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_val), len(Y_val))


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
    
def get_unet(n_ch,patch_height,patch_width):
    concat_axis = 3

    inputs = Input((patch_height, patch_width, n_ch))
    
    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
    up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
    up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
    up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
    up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

    #ch, cw = get_crop_shape(inputs, conv9)
    #conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
    #conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)
    
    flatten =  Flatten()(conv9)
    Dense1 = Dense(512, activation='relu')(flatten)
    BN =BatchNormalization() (Dense1)
    Dense2 = Dense(17, activation='sigmoid')(BN)
    
    model = Model(input=inputs, output=Dense2)
    
    return model


model = get_unet(channels, img_size, img_size)


epochs_arr  = [   20,      5,      5]
learn_rates = [0.001, 0.0003, 0.0001]

for learn_rate, epochs in zip(learn_rates, epochs_arr):
    if os.path.isfile(weights_path):
        print("loading existing weight for training")
        model.load_weights(weights_path)
    
    opt  = optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=opt,
                  metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                 ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=2)]

    model.fit(x = X_train, y= Y_train, validation_data=(X_val, Y_val),
          batch_size=256, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)

if os.path.isfile(weights_path):
    model.load_weights(weights_path)


p_val = model.predict(X_val, batch_size = 128, verbose=1)
print(fbeta_score(Y_val, np.array(p_val) > 0.2, beta=2, average='samples'))

p_test = model.predict(x_test, batch_size = 128, verbose=1)





result = p_test
result = pd.DataFrame(result, columns = labels)

from tqdm import tqdm
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index))) 
    
df_test['tags'] = preds
df_test.to_csv('F:/DS-main/Kaggle-main/Planet Understanding the Amazon from Space/submission_unet.csv', index=False)
