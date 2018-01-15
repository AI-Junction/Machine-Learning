# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:39:54 2017

@author: echtpar
"""

##########################################
##
## One of the winning solutions 
##
## Peter Giannakopoulos
## Keras VGG19 [0.93028 Private LB]
## Peter Giannakopoulos
## Planet: Understanding the Amazon from Space
## voters
## last run 3 days ago · Python script · 866 views
## using data from Planet: Understanding the Amazon from Space ·
## Public
##########################################



from collections import Counter

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.applications.vgg19 import VGG19

import cv2
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score

input_size = 128
input_channels = 3

epochs = 50
batch_size = 128

n_folds = 5

training = True

ensemble_voting = False  # If True, use voting for model ensemble, otherwise use averaging

df_train_data = pd.read_csv('input/train_v2.csv')
df_test_data = pd.read_csv('input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

fold_count = 0

y_full_test = []
thres_sum = np.zeros(17, np.float32)

for train_index, test_index in kf.split(df_train_data):

    fold_count += 1
    print('Fold ', fold_count)


    def transformations(src, choice):
        if choice == 0:
            # Rotate 90
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        if choice == 1:
            # Rotate 90 and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            src = cv2.flip(src, flipCode=1)
        if choice == 2:
            # Rotate 180
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        if choice == 3:
            # Rotate 180 and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
            src = cv2.flip(src, flipCode=1)
        if choice == 4:
            # Rotate 90 counter-clockwise
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        if choice == 5:
            # Rotate 90 counter-clockwise and flip horizontally
            src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            src = cv2.flip(src, flipCode=1)
        return src

    df_valid = df_train_data.ix[test_index]
    print('Validating on {} samples'.format(len(df_valid)))


    def valid_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for f, tags in df_valid_batch.values:
                    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, np.random.randint(6))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    df_train = df_train_data.ix[train_index]
    if training:
        print('Training on {} samples'.format(len(df_train)))


    def train_generator():
        while True:
            for start in range(0, len(df_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_train))
                df_train_batch = df_train[start:end]
                for f, tags in df_train_batch.values:
                    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, np.random.randint(6))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    base_model = VGG19(include_top=False,
                       weights='imagenet',
                       input_shape=(input_size, input_size, input_channels))

    model = Sequential()
    # Batchnorm input
    model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
    # Base model
    model.add(base_model)
    # Classifier
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))

    opt = Adam(lr=1e-4)

    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=4,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=2,
                                   cooldown=2,
                                   verbose=1),
                 ModelCheckpoint(filepath='weights/best_weights.fold_' + str(fold_count) + '.hdf5',
                                 save_best_only=True,
                                 save_weights_only=True)]

    if training:
        model.fit_generator(generator=train_generator(),
                            steps_per_epoch=(len(df_train) // batch_size) + 1,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=valid_generator(),
                            validation_steps=(len(df_valid) // batch_size) + 1)


    def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
        def mf(x):
            p2 = np.zeros_like(p)
            for i in range(17):
                p2[:, i] = (p[:, i] > x[i]).astype(np.int)
            score = fbeta_score(y, p2, beta=2, average='samples')
            return score

        x = [0.2] * 17
        for i in range(17):
            best_i2 = 0
            best_score = 0
            for i2 in range(resolution):
                i2 /= float(resolution)
                x[i] = i2
                score = mf(x)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)
        return x


    # Load best weights
    model.load_weights(filepath='weights/best_weights.fold_' + str(fold_count) + '.hdf5')

    p_valid = model.predict_generator(generator=valid_generator(),
                                      steps=(len(df_valid) // batch_size) + 1)

    y_valid = []
    for f, tags in df_valid.values:
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_valid.append(targets)
    y_valid = np.array(y_valid, np.uint8)

    # Find optimal f2 thresholds for local validation set
    thres = optimise_f2_thresholds(y_valid, p_valid, verbose=False)

    print('F2 = {}'.format(fbeta_score(y_valid, np.array(p_valid) > thres, beta=2, average='samples')))

    thres_sum += np.array(thres, np.float32)


    def test_generator(transformation):
        while True:
            for start in range(0, len(df_test_data), batch_size):
                x_batch = []
                end = min(start + batch_size, len(df_test_data))
                df_test_batch = df_test_data[start:end]
                for f, tags in df_test_batch.values:
                    img = cv2.imread('input/test-jpg/{}.jpg'.format(f))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, transformation)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                yield x_batch

    # 6-fold TTA
    p_full_test = []
    for i in range(6):
        p_test = model.predict_generator(generator=test_generator(transformation=i),
                                         steps=(len(df_test_data) // batch_size) + 1)
        p_full_test.append(p_test)

    p_test = np.array(p_full_test[0])
    for i in range(1, 6):
        p_test += np.array(p_full_test[i])
    p_test /= 6

    y_full_test.append(p_test)

result = np.array(y_full_test[0])
if ensemble_voting:
    for f in range(len(y_full_test[0])):  # For each file
        for tag in range(17):  # For each tag
            preds = []
            for fold in range(n_folds):  # For each fold
                preds.append(y_full_test[fold][f][tag])
            pred = Counter(preds).most_common(1)[0][0]  # Most common tag prediction among folds
            result[f][tag] = pred
else:
    for fold in range(1, n_folds):
        result += np.array(y_full_test[fold])
    result /= n_folds
result = pd.DataFrame(result, columns=labels)

preds = []
thres = (thres_sum / n_folds).tolist()

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)













################################################################
"""
Sasha Korekov
End-to-End ResNet50 with TTA [LB ~0.93]
Sasha Korekov
Planet: Understanding the Amazon from Space
voters
last run 4 days ago · Python script · 765 views
using data from Planet: Understanding the Amazon from Space ·
Public
"""
################################################################






from __future__ import division
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Merge, merge
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from multi_gpu import make_parallel #Available here https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
from sklearn.utils import shuffle
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import fbeta_score
from keras.optimizers import Adam, SGD

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def fbeta_loss(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = 1 - (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result

def fbeta_score_K(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result

def rotate(img):
    rows = img.shape[0]
    cols = img.shape[1]
    angle = np.random.choice((10, 20, 30))#, 40, 50, 60, 70, 80, 90))
    rotation_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_M, (cols, rows))
    return img

def rotate_bound(image, size):
    #credits http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    angle = np.random.randint(10,180)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    output = cv2.resize(cv2.warpAffine(image, M, (nW, nH)), (size, size))
    return output

def perspective(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shrink_ratio1 = np.random.randint(low=85, high=110, dtype=int) / 100
    shrink_ratio2 = np.random.randint(low=85, high=110, dtype=int) / 100

    zero_point = rows - np.round(rows * shrink_ratio1, 0)
    max_point_row = np.round(rows * shrink_ratio1, 0)
    max_point_col = np.round(cols * shrink_ratio2, 0)

    src = np.float32([[zero_point, zero_point], [max_point_row-1, zero_point], [zero_point, max_point_col+1], [max_point_row-1, max_point_col+1]])
    dst = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

    perspective_M = cv2.getPerspectiveTransform(src, dst)

    img = cv2.warpPerspective(img, perspective_M, (cols,rows))#, borderValue=mean_pix)
    return img

def shift(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shift_ratio1 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)
    shift_ratio2 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)

    shift_M = np.float32([[1,0,shift_ratio1], [0,1,shift_ratio2]])
    img = cv2.warpAffine(img, shift_M, (cols, rows))#, borderValue=mean_pix)
    return img

def batch_generator_train(zip_list, img_size, batch_size, is_train=True, shuffle=True):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle == True:
        random.shuffle(zip_list)
    counter = 0
    while True:
        if shuffle == True:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image = cv2.imread(file) #cv2.resize(cv2.imread(file), (img_size,img_size)) / 255.
            image = image[:, :, [2, 1, 0]] - mean_pix

            rnd_flip = np.random.randint(2, dtype=int)
            rnd_rotate = np.random.randint(2, dtype=int)
            rnd_zoom = np.random.randint(2, dtype=int)
            rnd_shift = np.random.randint(2, dtype=int)

            if (rnd_flip == 1) & (is_train == True):
                rnd_flip = np.random.randint(3, dtype=int) - 1
                image = cv2.flip(image, rnd_flip)

            if (rnd_rotate == 1) & (is_train == True):
                image = rotate_bound(image, img_size)

            if (rnd_zoom == 1) & (is_train == True):
                image = perspective(image)

            if (rnd_shift == 1) & (is_train == True):
                image = shift(image)

            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            if shuffle == True:
                random.shuffle(zip_list)
            counter = 0

def batch_generator_test(zip_list, img_size, batch_size, shuffle=True):
    number_of_batches = np.ceil(len(zip_list)/batch_size)
    print(len(zip_list), number_of_batches)
    counter = 0
    if shuffle:
        random.shuffle(zip_list)
    while True:
        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image = cv2.resize(cv2.imread(file), (img_size, img_size))
            image = image[:, :, [2, 1, 0]] - mean_pix
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            random.shuffle(zip_list)
            counter = 0

def predict_generator(files, img_size, batch_size):
    number_of_batches = np.ceil(len(files) / batch_size)
    print(len(files), number_of_batches)
    counter = 0
    int_counter = 0

    while True:
            beg = batch_size * counter
            end = batch_size * (counter + 1)
            batch_files = files[beg:end]
            image_list = []

            for file in batch_files:
                int_counter += 1
                image = cv2.resize(cv2.imread(file), (img_size, img_size))
                image = image[:, :, [2, 1, 0]] - mean_pix

                rnd_flip = np.random.randint(2, dtype=int)
                rnd_rotate = np.random.randint(2, dtype=int)
                rnd_zoom = np.random.randint(2, dtype=int)
                rnd_shift = np.random.randint(2, dtype=int)

                if rnd_flip == 1:
                    rnd_flip = np.random.randint(3, dtype=int) - 1
                    image = cv2.flip(image, rnd_flip)

                if rnd_rotate == 1:
                    image = rotate_bound(image, img_size)

                if rnd_zoom == 1:
                    image = perspective(image)

                if rnd_shift == 1:
                    image = shift(image)

                image_list.append(image)

            counter += 1

            image_list = np.array(image_list)

            yield (image_list)


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    score = fbeta_score(y_true, y_pred, beta=2, average='samples')
    return score

GLOBAL_PATH = 'D:/G/Amazon/'
TRAIN_FOLDER = 'D:/g/amazon/train-jpg-res/' #All train files resized to 224*224
TEST_FOLDER = 'D:/G/Amazon/test-jpg/' #All test files in one folder
F_CLASSES = GLOBAL_PATH + 'train_v2.csv'

df_train = pd.read_csv(F_CLASSES)
df_test = pd.read_csv(GLOBAL_PATH + 'sample_submission_v4.csv')

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

flatten = lambda l: [item for sublist in l for item in sublist]

x_train = []
x_test = []
y_train = []


for f, tags in tqdm(df_train.values, miniters=1000):
    img = TRAIN_FOLDER + '{}.jpg'.format(f)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)


x_train, x_holdout, y_train, y_holdout = x_train[3000:-1], x_train[:3000], y_train[3000:-1], y_train[:3000]

x_train, y_train = shuffle(x_train, y_train, random_state = 24)

part = 0.85
split = int(round(part*len(y_train)))
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]
print('x tr: ', len(x_train))

#define callbacks
callbacks = [ModelCheckpoint('amazon_2007.hdf5', monitor='val_loss', save_best_only=True, verbose=2, save_weights_only=False),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0000001),
             EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

BATCH = 128
IMG_SIZE = 224
mean_pix = np.array([102.9801, 115.9465, 122.7717]) #It is BGR

from keras.applications import ResNet50

#Compile model and set non-top layets non-trainable (warm-up)
base_model = ResNet50(include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='avg', weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
output = Dense(17, activation='sigmoid')(x)

optimizer = Adam(0.001, decay=0.0003)
model = Model(inputs=base_model.inputs, outputs=output)
model = make_parallel(model, 2)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])

model.fit_generator(generator=batch_generator_train(list(zip(x_train, y_train)), IMG_SIZE, BATCH),
                          steps_per_epoch=np.ceil(len(x_train)/BATCH),
                          epochs=1,
                          verbose=1,
                          validation_data=batch_generator_train(list(zip(x_valid, y_valid)), IMG_SIZE, 16),
                          validation_steps=np.ceil(len(x_valid)/16),
                          callbacks=callbacks,
                          initial_epoch=0)


#Compile model and set all layers trainable
optimizer = Adam(0.0001, decay=0.00000001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])
model.load_weights('amazon_2007.hdf5', by_name=True)
for layer in base_model.layers:
    layer.trainable = True

BATCH = 32
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])
model.fit_generator(generator=batch_generator_train(list(zip(x_train, y_train)), IMG_SIZE, BATCH),
                          steps_per_epoch=np.ceil(len(x_train)/BATCH),
                          epochs=50,
                          verbose=1,
                          validation_data=batch_generator_train(list(zip(x_valid, y_valid)), IMG_SIZE, 16),
                          validation_steps=np.ceil(len(x_valid)/16),
                          callbacks=callbacks,
                          initial_epoch=0)

model.load_weights('amazon_2007.hdf5')


x_val = []
y_val = []
x_hld = []
y_hld = []
x_test = []
y_test = []


#====================== validation set est =================================
for f, tags in tqdm(list(zip(x_valid, y_valid)), miniters=1000):
    y_val.append(tags)

p_valid = model.predict_generator(batch_generator_test(list(zip(x_valid, y_valid)), IMG_SIZE, 8, shuffle=False), steps=np.ceil(len(x_valid)/8))

print('val_set: ', fbeta_score(np.array(y_val), np.array(p_valid) > 0.2, beta=2, average='samples'))
#===========================================================================

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    #credits https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

X = optimise_f2_thresholds(np.array(y_val), np.array(p_valid))

#====================== holdout set est =================================
for f, tags in tqdm(list(zip(x_holdout, y_holdout)), miniters=1000):
    img = cv2.resize(cv2.imread(f), (IMG_SIZE, IMG_SIZE))
    x_hld.append(img)
    y_hld.append(tags)

if len(x_holdout) % 2 > 0:
    x_hld.append(x_hld[0])
    y_hld.append(y_hld[0])

x_hld = np.array(x_hld, np.float16)

p_valid = model.predict(x_hld, batch_size=28, verbose=2)
print('holdout set: ', f2_score(np.array(y_hld), np.array(p_valid) > 0.2))
print('holdout set w/ thresh: ', f2_score(np.array(y_hld), np.array(p_valid) > 0.19))
#===========================================================================


for f, tags in tqdm(df_test.values, miniters=1000):
    img = TEST_FOLDER + '{}.jpg'.format(f)
    x_test.append(img)

batch_size_test = 32
len_test = len(x_test)
x_tst = []
yfull_test = []


TTA_steps = 30

for k in range(0, TTA_steps):
    print(k)
    probs = model.predict_generator(predict_generator(x_test,IMG_SIZE,batch_size_test), steps=np.ceil(len(x_test)/batch_size_test),verbose=1)
    yfull_test.append(probs)
    k += 1

result = np.array(yfull_test[0])

for i in range(1, TTA_steps):
    result += np.array(yfull_test[i])
result /= TTA_steps

res = pd.DataFrame(result, columns=labels)
preds = []

for i in tqdm(range(res.shape[0]), miniters=1000):
    a = res.ix[[i]]
    a = a.apply(lambda x: x > X, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

print(len(preds))

df_test['tags'] = preds
df_test = df_test[:-57]
df_test.to_csv('submission.csv', index=False)



#################################################################
"""
Tuatini GODARD
[0.92837 on private LB] Solution with Keras
Tuatini GODARD
Planet: Understanding the Amazon from Space
voters
last run 2 days ago · Python notebook · 6306 views
using data from Planet: Understanding the Amazon from Space ·
Public
"""
#################################################################






# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Planet: Understanding the Amazon deforestation from Space challenge

# <markdowncell>

# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **

# <markdowncell>

# Start by adding the helper files to the python path

# <codecell>

import sys

sys.path.append('../src')
sys.path.append('../tests')

# <markdowncell>

# ## Import required modules

# <codecell>

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import ModelCheckpoint, CSVLogger

import data_helper
from data_helper import AmazonPreprocessor
from keras_helper import AmazonKerasClassifier, Fbeta
from kaggle_data.downloader import KaggleDataDownloader

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# <markdowncell>

# Print tensorflow version for reuse (the Keras module is used directly from the tensorflow framework)

# <codecell>

tf.__version__

# <markdowncell>

# ## Download the competition files
# Download the dataset files and extract them automatically with the help of [Kaggle data downloader](https://github.com/EKami/kaggle-data-downloader)

# <codecell>

competition_name = "planet-understanding-the-amazon-from-space"

train, train_u = "train-jpg.tar.7z", "train-jpg.tar"
test, test_u = "test-jpg.tar.7z", "test-jpg.tar"
test_additional, test_additional_u = "test-jpg-additional.tar.7z", "test-jpg-additional.tar"
test_labels = "train_v2.csv.zip"
destination_path = "../input/"
is_datasets_present = False

# If the folders already exists then the files may already be extracted
# This is a bit hacky but it's sufficient for our needs
datasets_path = data_helper.get_jpeg_data_files_paths()
for dir_path in datasets_path:
    if os.path.exists(dir_path):
        is_datasets_present = True

if not is_datasets_present:
    # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
    downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)
    
    train_output_path = downloader.download_dataset(train, destination_path)
    downloader.decompress(train_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + train_u, destination_path) # Extract the content of the previous tar file
    os.remove(train_output_path) # Removes the 7z file
    os.remove(destination_path + train_u) # Removes the tar file
    
    test_output_path = downloader.download_dataset(test, destination_path)
    downloader.decompress(test_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_output_path) # Removes the 7z file
    os.remove(destination_path + test_u) # Removes the tar file
    
    test_add_output_path = downloader.download_dataset(test_additional, destination_path)
    downloader.decompress(test_add_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_additional_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_add_output_path) # Removes the 7z file
    os.remove(destination_path + test_additional_u) # Removes the tar file
    
    test_labels_output_path = downloader.download_dataset(test_labels, destination_path)
    downloader.decompress(test_labels_output_path, destination_path) # Outputs a csv file
    os.remove(test_labels_output_path) # Removes the zip file
else:
    print("All datasets are present.")

# <markdowncell>

# ## Inspect image labels
# Visualize what the training set looks like

# <codecell>

train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

# <markdowncell>

# Each image can be tagged with multiple tags, lets list all uniques tags

# <codecell>

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))

# <markdowncell>

# ### Repartition of each labels

# <codecell>

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')

# <markdowncell>

# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# <codecell>

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))

# <markdowncell>

# # Image resize & validation split
# Define the dimensions of the image data trained by the network. Recommended resized images could be 32x32, 64x64, or 128x128 to speedup the training. 
# 
# You could also use `None` to use full sized images.
# 
# Be careful, the higher the `validation_split_size` the more RAM you will consume.

# <codecell>

img_resize = (128, 128) # The resize size of each image ex: (64, 64) or None to use the default image size
validation_split_size = 0.2

# <markdowncell>

# # Data preprocessing
# Due to the hudge amount of memory the preprocessed images can take, we will create a dedicated `AmazonPreprocessor` class which job is to preprocess the data right in time at specific steps (training/inference) so that our RAM don't get completely filled by the preprocessed images. 
# 
# The only exception to this being the validation dataset as we need to use it as-is for f2 score calculation as well as when we calculate the validation accuracy of each batch.

# <codecell>

preprocessor = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional, 
                                  img_resize, validation_split_size)
preprocessor.init()

# <codecell>

print("X_train/y_train lenght: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val lenght: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/X_test_filename lenght: {}/{}".format(len(preprocessor.X_test), len(preprocessor.X_test_filename)))
preprocessor.y_map

# <markdowncell>

# ## Callbacks
# 
# 
# 
# __Checkpoint__
# 
# Saves the best model weights across all epochs in the training process.
# 
# __CSVLogger__
# 
# Creates a file with a log of loss and accuracy per epoch
# 
# __FBeta__
# 
# Calculates fbeta_score after each epoch during training

# <codecell>

checkpoint = ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

# creates a file with a log of loss and accuracy per epoch
csv = CSVLogger(filename='training.log', append=True)

# Calculates fbeta_score after each epoch during training
fbeta = Fbeta()

# <markdowncell>

# ## Choose Hyperparameters
# 
# Choose your hyperparameters below for training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults.
# 
# If you opted for a high `img_resize` then you may want to lower the `batch_size` to fit your images matrices into the GPU memory. With an `img_resize` of `(256, 256)` (the default size) and a batch_size of `64` you should at least have a GPU with 8GB or VRAM.

# <codecell>

batch_size = 64
epochs_arr = [35, 15, 5]
learn_rates = [0.002, 0.0002, 0.00002]

# <markdowncell>

# ## Define and Train model
# 
# Here we define the model and begin training. 

# <codecell>

classifier = AmazonKerasClassifier(preprocessor)
classifier.add_conv_layer()
classifier.add_flatten_layer()
classifier.add_ann_layer(len(preprocessor.y_map))

train_losses, val_losses = [], []
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(learn_rate, epochs, batch_size, 
                                                                           train_callbacks=[checkpoint, fbeta, csv])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

print(fbeta.fbeta)

# <markdowncell>

# ## Load Best Weights

# <markdowncell>

# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

# <markdowncell>

# ## Monitor the results

# <markdowncell>

# Check that we do not overfit by plotting the losses of the train and validation sets

# <codecell>

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();

# <markdowncell>

# Look at our overall fbeta_score

# <codecell>

fbeta_score

# <markdowncell>

# ## Make predictions

# <codecell>

predictions, x_test_filename = classifier.predict(batch_size)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# <codecell>

# For now we'll just put all thresholds to 0.2 but we need to calculate the real ones in the future
#thresholds = [0.2] * len(labels_set)


# <codecell>

thresholds = [0.24, 0.2, 0.2, 0.2, 0.2, 0.14, 0.05, 0.2, 0.2, 0.25, 0.25, 0.24, 0.2, 0.25, 0.2, 0.2, 0.25]

# <markdowncell>

# Now lets map our predictions to their tags by using the thresholds

# <codecell>

predicted_labels = classifier.map_predictions(predictions, thresholds)

# <markdowncell>

# Finally lets assemble and visualize our predictions for the test dataset

# <codecell>

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# <codecell>

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("Predictions rows:", final_df.size)
final_df.head()

# <codecell>

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

# <markdowncell>

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# <markdowncell>

# And save it to a submission file

# <codecell>

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()

# <markdowncell>

# #### That's it, we're done!
