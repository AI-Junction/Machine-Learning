from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#%%
#  data

path1 = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\yaleface\\image'    #path of folder of images    
path2 = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\yaleface\\image_resized'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print(num_samples)

for file in listing:
    Y=r'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\yaleface\\image'
    im = Image.open(Y + '\\' + file)   
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here           
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)
xyz = r'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\yaleface\\image_resized'
im1 = array(Image.open(xyz + '\\'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(xyz +'\\'+ im2)).flatten()
              for im2 in imlist],'f')
                
label=np.ones((num_samples,),dtype = int)
label[0:10]=0
label[11:21]=1
label[22:33]=2
label[34:44]=3
label[45:55]=4
label[56:66]=5
label[67:77]=6
label[78:88]=7
label[89:99]=8
label[100:110]=9
label[111:121]=10
label[122:132]=11
label[133:143]=12
label[144:153]=13
label[154:163]=14


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[12].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#%%

#batch_size to train
batch_size = 40
# number of output classes
nb_classes = 15
# number of epochs to train
nb_epoch = 10

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
#In image processing, a kernel, convolution matrix, or mask is a small matrix useful for blurring, sharpening, embossing, edge detection, and more.
# This is accomplished by means of convolution between a kernel and an image.
#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 90
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

#%%

model = Sequential()

#model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
convout2 = Activation('relu')
model.add(convout2)
##ADD
nb_filters = 15
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
convout3 = Activation('relu')
model.add(convout3)
##
#model.add(MaxPooling2D(pool_size=(2, 2), border_mode = 'valid'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#%%
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
            
            
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_split=0.2)

#%%
#i = 11
#plt.imshow(X_test[i, 0], interpolation='nearest')
#print("label : ", Y_test[i,:])
# visualizing losses and accuracy

print(hist.history['loss'])
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
#train_acc=hist.history['acc']
#val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#plt.figure(2,figsize=(7,5))
#plt.plot(xc,train_acc)
#plt.plot(xc,val_acc)
#plt.xlabel('num of Epochs')
#plt.ylabel('accuracy')
#plt.title('train_acc vs val_acc')
#plt.grid(True)
#plt.legend(['train','val'],loc=4)
##print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])




#%%       
#Computes the loss on some input data, batch by batch.
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score)
print('Test accuracy:', score)
#Generates output predictions for the input samples, processing the samples in a batched way.
print(model.predict_classes(X_test[1:15]))
print(Y_test[1:15])

i = 32
plt.imshow(X_test[i, 0], interpolation='nearest')
print("label : ", Y_test[i,:])



#%%

# visualizing intermediate layers

output_layer = model.layers[1].get_output()
output_fn = theano.function([model.layers[0].get_input()], output_layer)

# the input image

input_image=X_train[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:],cmap ='gray')
#plt.imshow(input_image[0,0,:,:])


output_image = output_fn(input_image)
print(output_image.shape)

# Rearrange dimension so we can plot the result 
output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
print(output_image.shape)


fig=plt.figure(figsize=(8,8))
for i in range(32):
    ax = fig.add_subplot(6, 6, i+1)
    #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)        
#               (or)
#
#y_pred = model.predict_classes(X_test)
#print(y_pred)

p=model.predict_proba(X_test) # to predict probability
print(p)
target_names = ['class 0(A)', 'class 1(K)', 'class 2(S)','class 3','class 4', 'class 5', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10','class 11', 'class 12', 'class 13','class 14']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))


i = 22
plt.imshow(X_test[i, 0], interpolation='nearest')

print("label : ", Y_pred[i,:])

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)



# Loading weights

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)


# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

