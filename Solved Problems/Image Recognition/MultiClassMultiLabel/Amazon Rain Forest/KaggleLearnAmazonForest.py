# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:16:26 2017

@author: Chandrakant Pattekar
"""
import numpy as np 
import pandas as pd 
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy

print(scipy.__version__)
print(python.__version__)



from subprocess import check_output
from scipy import ndimage
from IPython.display import display

import plotly.graph_objs as go
trace = go.Heatmap(z=np.array(heatmapdatanew),
                   x=amazon_condition_labels,
                   y=amazon_condition_labels, colorscale='Viridis')

data=[trace]
layout=go.Layout(height=600, width=600, title='heatmap')
fig=dict(data=data, layout=layout)

#py.iplot(data)
py.iplot(data, filename='pandas-heatmap')


print(check_output(["ls", "../input"]).decode("utf8"))
pal = sns.color_palette('husl')
#%matplotlib inline
import os
import sys
import gc


train_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\train.csv"

# Set root directory (src: the github repo)
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, './train.csv')


PLANET_KAGGLE_LABEL_CSV = train_path
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)





# unique labels
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
label_list = []

labels = labels_df['tags'].apply(lambda x: x.split(' '))
print(type(labels))
print(labels.head())

y = labels_df['tags'].str.get_dummies(sep=' ')
print(y.head())

df_test = pd.DataFrame()
df_test['split_tags'] = labels_df['tags'].map(lambda row: row.split(" "))
print(df_test.head())

in_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\"
labels_df['path'] = labels_df['image_name'].map(lambda x: in_path + 'train-jpg\\' + x + '.jpg')
print(labels_df['path'][:1].str.get_dummies(sep='\\'))

print(labels.head())


z = list(labels_df['tags'])

print(z[:10])

print(labels_df[:10])

z = labels_df['tags']

print(z[:10])

for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)
            
            
print(len(labels_df.tags.values))            
print(len(label_list))            


label1 = "my first_name is Chandrakant and my last_name is Pattekar"
label2 = "this is my address"
dfnew = pd.DataFrame()
label1s = label1.split(' ')
dfnew['label1'] = label1s
print(dfnew)
print(label1s)
print('is' in label1s)


q = label1.get_dummies(sep=' ')

y = dfnew['label1'].str.get_dummies(sep=' ')
y = pd.get_dummies(dfnew['label1']).astype(int)
y

dfnew2 = pd.DataFrame({'tag1': label1s})
dfnew2

listalllabels = []

[listalllabels.append(i) for x in list(dfnew['label1'])  for i in x]
 
[listalllabels.append(i) for x in label1s  for i in x]
 
[listalllabels.append(x) for x in label1s] 

print(listalllabels)



print([i for i in label1s])

 
dfnew['label1'] 

z = dfnew['label1'].apply(lambda x: x.split ('_'))

print(type(z))
print(z)
listalllabels = []
[listalllabels.append(y) for x in z for y in x]
print(listalllabels)

 
for x in z:
    for y in x:
        listalllabels.append(y)        
 
print(listalllabels)


s = list(z)

print(type(s))

d = 'Hello'
nestedlist = [[['a','b'],['c','d'],['e','f']],[['g','h'],['i','j'],['k','l']],[['m','n'],['o','p'],['q','r']]]
[w for x in nestedlist for i in x for q in i for w in q]




flatten = np.array(nestedlist).flatten()
print(len(flatten))
print(flatten)

nestedlist_reshape = flatten.reshape(-1,1,18)
print(nestedlist_reshape)

nestedlist_reshape = flatten.reshape(1,18)
print(nestedlist_reshape)

nestedlist_reshape = flatten.reshape(len(flatten))
print(nestedlist_reshape)


# Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

#print(labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0))    

print(labels_df.head())    
    
# Histogram of label instances
fig = plt.subplots(figsize=(8,4))
labels_df[label_list].sum().sort_values().plot.bar()
labels_df[label_list].sum().sort_values().plot.line()
plt.show()


all_tags = [item for sublist in list(labels_df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
print('total of {} non-unique tags in all training images'.format(len(all_tags)))
print('average number of labels per image {}'.format(1.0*len(all_tags)/labels_df.shape[0]))

print(len(all_tags))

tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
tags_counted_and_sorted.head()
print(tags_counted_and_sorted)

from scipy.stats import bernoulli

sample = pd.read_csv('C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllAmazonData\\sample_submission.csv\\sample_submission.csv')

tag_probas = tags_counted_and_sorted[0].values/tags_counted_and_sorted[0].values.sum()
print(tags_counted_and_sorted[0].values.sum())
print(tag_probas)

indicators = np.hstack([bernoulli.rvs(p, 0, sample.shape[0]).reshape(sample.shape[0], 1) for p in tag_probas])
print(indicators)
indicators = np.array(indicators)
indicators.shape
print(sample.shape[0])

sample['tags'] = all_test_tags
sample.head()
sample.to_csv('bernoulli_submission.csv', index=False)

x = np.arange(9.).reshape(3, 3)
y = np.arange(9)
print(x)

print(x)
print(np.where( x < 6 ))


print(np.array(sample.shape[0]).reshape(sample.shape[0], 1))
print(np.array(sample.shape[0]).reshape(sample.shape[0], 1))

z = pd.DataFrame({'tag': all_tags})

print(z)


sorted_tags = tags_counted_and_sorted['tag'].values

print(sorted_tags)

all_test_tags = []
for index in range(indicators.shape[0]):
    all_test_tags.append(' '.join(list(sorted_tags[np.where(indicators[index, :] == 1)[0]])))
len(all_test_tags)
print(all_test_tags[:50])

print(range(len(list(label1s))))

print(list(label1s))

all_test_tags = []
for index in range(label1s.shape[0]):
    label1s.append(' '.join(list(label1s[np.where(label1s[index, :] == 1)[0]])))
len(all_test_tags)
print(all_test_tags[:50])



print([item for sublist in list(dfnew['label1'].apply(lambda row: row.split(" ")).values) for item in sublist])


z = [[x] for sublist in list(dfnew['label1'].apply(lambda row: row.split(" ")).values) for x in sublist]
print(z)


for sublist in list(dfnew['label1'].apply(lambda row: row.split(" ")).values): 
    for x in sublist:
        print([x])





print(len(dfnew['label1'].apply(lambda row: row.split(" ")).values))


tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))

#%%

from multiprocessing import Pool, cpu_count
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
import xgboost as xgb
import pandas as pd
import numpy as np
import glob, cv2
import random

random.seed(1)
np.random.seed(1)

def get_features(path):
    img = cv2.imread(path)
    hist = cv2.calcHist([cv2.imread(path,0)],[0],None,[256],[0,256])
    m, s = cv2.meanStdDev(img)
    img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
    img = np.append(img.flatten(), m.flatten())
    img = np.append(img, s.flatten())
    img = np.append(img, hist.flatten()) #/ 255
    return [path, img]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    return fdata

in_path = '../input/'
train = pd.read_csv(in_path + 'train.csv')
train['path'] = train['image_name'].map(lambda x: in_path + 'train-jpg/' + x + '.jpg')
y = train['tags'].str.get_dummies(sep=' ')
xtrain = normalize_img(train['path']); print('train...')

test_jpg = glob.glob(in_path + 'test-jpg/*')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['image_name','path']
xtest = normalize_img(test['path']); print('test...')


#%%

# Model 1

etr = ExtraTreesRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=1)
etr.fit(xtrain, y); print('etr fit...')

train_pred = etr.predict(xtrain)
train_pred[train_pred > 0.20] = 1
train_pred[train_pred < 1] = 0
print(fbeta_score(y,train_pred,beta=2, average='samples'))

pred1 = etr.predict(xtest); print('etr predict...')
etr_test = pd.DataFrame(pred1, columns=y.columns)
etr_test['image_name'] =  test[['image_name']]


#%%

# Model 2

xgb_train = pd.DataFrame(train[['path']], columns=['path'])
xgb_test = pd.DataFrame(test[['image_name']], columns=['image_name'])
print('xgb fit...')
for c in y.columns:
    model = xgb.XGBClassifier(n_estimators=5, max_depth=4, seed=1)
    model.fit(xtrain, y[c])
    xgb_train[c] = model.predict_proba(xtrain)[:, 1]
    xgb_test[c] = model.predict_proba(xtest)[:, 1]
    print(c)

train_pred = xgb_train[y.columns].values
train_pred[train_pred >0.20] = 1
train_pred[train_pred < 1] = 0
print(fbeta_score(y,train_pred,beta=2, average='samples')) 
print('xgb predict...')


#%%

# Blend

xgb_test.columns = [x+'_' if x not in ['image_name'] else x for x in xgb_test.columns]
blend = pd.merge(etr_test, xgb_test, how='left', on='image_name')

for c in y.columns:
    blend[c] = (blend[c] * 0.45)  + (blend[c+'_'] * 0.55)

blend = blend[etr_test.columns]

#%%

# Prepare submission

tags = []
for r in blend[y.columns].values:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],y.columns[i]] for i in range(len(y.columns)) if r[i]>.20], reverse=True)]))

test['tags'] = tags
test[['image_name','tags']].to_csv('submission_blend.csv', index=False)
test.head()



import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Penguins.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


img = cv2.imread('Penguins.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

m, s = cv2.meanStdDev(img)
img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
plt.imshow(img)
img = np.append(img.flatten(), m.flatten())
plt.imshow(img)
img = np.append(img, s.flatten())
print(s)
print(img.shape)
print(m)
img = np.append(img, histr.flatten()) #/ 255
print(img.shape)
print(img)
plt.imshow(img)
plt.plot(histr,color = col)
plt.xlim([0,256])
plt.show()

from multiprocessing import Pool, cpu_count
path = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Penguins.jpg'
imf_d = {}
p = Pool(cpu_count())
ret = p.map(get_features, paths)
for i in range(len(ret)):
    imf_d[ret[i][0]] = ret[i][1]
ret = []
fdata = [imf_d[f] for f in paths]
fdata = np.array(fdata, dtype=np.uint8)
return fdata

import glob

q = glob.glob('./*.jpg')
print(q)


#%%

# from 1ow1

from multiprocessing import Pool, cpu_count
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
import pandas as pd
import numpy as np
import glob, cv2
from PIL import Image, ImageStat

path = 'C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\Penguins.jpg'
st = []
#pillow
im_stats_ = ImageStat.Stat(Image.open(path))
#print(im_stats_.sum)
st += im_stats_.sum
print(st)

st += im_stats_.mean
print(st)

st += im_stats_.rms
print(st)

st += im_stats_.var
print(st)

st += im_stats_.stddev
print(st)

#cv2
img = cv2.imread(path)
bw = cv2.imread(path,0)
st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten()) #histogram
print(st)

m, s = cv2.meanStdDev(img) #mean and standard deviation
st += list(m)
print(st)

st += list(s)
print(st)

st += cv2.Laplacian(bw, cv2.CV_64F).var() #blurr
print(st)

st += (bw<10).sum()
print(st)

st += (bw>245).sum()
print(st)

print(len(st))





def extract_features(path):
    try:
        st = []
        img = cv2.imread(path)
        bw = cv2.imread(path,0)
        im_stats_ = ImageStat.Stat(Image.open(path))
        
        st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten())
        
        #pillow
        
        #print(im_stats_.sum)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        #cv2
        
         #histogram
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        st += list(m)
        st += list(s)
        st += cv2.Laplacian(bw, cv2.CV_64F).var() #blurr
        st += (bw<10).sum()
        st += (bw>245).sum()
        #img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
    except:
        print(path)
    return [path, st]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    return fdata

in_path = '../input/'
train = pd.read_csv(in_path + 'train.csv')
train['path'] = train['image_name'].map(lambda x: in_path + 'train-jpg/' + x + '.jpg')
y = train['tags'].str.get_dummies(sep=' ')
xtrain = normalize_img(train['path']); print('train...')

test_jpg = glob.glob(in_path + 'test-jpg/*')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['image_name','path']
xtest = normalize_img(test['path']); print('test...')

etr = ExtraTreesRegressor(n_estimators=18, max_depth=12, n_jobs=-1, random_state=1)
etr.fit(xtrain, y); print('fit...')

train_pred = etr.predict(xtrain)
train_pred[train_pred >0.24] = 1
train_pred[train_pred < 1] = 0
print(fbeta_score(y,train_pred,beta=2, average='samples'))

pred = etr.predict(xtest); print('predict...')

tags = []
for r in pred:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],y.columns[i]] for i in range(len(y.columns)) if r[i]>.24], reverse=True)]))

test['tags'] = tags
test[['image_name','tags']].to_csv('submission_boc_01.csv', index=False)
test.head()

#%%

#mutualy exclusive tags from cooccurence_matrix in following script
#https://github.com/planetlabs/planet-amazon-deforestation/blob/master/planet_chip_examples.ipynb

def me_clean(row): #
    row = row.split(' ')
    d = {k: i for i, k in enumerate(row)}
    me_list = [['artisinal_mine','conventional_mine','blow_down'],['clear','partly_cloudy','cloudy','haze']]
    for l in me_list:
        l2 = [j for j in l if j in row]
        if len(l2)>1:
            l2 = [c[0] for c in sorted([[c, d[c]] for c in l2], reverse=True)] #give priority to lower bound pred
            row = [j for j in row if j not in l2[1:]]
    return ' '.join(row)

test['tags'] = test['tags'].apply(lambda x: me_clean(x))
test[['image_name','tags']].to_csv('submission_boc_02.csv', index=False)
test.head()

#%%

import matplotlib.pyplot as plt
%matplotlib inline

th = []
train_predx = etr.predict(xtrain)
for i in np.arange(0.0, 0.9, 0.01):
    train_pred = train_predx.copy()
    train_pred[train_pred >i] = 1
    train_pred[train_pred < 1] = 0
    th.append([i, fbeta_score(y,train_pred,beta=2, average='samples')])
_ = pd.DataFrame(th, columns=['th','f2_score']).plot(kind='line', x='th', y='f2_score')


from tqdm import *
import time

for i in tqdm(range(1000)):
    time.sleep(0.05)

z = [time.sleep(0.05) for i in tqdm(range(200))]
    
#%%

# code of Robert J. Regalado - Notebook: Script_1234


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

import scipy
from sklearn.metrics import fbeta_score

from PIL import Image

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')

def extract_features(df, data_path):
    im_features = df.copy()

    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.image_name.values, miniters=500): 
        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]

        r_mean.append(np.mean(im[:,:,0].ravel()))
        g_mean.append(np.mean(im[:,:,1].ravel()))
        b_mean.append(np.mean(im[:,:,2].ravel()))

        r_std.append(np.std(im[:,:,0].ravel()))
        g_std.append(np.std(im[:,:,1].ravel()))
        b_std.append(np.std(im[:,:,2].ravel()))

        r_max.append(np.max(im[:,:,0].ravel()))
        g_max.append(np.max(im[:,:,1].ravel()))
        b_max.append(np.max(im[:,:,2].ravel()))

        r_min.append(np.min(im[:,:,0].ravel()))
        g_min.append(np.min(im[:,:,1].ravel()))
        b_min.append(np.min(im[:,:,2].ravel()))

        r_kurtosis.append(scipy.stats.kurtosis(im[:,:,0].ravel()))
        g_kurtosis.append(scipy.stats.kurtosis(im[:,:,1].ravel()))
        b_kurtosis.append(scipy.stats.kurtosis(im[:,:,2].ravel()))
        
        r_skewness.append(scipy.stats.skew(im[:,:,0].ravel()))
        g_skewness.append(scipy.stats.skew(im[:,:,1].ravel()))
        b_skewness.append(scipy.stats.skew(im[:,:,2].ravel()))


    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return im_features

# Extract features
print('Extracting train features')
train_features = extract_features(train, train_path)
print('Extracting test features')
test_features = extract_features(test, test_path)

# Prepare data
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, miniters=2000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.uint8)

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))

n_classes = y.shape[1]

X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))

# Train and predict with one-vs-all strategy
y_pred = np.zeros((X_test.shape[0], n_classes))

print('Training and making predictions')
for class_i in tqdm(range(n_classes), miniters=1): 
#     print('Analysing class ' + str(class_i))
    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.3, n_estimators=100, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = [' '.join(labels[y_pred_row > 0.2]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = test_features.image_name.values
subm['tags'] = preds
subm.to_csv('submission.csv', index=False)

#%%     
    
