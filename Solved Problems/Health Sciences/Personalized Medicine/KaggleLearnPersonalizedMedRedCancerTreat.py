# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:05:43 2018

@author: echtpar
"""

import pandas as pd
import numpy as np

input_path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllPersonalizedMedicineData\\"

train_file = "training_variants"
test_file = "test_variants"
trainx_file = "training_text"
testx_file = "test_text"


'''
text_file = open(input_path + train_file, "r", encoding = "utf8")
lines = text_file.readlines()
print (lines[:10])
print(type(lines))
print (len(lines))
text_file.close()

text_file = open(input_path + trainx_file, "r", encoding = "utf8")
lines = text_file.read().splitlines()
lines = text_file.readlines()
print (lines[:10])
print(type(lines))
print (len(lines))
text_file.close()
'''

train = pd.read_csv(input_path + train_file)
test = pd.read_csv(input_path + test_file)
trainx = pd.read_csv(input_path + trainx_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv(input_path + testx_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

print(train.columns.values)
print(test.columns.values)
print(trainx.columns.values)
print(testx.columns.values)

'''
print(trainx['Text'][:10].apply(lambda x: str (x[:20])))
print(trainx.count())
print(len(trainx))
print(*trainx)
print(trainx.columns.values)
print(trainx.index)
print(type(trainx.index.values))
print(set(trainx.index.values))
print(np.random.randn(*trainx.shape)[:10])
'''

'''
print(type(train))
df_cols = train.dtypes.reset_index()
df_cols.columns = ['col', 'type']
print(df_cols.columns.values)
df_cols_grouped = df_cols.groupby(['type']).aggregate('count').reset_index()
print(df_cols_grouped)
print(train.columns.values)
print(df_cols)
print(len(set(train['Gene'])))
'''

'''
df_train_group = train.groupby(['Gene'])['ID'].aggregate('count').reset_index()
print(df_train_group)
z = train.loc[train['Gene'] == 'RET', ['Gene', 'ID']]
print(z)
w = train.iloc[:10, :]
print(w)
for rec in enumerate(np.array(w)):
#    print(len(rec))
#    print(type(rec))
    print(np.array(rec)[1][2])
#    print(list(rec))

'''
'''
df_train_filled = train.fillna('WOW')
df_trainx_filled = trainx.fillna('WOW')
print(df_trainx_filled.columns.values)
print(df_trainx_filled.loc[df_trainx_filled['Text'] == 'WOW', ['Text']][:100])
'''

#train_merged.plot.hist()

#train_merged.plot.hist()

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis = 1)
print(train.columns.values)


test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
print(pid[:10])


df_all = pd.concat((train, test), axis=0, ignore_index=True)

df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)


print(df_all.columns.values)
print(df_all['Variation_Share'][:10])

a = ['A','B','C','D','E','F','G']
b = a[:-1]
a = ['A','B','C','D','E','F','G']
b = a[:-1]



print(b)

df_all['Gene'].map(lambda x: x[:-1])
df_all['Gene']
print(df_all['test'][:10])

df_all['Gene'].apply(lambda x: x[:-1])



#commented for Kaggle Limits
for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].apply(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].apply(lambda x: str(x[i]) if len(x)>i else '')

