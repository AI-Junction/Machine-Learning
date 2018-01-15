# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:14:53 2017

@author: Chandrakant Pattekar
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

url = 'https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls'
df = pd.read_excel(url)
print(df.head())
print(df.columns.values)


df.drop(['body', 'name'], 1 , inplace=True)
print(set(df['survived'].values.tolist()))

print(df.head())
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace = True)
print(df.head())



def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
#            print('building the directory - column name = ' +  column) 
#            print('val = ' + val) 
#            print('directory value =')
#            print(text_digit_vals[val])
            return text_digit_vals[val]
            
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            unique_elements = set(df[column].values.tolist())
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
#                    print('text_digit_vals Dictionary = ') 
#                    print(text_digit_vals)
                    x +=1
            
            df[column] = list(map(convert_to_int, df[column])) 
            
            
    
    return df
    
df = handle_non_numerical_data(df)
#print(df.head())

df.drop(['boat'],1, inplace = True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)


y = np.array(df['survived'])

clf = KMeans(n_clusters = 2)
clf.fit(X)

correct = 0

print (X[:10][:])
print (X[1][2])


for i in range(len(X)):
    predict_me = np.array(X[i]).astype(float)
    print('now preparing predict_me i = ' + str(i))
    print(predict_me)
    predict_me = predict_me.reshape(-1, len(predict_me))

    print('now reshaping predict_me i = ' + str(i))
    print(predict_me)

    prediction = clf.predict(predict_me)
    
    print('now check y[i] = ' )
    print(y[i])

    
    if prediction[0]==y[i]:
        correct += 1
        
print (correct/len(X))

        

             
             
