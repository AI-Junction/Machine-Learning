# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:52:08 2017

@author: Chandrakant Pattekar
"""

import pandas as pd

simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

vect.fit(simple_train)
vect.get_feature_names()

simple_train_dtm = vect.transform(simple_train)

print(simple_train_dtm)

simple_train_dtm.toarray()

print(simple_train_dtm)


pd.DataFrame(simple_train_dtm.toarray(), columns = vect.get_feature_names())

type(pd.DataFrame())

simple_test = ["please don't call me"]

simple_test_dtm = vect.transform(simple_test)

simple_test_dtm.toarray()

pd.DataFrame(simple_test_dtm.toarray(), columns = vect.get_feature_names())

# https://github.com/justmarkham/pydata-dc-2016-tutorial/blob/master/sms.tsv

path = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'

sms = pd.read_table(path, header = None, names = ['label', 'message'])

sms.shape
type(sms)


sms.head(10)


sms.label.value_counts()

sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

sms.shape

sms.head(10)


X = sms.message
y = sms.label_num

print(X.shape)
print(y.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


vect = CountVectorizer()

vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

print(X_train_dtm)

X_train_dtm = vect.fit_transform(X_train)

print(X_train_dtm)

X_test_dtm = vect.transform(X_test)
X_test_dtm

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)

print(y_pred_class[1])
print(X_test_dtm[1])
print(y_test)
print(set(X_test_dtm[0]))




from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)

metrics.confusion_matrix(y_test, y_pred_class)
