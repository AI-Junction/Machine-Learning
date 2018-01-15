from math import sqrt
import warnings
from collections import Counter


import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np

import pandas as pd
import random



#dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]
#
#[[plt.scatter(ii[0], ii[1], color = i, s = 150) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], marker = 'x', s=150, linewidths = 5)
#plt.show()
#  

X = np.array([[1, 2],
             [1.5, 1.0],
             [5, 0],
             [0, 0],
             [1, 0.6],
             [9, 11]])

plt.scatter(X[:,0], X[:,1], s=150)
print (X[:,1])
print (X.shape)

plt.show

colors = 10*["g","r","c","b","k"]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than data length')
        
    distances = []
    
    for group in data:
        for featureset in data[group]:
            euclidean_distance = np.linalg.norm(np.array(featureset) - np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    
#    print (distances)      
#    print (sorted(distances)[:k])         
#    print (votes)
#    print (Counter(votes).most_common(1))         
    vote_result = Counter(votes).most_common(1)[0][0]
             
             
    return vote_result
        
    


# the column headers are given at the URL itself
url = 'https://raw.githubusercontent.com/nrkfeller/machinelearningnotes/master/breast-cancer-wisconsin.data.txt'
df = pd.read_csv(url)
df.replace('?', -99999, inplace = True)
df.drop(['id'],1,inplace = True)
full_data = df.astype(float).values.tolist()
print(full_data[:5])
print(len(full_data))
random.shuffle(full_data)

print(full_data[:5])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

                    
                      
for i in train_data:
#    print ('printing each record in train data - 1', i[:-1])
    train_set[i[-1]].append(i[:-1])

print (train_set[4])
    
for i in test_data:
#    print ('printing each record in test data - 1', i[:-1])
#    print ('printing group in test data - 1', i[-1])
    test_set[i[-1]].append(i[:-1])    
    
print (test_set[4])    
correct = 0
total = 0  
    
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=25)
        if group == vote:
            correct += 1
        total += 1  

print ('Accuracy: ', correct/total)
          
                      
#result = k_nearest_neighbors(train_set, data, k=3)
#print (result)
#[[plt.scatter(ii[0], ii[1], color = i, s = 150) for ii in data[i]] for i in data]
#plt.scatter(train_set[0], train_set[1], marker = 'x', s=150, linewidths = 5, color = result)
#plt.show()
                      


#print (df.head())






