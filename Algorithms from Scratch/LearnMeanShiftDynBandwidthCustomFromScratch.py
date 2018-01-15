# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 01:06:17 2017

@author: echtpar
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

centers = random.randrange(2,5)

X, y = make_blobs(n_samples = 5, centers = 3, n_features = 2)

print ('X: ') 
print (X)
print ('y: ') 
print (y)

#X = np.array([[1, 2],
#             [1.5, 1.8],
#             [5, 8],
#             [8, 8],
#             [1, 0.6],
#             [9, 11],
#             [8,2],
#             [10,2],
#             [9,3]])





#plt.scatter(X[:,0], X[:,1], s=150)
#print (X[:,1])
#print (X.shape)

#plt.show

colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self, radius = None, radius_norm_step = 10):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        print ('radius norm step: ') 
        print (self.radius_norm_step)
        
    def fit(self, data):
        
        if self.radius == None:
            all_data_centroid = np.average(data, axis = 0)
            print ('all_data_centroid: ')
            print (all_data_centroid)
            all_data_norm = np.linalg.norm(all_data_centroid)
            print ('all_data_norm: ')
            print (all_data_norm)
            self.radius = all_data_norm/self.radius_norm_step
            print ('self.radius: ')
            print (self.radius)
            
        
        
        
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]
            print ('centroids[i]: ')            
            print (centroids[i])


        while True:
            new_centroids = []
            for i in centroids:
                
                weights = [i for i in range(self.radius_norm_step)][::-1]
                print ('weights: ')                
                print (weights)                


                
                in_bandwidth = []
                centroid = centroids[i]
                
                for featureset in data:
                    
#                    print ('featureset in data: ')
#                    print (featureset)
                    
                    distance = np.linalg.norm(featureset - centroid)
                    
#                    print ('distance: ')
#                    print (distance)
                    
                    if distance == 0:
                        distance = 0.000000001
                    
                    weight_index = int(distance/self.radius)
                    print ('distance: ')
                    print (distance)
                    print ('radius: ')
                    print (self.radius)
                    print ('weight_index: ')                
                    print (weight_index)
                    
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                        
                    to_add = (weights[weight_index]**2)*[featureset]
#                    print ('weight_index: ')
#                    print (weight_index)
                    print ('weights[weight_index]**2): ')
                    print (weights[weight_index]**2)
                              
                    in_bandwidth += to_add
#                    print ('in_bandwidth')
#                    print (in_bandwidth)
                    
                              
                    
                        
                    
                    
#                    if np.linalg.norm(featureset-centroid) < self.radius:
#                        in_bandwidth.append(featureset)
                        

                print ('i th iteration: ')
                print (i)
                print ('to_add: ')
                print (to_add)
                print ('in_bandwidth')
                print (in_bandwidth)


                new_centroid = np.average(in_bandwidth, axis = 0)
                print ('new_centroid')
                print (new_centroid)
                
                new_centroids.append(tuple(new_centroid))
                print ('new_centroids: ')
                print (new_centroids)
                
                
            uniques = sorted(list(set(new_centroids)))
            print ('uniques: ')
            print (uniques)
            
            
            to_pop = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
#                        print ('to_pop: ')
#                        print (to_pop)
                        
                        break
            
            for i in to_pop:
                try:
                    uniques.remove[i]
                except:
                    pass
                
            print ('to_pop: ')
            print (to_pop)
                

            
            
            
            prev_centroids = dict(centroids)
            
            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
                
            optimized = True
            
            for i in centroids:
                if not np.array_equal(prev_centroids[i], centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break
        
        self.centroids = centroids
        

        self.classifications = {}
        
        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            print ('distances: ')
            print (distances)
            
            classification = distances.index(min(distances))
            print ('classification: ')
            print (classification)
            
            
            self.classifications[classification].append(featureset)
            print ('classifications[classification]')            
            print (self.classifications[classification])
        
    def predict(self, data):
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = Mean_Shift()

clf.fit(X)

centroids = clf.centroids

#plt.scatter(X[:,0], X[:,1], s = 150)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker = 'x', color = color, s = 150, linewidths = 3)
        
        
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color = 'g', marker = '*', s = 150, linewidths = 3)

plt.show()        