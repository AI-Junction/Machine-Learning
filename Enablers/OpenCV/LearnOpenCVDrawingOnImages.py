# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 07:14:32 2017

@author: echtpar
"""

import numpy as np
import cv2


''' uncomment below code to learn how to write on image
img = cv2.imread('Penguins.jpg', cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (450,350), (255,255,255), 5)
cv2.rectangle(img, (100,100), (500,500), (0,255,0), 5)
cv2.circle(img, (400, 400), 100, (0,0,255), -1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#below section gives example of how to manipulate images at pixel level
img = cv2.imread('Penguins.jpg', cv2.IMREAD_COLOR)

print (img[5,5])

img[55,55] = [255,255,255]
px = img[55,55]
print(px) 


#region of image (roi)
roi = img[100:150, 100:150]
print(roi.shape)

penguin_face = img[200:500, 200:400]
img[100:400,500:700] = penguin_face

#img[0:500, 100:400] = [255,255,255]
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
