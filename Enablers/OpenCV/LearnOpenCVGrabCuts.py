# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:42:17 2017

@author: echtpar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
img = cv2.imread('vodafone.png')
print(img.shape)

mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)


bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

print(fgdModel.shape)

print(type(fgdModel))


rect = (161, 79, 150, 150)

print(rect)
print(type(rect))


imgGrabCut = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)



print(type(imgGrabCut))
print(len(imgGrabCut))

plt.imshow(fgdModel)
plt.colorbar()
plt.show()

#cv2.imshow('imgGrabCut', imgGrabCut)



mask2 = np.where((mask ==2) | (mask ==0), 0,1).astype('uint8')

#cv2.imshow('imgshape', img.shape[:2])

#cv2.imshow('mask', mask)
#cv2.imshow('mask2', mask2)

img = img*mask2[:, :, np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



