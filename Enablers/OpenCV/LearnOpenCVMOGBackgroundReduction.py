# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 07:34:23 2017

@author: echtpar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
    
    