# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 05:44:58 2017

@author: echtpar
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

#   hsv = hue saturation value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([150,150,50])
    upper_red = np.array([180,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
#    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    
    kernel = np.ones((15,15), np.float32)/255
    smoothed = cv2.filter2D(res, -1, kernel)
    blur = cv2.GaussianBlur(res, (15,15),0)
    median = cv2.medianBlur(res,15)
    bilateral = cv2.bilateralFilter(res, 15,75, 75)
    
    
#    cv2.imshow('hsv', hsv)
    cv2.imshow('frame', frame)
#    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
#    cv2.imshow('smoothed',smoothed)
#    cv2.imshow('blur',blur)
#    cv2.imshow('median',median)
    cv2.imshow('bilateral',bilateral)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    