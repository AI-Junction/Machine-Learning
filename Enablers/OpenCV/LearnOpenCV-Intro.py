# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 00:39:40 2017

@author: echtpar
"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
#    out.write(frame)
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    k = ord('q')    
    if cv2.waitKey(1) & 0xFF == k:
        break


#print(cv2.waitKey(1))
#print(0xFF)
#print(cv2.waitKey(1)&0xFF==k)


print (k)
    
cap.release()
#out.release()
cv2.destroyAllWindows()

    