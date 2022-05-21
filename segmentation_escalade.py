# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:25:40 2019

@author: lucas
"""

import numpy as np
import cv2
#import matplotlib.pylot as plt

lien = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Images\\IMG_6243.jpg"
cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Fenetre', 1512, 2016)   
img = cv2.imread(lien)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
cv2.imshow('Fenetre', img)
cv2.waitKey(0)
cv2.destroyWindow('Fenetre')

cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
cv2.imshow('Fenetre', thresh)
cv2.waitKey(0)
cv2.destroyWindow('Fenetre')

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
cv2.imshow('Fenetre', sure_bg)
cv2.waitKey(0)
cv2.destroyWindow('Fenetre')

cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
cv2.imshow('Fenetre', dist_transform)
cv2.waitKey(0)
cv2.destroyWindow('Fenetre')

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0] 

cv2.namedWindow('Fenetre', cv2.WINDOW_NORMAL)
cv2.imshow('Fenetre', img)
cv2.waitKey(0)
cv2.destroyWindow('Fenetre')