#from __future__ import print_function
#from skimage.feature import peak_local_max
#from skimage.morphology import watershed
#from scipy import ndimage
#import argparse
#import imutils
import numpy as np
import cv2

img = cv2.imread('C:\\Users\\yodam\\Pictures\\Projet S2\\IMG_6260.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
edges = cv2.Canny(gray, 0, 150, apertureSize = 3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)
    

#cv2.imwrite('img_test_relief.jpg', img)
cv2.namedWindow('A',cv2.WINDOW_NORMAL)
cv2.imshow('A', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.namedWindow('A',cv2.WINDOW_NORMAL)
cv2.imshow('A', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


#D = ndimage.distance_transform_edt(thresh)
#localMax = peak_local_max(D, indices=False, min_distance=20,
#	labels=thresh)
# 
## perform a connected component analysis on the local peaks,
## using 8-connectivity, then appy the Watershed algorithm
#markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
#labels = watershed(-D, markers, mask=thresh)
#
#
#cv2.namedWindow('A',cv2.WINDOW_NORMAL)
#cv2.imshow('A', labels)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

cv2.namedWindow('A',cv2.WINDOW_NORMAL)
cv2.imshow('A', sure_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv2.namedWindow('A',cv2.WINDOW_NORMAL)
cv2.imshow('A', dist_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#cv2.namedWindow('A',cv2.WINDOW_NORMAL)
#cv2.imshow('A', unknown)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#cv2.imshow('A', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.namedWindow('A',cv2.WINDOW_NORMAL)
cv2.imshow('A', img)
cv2.waitKey(0)
cv2.destroyAllWindows()