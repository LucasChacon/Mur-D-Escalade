# -*- coding: utf-8 -*-

import cv2
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import time

lien = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Images\\IMG_6248.jpg"
tampon = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Resultats\\tampon.jpg"

t=time.time()
img = cv2.imread(lien)
img=cv2.split(img)[0]
(retVal,img2)=cv2.threshold(img,100,255,cv2.THRESH_TRUNC)
cv2.imwrite(tampon,img2)
img2 = cv2.imread(tampon)


gray = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
edges = cv2.Canny(img2,50,150,apertureSize = 3) 
lines = cv2.HoughLines(edges,1,np.pi/180,200)
LIST=[]
for i in range (0,lines.shape[0]):
    L_bis=[lines[i][0][0],lines[i][0][1]]
    LIST.append(L_bis)

fct=[[],[]]
for rho,theta in LIST:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    p=(y2-y1)/(x2-x1+1)
    k=(y1-p*x1)
    
    fct[0].append(p)
    fct[1].append(k)
    x0,y0=0,int(k)
    xf,yf=x0,y0
    
    if p<0:
        xf,yf=-int(k/p),0
        cv2.line(img2,(x0,y0),(xf,yf),(0,0,255),2)
    elif p>=0:
        xf,yf=int((img2.shape[0]-k)/p),img2.shape[0]
        cv2.line(img2,(x0,y0),(xf,yf),(255,255,0),2)

print(fct)
#fct est la liste de listes qui contient les droites du type y=a*x+b : [[a_1  a_2 ...a_n][b_1  b_2 ... b_n]]
Segm1 = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Resultats\\Segm1.jpg"

cv2.imwrite(Segm1,img2)

segm=cv2.imread(Segm1)
segm=cv2.split(segm)[0]

dim=Image.open(Segm1)
larg,haut=dim.size


mask = np.zeros((haut,larg), np.uint8)

def decimal(vecteur):
    s=0
    for i in range (len(vecteur)):
        s=s+2**(len(vecteur)-1-i)*vecteur[i]
    return(s)


nmIdx = 1


vecteur=[0 for i in range(len(fct[0]))]
pixels=[]

for m in range(haut):
    pixels.append([])
    for q in range (larg):
        pixels[m].append([])
        
        
vect_max=2**(len(fct[0]))-1


image = np.zeros((haut,larg), np.uint8)
for p in range (0,len(fct[0])):
    print("p=",p)
    mask = np.zeros((haut,larg), np.uint8)
    for j in range (0,haut):
        for i in range (0,larg):
            if j>(fct[0][p]*i+fct[1][p]):
                mask[j][i]=255
                pixels[j][i].append(1)
            else:
                pixels[j][i].append(0)
    
    maskp = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Resultats\\maskp_"
    cv2.imwrite(maskp + str(nmIdx) + ".jpg",mask)
    #cv2.imshow("image_final",image)
    nmIdx+=1
    print(100*(p+1)/len(fct[0]),"%")
    

for i in range (0,larg):
        for j in range (0,haut):
            base10=decimal(pixels[j][i])
            image[j][i]=int(float((base10/vect_max))*255)
            
final = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Resultats\\image.jpg"
cv2.imwrite(final,image)
print("time :",time.time()-t, "secondes")


cv2.waitKey(0)
cv2.destroyAllWindows()