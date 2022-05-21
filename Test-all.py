import cv2  
  
import numpy as np

lien = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Images\\IMG_6243.jpg"
img = cv2.imread(lien)


rholist = []
thlist = []
ilist = []
listerho = []
listxt = []
tlist = []
xlist = []
listl = []
listx1 = []
listx11 = []
listx2 = []
lmask = []
test = True
cpt = 0
#b,g,r = cv2.split (img)

#r = img.copy()
## set blue and green channels to 0
#r[:, :, 0] = 0
#r[:, :, 1] = 0


##BLEU##
r = img.copy()
# set green and red channels to 0
r[:, :, 1] = 0
r[:, :, 2] = 0

##VERT##
#r = img.copy()
## set blue and red channels to 0
#r[:, :, 0] = 0
#r[:, :, 2] = 0

ret, thresh = cv2.threshold(r,127,255,cv2.THRESH_TRUNC)

img2 = thresh.copy()

edges = cv2.Canny(img2,50,150, apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for i in range(len(lines)):
    for rho,theta in lines[i]:
        thlist.append(theta)
        rholist.append(rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))
        
        x3 = int(x0 + 4000*(-b)+100)
        y3 = int(y0 + 4000*(a)+100)
        x4 = int(x0 - 4000*(-b)+100)
        y4 = int(y0 - 4000*(a)+100)
        
        x5 = int(x0 + 4000*(-b)-100)
        y5 = int(y0 + 4000*(a)-100)
        x6 = int(x0 - 4000*(-b)-100)
        y6 = int(y0 - 4000*(a)-100)
        
        listx11.append((x1,y1,x2,y2))
        
        test = True
        testx = True
        cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2) 

        if i == 0:
            print(i,' x1= ', x1,' y1= ',y1,' x2= ',x2,' y2= ',y2,'theta = ',theta)
            tlist.append(theta)
            listerho.append(rho)
            cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2) 
            listx1.append((x1,y1,x2,y2))
            listx2.append((x2,y2))
        else :
        
            for j in tlist:
                
                if theta in tlist or (theta >= j-0.05 and theta <= j+0.05):
                    test= False
                    
            for k in xlist:
                
                if (x1 >= k+10 and x1 <= k-10):
                    testx = False
                    
             ###TEST Pour Angle###       
            if (theta not in tlist and test == True) and (x1 not in xlist and testx == True) : 
                xlist.append(x1)
                tlist.append(theta)
                ilist.append(i)
                listerho.append(rho)
                cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2) 
                cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2) 
                listx1.append((x1,y1,x2,y2))
                listx2.append((x2,y2))
        
for i in range(len(lines)):
    if i in ilist:
        theta = thlist[i]
        rho = rholist[i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = listx11[i][0]
        y1 = listx11[i][1]
        x2 = listx11[i][2]
        y2 = listx11[i][3]
        
        x3 = int(x0 + 4000*(-b)+100)
        y3 = int(y0 + 4000*(a)+100)
        x4 = int(x0 - 4000*(-b)+100)
        y4 = int(y0 - 4000*(a)+100)
        
        x5 = int(x0 + 4000*(-b)-100)
        y5 = int(y0 + 4000*(a)-100)
        x6 = int(x0 - 4000*(-b)-100)
        y6 = int(y0 - 4000*(a)-100)
        
        cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2) 
        print(i,' x1= ', x1,' y1= ',y1,' x2= ',x2,' y2= ',y2,'theta = ',theta,'rho = ',rho)
    
        ###TEST Pour Couleur###
        liste= []
        tst = int((x3+x4)/2)
        tst2 = int((y3+y4)/2)
        tst3 = int((x5+x6)/2)
        tst4 = int((y5+y6)/2)
        cv2.line(img,(tst,tst2),(tst3,tst4),(255,255,255),2)        
        cpt += 1
        test3 = True
    
        for s in range(0,3):
            if (img[tst,tst2][s]+30 >= img[tst3,tst4][s]) and (img[tst,tst2][s]-30 <= img[tst3,tst4][s]):
                print('ligne n°',cpt,' oui ',img[tst,tst2][s],'  ',img[tst3,tst4][s])
                liste.append(test3)
                test3-=1                    
            else:
                test3 = False
                print('ligne n°',cpt,' non ',img[tst,tst2][s],'  ',img[tst3,tst4][s])
                liste.append(test3)
                            
        if test3==False:
            listxt.append((x1,y1,x2,y2))

cv2.namedWindow('line',cv2.WINDOW_NORMAL)
clr = 130
for t in listxt:
    clr += 15
    print('couleur : ',clr)
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    pts = np.array([[0,0],[t[0],t[1]],[t[2],t[3]],[t[2],0]])
    _=cv2.drawContours(mask, np.int32([pts]),0, clr, -1)
    lmask.append(mask)
    
cv2.imshow('line',mask)

cv2.waitKey(0) 
cv2.destroyAllWindows() 
    
    
dts = lmask[0]
dst = dts
alpha = 0.5  
beta = (1.0 - alpha)
for t in range(1,len(lmask)):

    dst = cv2.addWeighted(dts, alpha, lmask[t], beta, 0.0)
    dts = dst    
  

dst = cv2.bitwise_and(img, img, mask=lmask[0])            
                
#masked_data = cv2.bitwise_and(img, img, mask=img2)
                
                
print("listl = ", listl)
print('tlist= ', tlist)
cv2.namedWindow('t',cv2.WINDOW_NORMAL)
cv2.imshow('t',img)
cv2.namedWindow('line',cv2.WINDOW_NORMAL)
cv2.imshow('line',dst)

masque = "C:\\Users\\Lucas\\Documents\\Informatique\\Python\\Projet mur d'escalade\\Images\\Img_Mask_6243.jpg"

cv2.imwrite(masque, dst)



cv2.waitKey(0)
cv2.destroyAllWindows() 