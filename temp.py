# -*- coding: utf-8 -*-

import numpy as np
import cv2 

def main():
    
    imgpath = "C:\\Users\\lucas\\Documents\\Fac 2018-2019\\Projet Si S2\\Images\\murescalade.jpg"
    img = cv2.imread(imgpath)
    
    cv2.imshow('Fenetre', img)
    cv2.waitKey(0)
    cv2.destroyWindow('Fenetre')

if __name__ == "__main__":
    main()
