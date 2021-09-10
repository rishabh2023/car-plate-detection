import cv2
import os
import easyocr
import numpy as np
algo = 'haarcascade_russian_plate_number.xml'
load_algo = cv2.CascadeClassifier(algo)
img = cv2.imread('datasets/Cars111.png')
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plate = load_algo.detectMultiScale(grayimg,1.1,4)
for (x,y,w,h) in plate:
        a,b = (int(0.02*img.shape[0]),int(0.010*img.shape[1]))
        nplate = img[y+a:y+h-a,x+b:x+w-b,:]
        kernal = np.ones((1,1),np.uint8)
        nplate = cv2.dilate(nplate,kernal,iterations=1)
        nplate = cv2.erode(nplate,kernal,iterations=1)
        nplate_gray = cv2.cvtColor(nplate,cv2.COLOR_BGR2GRAY)
        (thresh,nplate) = cv2.threshold(nplate_gray,127,255,cv2.THRESH_BINARY)

reader = easyocr.Reader(['en'])
result = reader.readtext(nplate,detail = 0)
print(result)
cv2.imshow('Number Plate',nplate)
cv2.waitKey(0)