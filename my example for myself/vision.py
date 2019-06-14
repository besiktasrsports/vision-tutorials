# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:15:06 2019

@author: Emre Candabakoglu for Sneaky Snakes 
"""

import cv2 # import opencv -1-
import numpy as np # import numpy -1-

def nothing(x):
    pass

cap = cv2.VideoCapture(0) # Define Camera Port 0 for Webcam

cv2.namedWindow('Colorbars') # Create trackbars named Colorbars

bh='Blue High'
bl='Blue Low'
gh='Green High'
gl='Green Low'
rh='Red High'
rl='Red Low'
wnd='Colorbars'

cv2.createTrackbar(bl, wnd, 0,   255, nothing) # Blue Low
cv2.createTrackbar(bh, wnd, 0,   255, nothing) # Blue High
cv2.createTrackbar(gl, wnd, 0,   255, nothing) # Green Low
cv2.createTrackbar(gh, wnd, 0,   255, nothing) # Green High
cv2.createTrackbar(rl, wnd, 0,   255, nothing) # Red Low
cv2.createTrackbar(rh, wnd, 0,   255, nothing) # Red High

while True: # -1-
    
    ret,frame = cap.read() # Read Camera
    #cv2.imshow("Video Capture", frame) # Show Camera
   
    lower_green = np.array([102,246,223]) # Minimum RGB or HSV values -2-
    upper_green = np.array([255,255,255]) # Maximum RGB or HSV values -2-
    
    test_image =  cv2.imread('images/middle_rocket.jpg') # Read image from folder direction -1-
    #test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2HSV) # RGB to HSV
    #cv2.imshow('Test',test_image) # Show test image -1-
    
    resizedImage = cv2.resize(test_image,(300,300)) # Resize image 300x300
    resizedImage = resizedImage[125:300, 0:300]
    cv2.imshow('Resized',resizedImage) # Show resized image
    
    staticColorImage = cv2.inRange(resizedImage,lower_green,upper_green) # inRange looks for objects between lower and upper green values
    
    #Ècv2.imshow('Static Color Image',staticColorImage) # Show static color image
    
    bLow  = cv2.getTrackbarPos(bl, wnd)
    bHigh = cv2.getTrackbarPos(bh, wnd)
    gLow  = cv2.getTrackbarPos(gl, wnd)
    gHigh = cv2.getTrackbarPos(gh, wnd)
    rLow  = cv2.getTrackbarPos(rl, wnd)
    rHigh = cv2.getTrackbarPos(rh, wnd)

    rgbLow=np.array([bLow,gLow,rLow]) # Define new lower color array depends on trackbar positions
    rgbHigh=np.array([bHigh,gHigh,rHigh]) # Define new upper color array depends on trackbar positions
    
    maskedImage = cv2.inRange(test_image, rgbLow, rgbHigh) # Look for values coming from trackbars
    
    cv2.imshow("Masked Image",maskedImage)
    
    kernel = np.ones((10,10),np.uint8)
    openedImage = cv2.morphologyEx(maskedImage, cv2.MORPH_OPEN, kernel) # Applying morphological operations
    kernel = np.ones((30,30),np.uint8)
    openedImage = cv2.morphologyEx(openedImage, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('opened',openedImage)
    
    contourImage, contours, hierarchy = cv2.findContours(staticColorImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE)
    contourIndexes =  []
    contourCounter = 0
    
    for contour in contours:
        contourIndexes.append(contourCounter)
        contourCounter += 1
        print(contourIndexes)
        
    cnt = contours[contourIndexes[-1]]
    cnt2 = contours[contourIndexes[-2]]
    cv2.drawContours(resizedImage,[cnt],0,(255,0,0),4)
    cv2.drawContours(resizedImage,[cnt2],0,(255,0,0),4)
    x,y,w,h = cv2.boundingRect(cnt) 
    x2,y2,w2,h2 = cv2.boundingRect(cnt2)
    
    cv2.imshow('Filled Image',resizedImage)
    
    
    
    
    
    
    keyPressed = cv2.waitKey(1)
    if keyPressed == 27:
        break
cv2.destroyAllWindows() #