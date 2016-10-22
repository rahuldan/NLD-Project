"""
Created on Wed Sep 28 2016

@author: Rahul Dandwate
"""
import cv2
import numpy as np

N = 50                                                  #size of one time pad
l = 1                                                   #lower limit of random integer
h = 255                                                 #upper limit of random integer
k = np.random.random_integers(l, high = h, size = N)    #random key vector
R = 3.9                                                 #parameter for logistic map

img = cv2.imread("image file name")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_shape = np.shape(img)
enc_img = np.zeros(shape = img_shape, dtype = np.uint8)
dec_img = np.zeros(shape = img_shape, dtype = np.uint8)

def logistic_map(r, x):
    return r*x*(1 - x)

def encryption():
    idx = 0
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            X = k[idx] / 256                            #maps int in (0, 256) to (0, 1)
            idx = (idx + 1) % N
            t = 16 + k[idx]
            for a in range(0, t):
                X = logistic_map(R, X)
            enc_img[i][j] = img[i][j] + int(X * 256)

def decryption():
    idx = 0
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            X = k[idx] / 256                            #maps int in (0, 256) to (0, 1)
            idx = (idx + 1) % N
            t = 16 + k[idx]
            for a in range(0, t):
                X = logistic_map(R, X)
            dec_img[i][j] = enc_img[i][j] - int(X * 256)

encryption()
decryption()

cv2.imshow('original', img)                             #displays image
cv2.imshow('encrypted image', enc_img)
cv2.imshow('decrypted image', dec_img)
cv2.waitKey()
