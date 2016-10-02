"""
Created on Sat Oct 1 2016

@author: Rahul Dandwate
"""
import numpy as np
import cv2

R = 3.8
Xmin = 0.2                                                  #lower limit of bin
Xmax = 0.8                                                  #upper limit of bin
S = 256
epsilon = (Xmax - Xmin) / S
n = 0.7
N0 = 250                                                    #minimum number of iteration
X0 = 0.2323232                                              #initial condition

img = cv2.imread("image file name")                         #Enter the name of the image file 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_shape = np.shape(img)
enc_img = np.zeros(shape = img_shape, dtype = np.uint16)
dec_img = np.zeros(shape = img_shape, dtype = np.uint8)

def logistic_map(X):
    return R*X*(1 - X)

def map_point(X):                                           #function mapping (0, S - 1) to (Xmin, Xmax)
    return (X * (Xmax - Xmin) / S) + Xmin

def find_bin(X):
    return int((X - Xmin) / epsilon)

def encryption():
    X1 = X0
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            bin_n = img[i][j]
            count = 0
            while True:
                X1 = logistic_map(X1)
                count += 1
                k = np.random.normal(0.5, 0.5)                 #generate a normal random variable
                if count > N0 and X1 >= map_point(bin_n) and X1 < (map_point(bin_n) + epsilon) and k >= n:
                    enc_img[i][j] = count
                    break

def decryption():
    X1 = X0
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            for k in range(0, enc_img[i][j]):
                X1 = logistic_map(X1)
            dec_img[i][j] = find_bin(X1)

encryption()
decryption()
cv2.imshow('original', img)                                     #displays image
cv2.imshow('encrypted image', enc_img)
cv2.imshow('decrypted image', dec_img)
cv2.waitKey()
