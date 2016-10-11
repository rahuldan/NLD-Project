"""
Created on Sat Oct 8 2016

@author: Rahul Dandwate
"""
import cv2
import numpy as np

Xmin = 0.0                                                  #lower limit of bin
Xmax = 1.0                                                  #upper limit of bin
S = 256
epsilon = (Xmax - Xmin) / S
n = 0.7                                                     #gaussian random variable parameter
N0 = 10                                                     #minimum number of iteration
h = 1e-2                                                    #width of step
sigma = 10.0                                                #parameters
R = 28.0                                                    #of lorenz
beta = 2.667                                                #system
p = 1                                                       #modulo parameter

img = cv2.imread("image file name")                         #Enter the name of the image file
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_shape = np.shape(img)
enc_img = np.zeros(shape = img_shape, dtype = np.uint16)
dec_img = np.zeros(shape = img_shape, dtype = np.uint8)

def lorenz(X):
    return sigma * (X[1] - X[0]), (R * X[0]) - X[1] - (X[0] * X[2]), (X[0] * X[1]) - (beta * X[2])

def RK4(X):                                                 #fourth order Runge-Kutta
    a = lorenz(X)
    b = lorenz(X + np.multiply((h / 2.0), a))
    c = lorenz(X + np.multiply((h / 2.0), b))
    d = lorenz(X + np.multiply(h, c))
    return X + np.multiply((h / 6), (a + np.multiply(2, b) + np.multiply(2, c) + d))

def find_bin(X):
    return int((X - Xmin) / epsilon)

def map_point(X):
    return (X * epsilon) + Xmin

def encryption():
    X = np.array([1, 1, 1])
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            bin_n = img[i][j]
            count = 0
            while True:
                X = RK4(X)
                count += 1
                k = np.random.normal(0.5, 0.5)              #generate a normal random variable
                if count > N0 and X[0] % p >= map_point(bin_n) and X[0] % p < (map_point(bin_n) + epsilon) and k >= n:
                    enc_img[i][j] = count                   #using x for encryption
                    print('enc', i, ' ', j)
                    break

def decryption():
    X = np.array([1, 1, 1])
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            for k in range(0, enc_img[i][j]):
                X = RK4(X)
            dec_img[i][j] = find_bin(X[0] % p)
            print('dec', i, ' ', j)
encryption()
decryption()
cv2.imshow('original', img)                                 #displays image
cv2.imshow('encrypted image', enc_img)
cv2.imshow('decrypted image', dec_img)
cv2.waitKey()