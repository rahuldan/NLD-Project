"""
Created on Sat Oct 1 2016

@author: Rahul Dandwate
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

R = 4.0
Xmin = 0.2                                                  #lower limit of bin
Xmax = 0.8                                                  #upper limit of bin
S = 256
epsilon = (Xmax - Xmin) / S
n = 0.7
N0 = 250                                                    #minimum number of iteration
X0 = 0.2323232                                              #initial condition

iter = 60000                                                #for statistical analysis
stat = np.zeros(S)
num_val = 200
stat2 = np.zeros(num_val)

#img = cv2.imread("square.jpg")                              #Enter the name of the image file
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = np.ones(shape = (S, S), dtype = np.uint8)

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

def logistic_stat():
    X = 0.2323232                                                     #make it random
    for i in range(0, iter):
        X = logistic_map(X)
        bin_n = find_bin(X)
        if bin_n < 256 and bin_n >= 0:
            stat[bin_n] += 1

b = np.linspace(Xmin, Xmax, num = num_val)

def logistic_stat2():
    iter = 10000
    for i in range(0, num_val):
        stat2[i] = b[i]
        for j in range(0, iter):
            stat2[i] = logistic_map(stat2[i])

"""logistic_stat2()
plt.plot(b, stat2, label = 'R = 4.0')
plt.xlabel('Initial Condition')
plt.ylabel('Mapped Value after 200 Iterations')
plt.legend(loc = 'upper right')
plt.show()

logistic_stat()
a = np.arange(S)

plt.xlabel('Index of bin')
plt.ylabel('Frequency')
plt.scatter(a, stat, label = 'x = 0.2323232, R = 4.0')
plt.legend(loc = 'upper right')
plt.show()"""

encryption()
decryption()

plt.hist(np.ravel(enc_img))
plt.show()

cv2.imshow('original', img)                                     #displays image
cv2.imshow('encrypted image', enc_img)
cv2.imshow('decrypted image', dec_img)
cv2.waitKey()
