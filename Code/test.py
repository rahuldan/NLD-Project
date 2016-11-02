"""
Created on Sat Oct 8 2016
@author: Rahul Dandwate
"""
"""
Edited by Sumukh on Sun 161023
This code extracts x coordinate values from a lorenz system and writes to file lorenzx.txt
I have commented out the rest of the code.
Hakuna matata!
"""

#import cv2
import numpy as np
#import matplotlib.pyplot as plt


Xub = 5.0
Xlb = -5.0
Xmax = 1.0                                                  #upper limit of bin
Xmin = 0                                                    #lower limit of bin
S = 256
epsilon = (Xmax - Xmin) / S
n = 0.7                                                     #gaussian random variable parameter
N0 = 10                                                     #minimum number of iteration
h = 1e-2                                                    #width of step
sigma = 10.0                                                #parameters
R = 28.0                                                    #of lorenz
beta = 2.667                                                #system
p = 1                                                       #modulo parameter
a0=[0.1,0.2,0.3]                                            #Initial condition ie the seed


iter = 100000                                                #for statistical analysis
#stat = np.zeros(S)

#img = cv2.imread("image file name")                         #Enter the name of the image file
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img_shape = np.shape(img)
#enc_img = np.zeros(shape = img_shape, dtype = np.uint16)
#dec_img = np.zeros(shape = img_shape, dtype = np.uint8)

def lorenz(X):                                               #X is a vector.
    return sigma * (X[1] - X[0]), (R * X[0]) - X[1] - (X[0] * X[2]), (X[0] * X[1]) - (beta * X[2]) #output is a vector too

#x,y,z=lorenz([.1, .1, .1])
#print "%f, %f, %f" % (x,y,z)

file= open('lorenzx.txt','w')



def RK4(X):                                                 #fourth order Runge-Kutta
    a = lorenz(X)
    b = lorenz(X + np.multiply((h / 2.0), a))
    c = lorenz(X + np.multiply((h / 2.0), b))
    d = lorenz(X + np.multiply(h, c))
    return X + np.multiply((h / 6), (a + np.multiply(2, b) + np.multiply(2, c) + d))


for i in range(0,1000000,1):
    if i==0:
        val=RK4(a0)
    else:
        val=RK4(val)
    file.write("%r \n" %(val[0]%1))
"""
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
                #k = np.random.normal(0.5, 0.5)              #generate a normal random variable
                k = 1
                if X[0] <= Xub and X[0] >= Xlb:
                    if count > N0 and X[0] % p >= map_point(bin_n) and X[0] % p < (map_point(bin_n) + epsilon) and k >= n:
                        enc_img[i][j] = count                   #using x for encryption
                        #print('enc', i, ' ', j)
                        break

def decryption():
    X = np.array([1, 1, 1])
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            for k in range(0, enc_img[i][j]):
                X = RK4(X)
            dec_img[i][j] = find_bin(X[0] % p)
            #print('dec', i, ' ', j)

def statistic():
    X = np.array([4.0, -1.0, -3.0])
    for i in range(0, iter):
        X = RK4(X)
        if X[0] >= Xlb and X[0] <= Xub:
            temp = X[0] % p
            bin_n = find_bin(temp)
            stat[bin_n] += 1

statistic()
a = np.arange(S)
print(stat)

plt.xlabel('Index of bin')
plt.ylabel('Frequency')
plt.scatter(a, stat, label = 'x = 4, y = -1, z = -3')
plt.legend(loc = 'upper right')
plt.show()

encryption()
decryption()
cv2.imshow('original', img)                                 #displays image
cv2.imshow('encrypted image', enc_img)
cv2.imshow('decrypted image', dec_img)
cv2.waitKey()
"""
