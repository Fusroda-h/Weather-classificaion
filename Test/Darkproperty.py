from cv2 import cv2
import numpy as np
import math

# Dark Channel property
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    
    return t

def Imgrefinedbydark(X,im_height,im_width):
    I_t,tr_t= np.zeros(X.shape),np.zeros((len(X),im_height,im_width))
    for i in range(len(X)):
        I_t[i] = X[i]
        dark_t = DarkChannel(I_t[i], 15)
        A_t = AtmLight(I_t[i], dark_t)
        te_t = TransmissionEstimate(I_t[i], A_t, 15)
        tr_t[i] = TransmissionRefine(X[i]*255, te_t)

    tr_t = np.repeat(tr_t,3).reshape(len(X),im_height,im_width,3)

    return X-0.75*tr_t

def Imgrefinedbydarkforhalf(X,im_height,im_width):
    I_t,tr_t= np.zeros(X.shape),np.zeros((len(X),im_height//2,im_width//2))
    for i in range(len(X)):
        I_t[i] = X[i]
        dark_t = DarkChannel(I_t[i], 15)
        A_t = AtmLight(I_t[i], dark_t)
        te_t = TransmissionEstimate(I_t[i], A_t, 15)
        tr_t[i] = TransmissionRefine(X[i]*255, te_t)

    tr_t = np.repeat(tr_t,3).reshape(len(X),im_height//2,im_width//2,3)
    
    return X-0.75*tr_t