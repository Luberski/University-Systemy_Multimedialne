from copy import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

#read image
img = cv2.imread('lab09/img.jpg', 1)
#change from bgr to rgb
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

Q = np.array([
    [6, 1, 5, 9, 1, 2, 5, 3],
    [9, 7, 9, 2, 5, 5, 2, 4],
    [6, 0, 7, 7, 1, 0, 8, 4],
    [0, 8, 2, 4, 3, 6, 3, 6],
    [6, 6, 7, 8, 9, 5, 2, 3],
    [9, 9, 9, 9, 0, 7, 6, 7],
    [3, 3, 6, 0, 4, 9, 7, 1],
    [9, 0, 0, 5, 5, 3, 8, 5],
])
tmp = np.zeros((Q.shape[0], Q.shape[1]))

def chroma_subsampling(img, J, a, b):
    tmp = []

    if J == 4 and a == 2 and b == 2:
        tmp = img[::, 0::a]
    if J == 4 and a == 4 and b == 4:
        tmp = img[::, ::]
    if J == 4 and a == 4 and b == 0:
        tmp = img[0::2, ::]
    if J == 4 and a == 2 and b == 0:
        tmp = img[0::2, 0::2]
    if J == 4 and a == 1 and b == 1:
        tmp = img[::, 0::4]
    if J == 4 and a == 1 and b == 0:
        tmp = img[0::2, 0::4]
    
    return tmp


def chroma_resampling(img, J, a, b, shape):
    tmp = np.zeros((shape[0], shape[1]))

    if J == 4 and a == 2 and b == 2:
        tmp[::, 0::a] = img[::, ::]
        tmp[::, 1::a] = tmp[::, 0::a]
    if J == 4 and a == 4 and b == 4:
        tmp = img[::, ::]
    if J == 4 and a == 4 and b == 0:
        tmp[0::2, ::] = img[::, ::]
        tmp[1::2, ::] = img[::, ::]
    if J == 4 and a == 2 and b == 0:
        tmp[::2, ::2] = img[::, ::]
        tmp[::2, 1::2] = img[::, ::]
        tmp[1::2, ::] = tmp[::2, ::]
    if J == 4 and a == 1 and b == 1:
        tmp[::, ::4] = img[::, ::]
        tmp[::, 1::4] = img[::, ::]
        tmp[::, 2::4] = img[::, ::]
        tmp[::, 3::4] = img[::, ::]
    if J == 4 and a == 1 and b == 0:
        tmp[::2, 0::4] = img[::, ::]
        tmp[1::2, 0::4] = img[::, ::]
        tmp[::, 1::4] = tmp[::, 0::4]
        tmp[::, 2::4] = tmp[::, 0::4]
        tmp[::, 3::4] = tmp[::, 0::4]

    return tmp

def RLE_encoding(img, bits=8,  binary=True):
    if binary:
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    encoded = []
    shape=img.shape
    count = 0
    prev = None
    fimg = img.flatten()
    th=127
    for pixel in fimg:
        if binary:
            if pixel<th:
                pixel=0
            else:
                pixel=1
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                if count<(2**bits)-1:
                    count+=1
                else:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1
    encoded.append((count, prev))
    
    return np.array(encoded)


def RLE_decode(encoded, shape):
    decoded=[]
    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*r)
    dimg = np.array(decoded).reshape(shape)
    return dimg

# be = RLE_encoding(b, binary=False)
# ge = RLE_encoding(g, binary=False)
# re = RLE_encoding(r, binary=False)

# encd = RLE_encoding(be, binary=False)
# dcdbe = RLE_decode(be, shape=be.shape)
# encd = RLE_encoding(ge, binary=False)
# dcdge = RLE_decode(encd, shape=be.shape)
# encd = RLE_encoding(re, binary=False)
# dcdre = RLE_decode(encd, shape=be.shape)

#show img
# plt.figure(1)
# plt.imshow(dcdbe)
# plt.show()

# test_arr = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 1])

# encoded_test_arr = RLE_encoding(test_arr, binary=False)
# print(f"encoded_test_arr: {encoded_test_arr}")

# decoded_test_arr = RLE_decode(encoded_test_arr, shape=test_arr.shape)
# print(f"decoded_test_arr = {decoded_test_arr}")

Q_tmp = chroma_subsampling(Q, 4, 1,1)
print(f"Q_tmp:\n {Q_tmp}\n\n")
Q_tmp = chroma_resampling(Q_tmp, 4, 1,1, shape=(8, 8))
print(f"Q_tmp:\n {Q_tmp}\n\n")