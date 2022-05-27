import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2

file = "img4.jpg"
filename = file[:-4]
chroma_reduct = "4:2:2"
table_name = "Tablica kwantyzująca"

y = 570
x = 350
y_end = y + 256
x_end = x + 256

class jpg:
    pass

def dct2(a):
    return scipy.fftpack.dct(
        scipy.fftpack.dct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def idct2(a):
    return scipy.fftpack.idct(
        scipy.fftpack.idct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def img_reconstuct(data, A):
    temp = np.zeros(A)
    counter = 0
    block = 8

    for i in range(0, A[0], block):
        for j in range(0, int(A[1]), block):
            temp[i:i+block, j:j+block] = data[counter] + 128
            counter = counter+1

    return temp


def kwantyzacja(data, operation, layer):
    QY = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 36, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )

    QC = np.array(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]
    )


    ret_dat = np.zeros(data.shape)
    if layer == "Y":
        if operation == "mult":
            ret_dat = data * QY

        if operation == "div":
            ret_dat = np.round(data / QY).astype(int)
        
    if layer == "Cr":
        if operation == "mult":
            ret_dat = data * QC

        if operation == "div":
            ret_dat = np.round(data / QC).astype(int)
        

    if layer == "Cb":
        if operation == "mult":
            ret_dat = data * QC
            
        if operation == "div":
            ret_dat = np.round(data / QC).astype(int)
        

    return ret_dat




def zigzag(A):
    template = np.array([
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],])

    if len(A.shape) == 1:
        B = np.zeros((8, 8))
        for r in range(0, 8):
            for c in range(0, 8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B


def eightCoder(A):
    tmp = []
    block = 8
    for i in range(0, A.shape[0], block):
        for j in range(0, A.shape[1], block):
            tmp.append(A[i : i + block, j : j + 8] - 128)
    tmp = np.array(tmp)
    return tmp


def chr_subsamp(img, J, a, b):
    ret_dat = []

    if J == 4 and a == 2 and b == 2:
        ret_dat = img[::, 0::a]
    if J == 4 and a == 4 and b == 4:
        ret_dat = img[::, ::]

    return ret_dat


def chr_resamp(img, J, a, b, shape):
    tmp = []

    if J == 4 and a == 2 and b == 2:
        tmp = np.zeros((shape[0], shape[1]))
        tmp[::, 0::a] = img[::, ::]
        tmp[::, 1::a] = tmp[::, 0::a]
        
    if J == 4 and a == 4 and b == 4:
        tmp = img[::, ::]

    return tmp

def compress(img, chroma_reduct):
    J, a, b = list(map(int, chroma_reduct.split(":")))
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    Y = YCrCb[:, :, 0]
    Cr = YCrCb[:, :, 1]
    Cb = YCrCb[:, :, 2]

    darr_Y = []
    darr_Cr = []
    darrCb = []

    Crsubsampl = chr_subsamp(Cr, J, a, b)
    Cbsubsampl = chr_subsamp(Cb, J, a, b)

    Cr = Crsubsampl
    Cb = Cbsubsampl

    Y = eightCoder(Y)
    Cr = eightCoder(Cr)
    Cb = eightCoder(Cb)

    
    for block in Y:
        darr_Y.append(dct2(block))
    darr_Y = np.array(darr_Y)
    Y = darr_Y.astype(int)

    for block in Cb:
        darrCb.append(dct2(block))
    darrCb = np.array(darrCb)
    Cb = darrCb.astype(int)
    
    for block in Cr:
        darr_Cr.append(dct2(block))
    darr_Cr = np.array(darr_Cr)
    Cr = darr_Cr.astype(int)

    #QUANTIZING
    quant_arr_Y = []
    quant_arr_Cr = []
    quant_arr_Cb = []

    for block in Y:
        quant_arr_Y.append(
            kwantyzacja(block, "div", "Y")
        )
    quant_arr_Y = np.array(quant_arr_Y)
    
    
    for block in Cb:
        quant_arr_Cb.append(
            kwantyzacja(block, "div", "Cb")
        )
    quant_arr_Cb = np.array(quant_arr_Cb)
    
    for block in Cr:
        quant_arr_Cr.append(
            kwantyzacja(block, "div", "Cr")
        )
    quant_arr_Cr = np.array(quant_arr_Cr)
    
    Y = quant_arr_Y
    Cr = quant_arr_Cr
    Cb = quant_arr_Cb

    #ZIGZAGGING
    zigY = []
    for block in Y:
        zigY.append(zigzag(block))
    zigY = np.array(zigY)
    
    zigCb = []
    for block in Cb:
        zigCb.append(zigzag(block))
    zigCb = np.array(zigCb)

    zigCr = []
    for block in Cr:
        zigCr.append(zigzag(block))
    zigCr = np.array(zigCr)

    Y = zigY
    Cr = zigCr
    Cb = zigCb

    return Y, Cr, Cb, Crsubsampl.shape


def decompress(Y,Cr,Cb,A,shape,chroma_reduct,):
    J, a, b = list(map(int, chroma_reduct.split(":")))

    blockY = []
    tmp = []

    for block in Y:
        tmp = zigzag(block)
        blockY.append(tmp.reshape(8,8))
    blockY = np.array(blockY)
    Y = blockY
    tmp = 0

    blockCr = []
    for block in Cr:
        tmp = zigzag(block)
        blockCr.append(tmp.reshape(8,8))
    blockCr = np.array(blockCr)
    Cr = blockCr
    tmp = 0

    blockCb = []
    for block in Cb:
        tmp = zigzag(block)
        blockCb.append(tmp.reshape(8,8))
    blockCb = np.array(blockCb)
    Cb = blockCb
    tmp = 0



    qarr_Y = []
    for block in Y:
        qarr_Y.append(
            kwantyzacja(block, "mult", "Y")
        )
    qarr_Y = np.array(qarr_Y)
    Y = qarr_Y

    qarr_Cr = []
    for block in Cr:
        qarr_Cr.append(
            kwantyzacja(block, "mult", "Cr")
        )
    qarr_Cr = np.array(qarr_Cr)
    Cr = qarr_Cr

    qarrCb = []
    for block in Cb:
        qarrCb.append(
            kwantyzacja(block, "mult", "Cb")
        )
    qarrCb = np.array(qarrCb)
    Cb = qarrCb




    arr_idct2_Y = []
    for block in qarr_Y:
        arr_idct2_Y.append(idct2(block))
    arr_idct2_Y = np.array(arr_idct2_Y)
    Y = arr_idct2_Y

    arr_idct2_Cr = []
    for block in qarr_Cr:
        arr_idct2_Cr.append(idct2(block))
    arr_idct2_Cr = np.array(arr_idct2_Cr)
    Cr = arr_idct2_Cr

    arr_idct2_Cb = []
    for block in qarrCb:
        arr_idct2_Cb.append(idct2(block))
    arr_idct2_Cb = np.array(arr_idct2_Cb)
    Cb = arr_idct2_Cb

    reconstructed_Y = img_reconstuct(Y, (A.shape[0], A.shape[1]))
    Y = reconstructed_Y


    reconstructed_Cr = img_reconstuct(Cr, shape)
    Cr = reconstructed_Cr


    reconstructed_Cb = img_reconstuct(Cb, shape)
    Cb = reconstructed_Cb

    resampled_Cr = chr_resamp(Cr, J, a, b, (A.shape[0], A.shape[1]))
    Cr = resampled_Cr

    resampled_Cb = chr_resamp(Cb, J, a, b, (A.shape[0], A.shape[1]))
    Cb = resampled_Cb

    Y = np.clip(Y, 0, 255)
    YCrCb = np.dstack([Y, Cr, Cb]).astype(np.uint8)
    RGB = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
    
    return RGB


combs = chroma_reduct.replace(':','')

# ORIGINAL
img = plt.imread(file)
plt.imshow(img)
plt.title("Original")
plt.show()

img = img[x:x_end, y:y_end]

Y_compressed, Cr_compressed, Cb_compressed, shape = compress(img, chroma_reduct)
decompressed = decompress(Y_compressed,Cr_compressed,Cb_compressed,img,shape,chroma_reduct)

plt.imshow(decompressed)
plt.show()

fig_title = file + " " + chroma_reduct

fig, axs = plt.subplots(4,2)
fig.suptitle(fig_title)
fig.set_size_inches(18.5, 10.5)

axs[0,0].imshow(img)

axs[0,1].imshow(decompressed[:, :])

axs[1,0].imshow(img[:, :, 0], cmap=plt.cm.gray)
axs[1,0].set_title("Y")

axs[1,1].imshow(decompressed[:, :, 0], cmap=plt.cm.gray)
axs[1,1].set_title("Y")

axs[2,0].imshow(img[:, :, 1], cmap=plt.cm.gray)
axs[2,0].set_title("Cr")

axs[2,1].imshow(decompressed[:, :, 1], cmap=plt.cm.gray)
axs[2,1].set_title("Cr")

axs[3,0].imshow(img[:, :, 2], cmap=plt.cm.gray)
axs[3,0].set_title("Cb")

axs[3,1].imshow(decompressed[:, :, 2], cmap=plt.cm.gray)
axs[3,1].set_title("Cb")
plt.savefig(
    f"{file[:-4]}_{chroma_reduct.replace(':','-')}.png"
)

plt.show()