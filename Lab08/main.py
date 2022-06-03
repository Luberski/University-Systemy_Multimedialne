import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2

def reconstructImage(data, A):
    tmp = np.zeros(A)
    cnt = 0
    block = 8
    for i in range(0, A[0], block):
        for j in range(0, int(A[1]), block):
            tmp[i : i + block, j : j + block] = data[cnt] + 128
            cnt = cnt + 1
    return tmp


def quantization(data, sign, layer, quant_or_ones):
    if quant_or_ones == "Tablica kwantyzująca":
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
    if quant_or_ones == "Tablica jedynek":
        QY = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        QC = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
    arr = np.zeros(data.shape)
    if layer == "Y":
        if sign == "divide":
            arr = np.round(data / QY).astype(int)
        if sign == "multiply":
            arr = data * QY
    if layer == "Cr":
        if sign == "divide":
            arr = np.round(data / QC).astype(int)
        if sign == "multiply":
            arr = data * QC
    if layer == "Cb":
        if sign == "divide":
            arr = np.round(data / QC).astype(int)
        if sign == "multiply":
            arr = data * QC

    return arr


def chromaSubsampling(img, J, a, b):
    tmp = []
    if J == 4 and a == 2 and b == 2:
        tmp = img[::, 0::a]
    if J == 4 and a == 4 and b == 4:
        tmp = img[::, ::]
    return tmp


def chromaResampling(img, J, a, b, shape):
    tmp = []
    if J == 4 and a == 2 and b == 2:
        tmp = np.zeros((shape[0], shape[1]))
        tmp[::, 0::a] = img[::, ::]
        tmp[::, 1::a] = tmp[::, 0::a]
    if J == 4 and a == 4 and b == 4:
        tmp = img[::, ::]
    return tmp


def zigzag(A):
    template = np.array(
        [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
    )
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


def dct2(a):
    return scipy.fftpack.dct(
        scipy.fftpack.dct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def idct2(a):
    return scipy.fftpack.idct(
        scipy.fftpack.idct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def eightCoder(A):
    tmp = []
    block = 8
    for i in range(0, A.shape[0], block):
        for j in range(0, A.shape[1], block):
            tmp.append(A[i : i + block, j : j + 8] - 128)
    tmp = np.array(tmp)
    return tmp


def reshapeToBlock(data):
    return data.reshape(8, 8)


def compress(img, chroma_subsampling_combination, quant_or_ones):
    J, a, b = list(map(int, chroma_subsampling_combination.split(":")))
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    Y = YCrCb[:, :, 0]
    Cr = YCrCb[:, :, 1]
    Cb = YCrCb[:, :, 2]

    Cr_subsampled = chromaSubsampling(Cr, J, a, b)
    Cb_subsampled = chromaSubsampling(Cb, J, a, b)
    Cr = Cr_subsampled
    Cb = Cb_subsampled
    Y = eightCoder(Y)
    Cr = eightCoder(Cr)
    Cb = eightCoder(Cb)

    # DCT2
    dct_arr_Y = []
    for block in Y:
        dct_arr_Y.append(dct2(block))
    dct_arr_Y = np.array(dct_arr_Y)
    Y = dct_arr_Y.astype(int)

    dct_arr_Cr = []
    for block in Cr:
        dct_arr_Cr.append(dct2(block))
    dct_arr_Cr = np.array(dct_arr_Cr)
    Cr = dct_arr_Cr.astype(int)

    dct_arr_Cb = []
    for block in Cb:
        dct_arr_Cb.append(dct2(block))
    dct_arr_Cb = np.array(dct_arr_Cb)
    Cb = dct_arr_Cb.astype(int)

    # Quantize
    q_arr_Y = []
    for block in Y:
        q_arr_Y.append(
            quantization(block, "divide", "Y", quant_or_ones)
        )
    q_arr_Y = np.array(q_arr_Y)
    Y = q_arr_Y

    q_arr_Cr = []
    for block in Cr:
        q_arr_Cr.append(
            quantization(block, "divide", "Cr", quant_or_ones)
        )
    q_arr_Cr = np.array(q_arr_Cr)
    Cr = q_arr_Cr

    q_arr_Cb = []
    for block in Cb:
        q_arr_Cb.append(
            quantization(block, "divide", "Cb", quant_or_ones)
        )
    q_arr_Cb = np.array(q_arr_Cb)
    Cb = q_arr_Cb

    # Zigzag
    zzY = []
    for block in Y:
        zzY.append(zigzag(block))
    zzY = np.array(zzY)
    Y = zzY

    zzCr = []
    for block in Cr:
        zzCr.append(zigzag(block))
    zzCr = np.array(zzCr)
    Cr = zzCr

    zzCb = []
    for block in Cb:
        zzCb.append(zigzag(block))
    zzCb = np.array(zzCb)
    Cb = zzCb

    return Y, Cr, Cb, Cr_subsampled.shape


def decompress(
    Y,
    Cr,
    Cb,
    A,
    shape,
    chroma_subsampling_combination,
    quant_or_ones,
):
    J, a, b = list(map(int, chroma_subsampling_combination.split(":")))
    # To blocks
    Y_to_blocks = []
    tmp = []
    for block in Y:
        tmp = zigzag(block)
        Y_to_blocks.append(reshapeToBlock(tmp))
    Y_to_blocks = np.array(Y_to_blocks)
    Y = Y_to_blocks
    tmp = 0

    Cr_to_blocks = []
    for block in Cr:
        tmp = zigzag(block)
        Cr_to_blocks.append(reshapeToBlock(tmp))
    Cr_to_blocks = np.array(Cr_to_blocks)
    Cr = Cr_to_blocks
    tmp = 0

    Cb_to_blocks = []
    for block in Cb:
        tmp = zigzag(block)
        Cb_to_blocks.append(reshapeToBlock(tmp))
    Cb_to_blocks = np.array(Cb_to_blocks)
    Cb = Cb_to_blocks
    tmp = 0

    # Quantize
    q_arr_Y = []
    for block in Y:
        q_arr_Y.append(
            quantization(block, "multiply", "Y", quant_or_ones)
        )
    q_arr_Y = np.array(q_arr_Y)
    Y = q_arr_Y

    q_arr_Cr = []
    for block in Cr:
        q_arr_Cr.append(
            quantization(block, "multiply", "Cr", quant_or_ones)
        )
    q_arr_Cr = np.array(q_arr_Cr)
    Cr = q_arr_Cr

    q_arr_Cb = []
    for block in Cb:
        q_arr_Cb.append(
            quantization(block, "multiply", "Cb", quant_or_ones)
        )
    q_arr_Cb = np.array(q_arr_Cb)
    Cb = q_arr_Cb

    # IDCT2
    arr_idct2_Y = []
    for block in q_arr_Y:
        arr_idct2_Y.append(idct2(block))
    arr_idct2_Y = np.array(arr_idct2_Y)
    Y = arr_idct2_Y

    arr_idct2_Cr = []
    for block in q_arr_Cr:
        arr_idct2_Cr.append(idct2(block))
    arr_idct2_Cr = np.array(arr_idct2_Cr)
    Cr = arr_idct2_Cr

    arr_idct2_Cb = []
    for block in q_arr_Cb:
        arr_idct2_Cb.append(idct2(block))
    arr_idct2_Cb = np.array(arr_idct2_Cb)
    Cb = arr_idct2_Cb

    # Reconstruct
    reconstructed_Y = reconstructImage(Y, (A.shape[0], A.shape[1]))
    Y = reconstructed_Y
    reconstructed_Cr = reconstructImage(Cr, shape)
    Cr = reconstructed_Cr
    reconstructed_Cb = reconstructImage(Cb, shape)
    Cb = reconstructed_Cb

    # Chroma resampling
    resampled_Cr = chromaResampling(Cr, J, a, b, (A.shape[0], A.shape[1]))
    Cr = resampled_Cr

    resampled_Cb = chromaResampling(Cb, J, a, b, (A.shape[0], A.shape[1]))
    Cb = resampled_Cb

    Y = np.clip(Y, 0, 255)
    YCrCb = np.dstack([Y, Cr, Cb]).astype(np.uint8)
    RGB = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
    return RGB


y = 800
y2 = y + 256
x = 275
x2 = x + 256

file = "img3.jpg"
img = plt.imread(file)
plt.imshow(img)
plt.show()
img = img[x:x2, y:y2]
chroma_subsampling_combination = "4:2:2"
quant_or_ones = "Tablica jedynek"
Y_compressed, Cr_compressed, Cb_compressed, shape = compress(
    img, chroma_subsampling_combination, quant_or_ones
)
decompressed = decompress(
    Y_compressed,
    Cr_compressed,
    Cb_compressed,
    img,
    shape,
    chroma_subsampling_combination,
    quant_or_ones,
)


plt.imshow(decompressed)
plt.show()

# plt.suptitle("Obraz oryginalny i JPEG")
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(decompressed)
# plt.show()

plt.figure(figsize=(6, 13))
plt.suptitle(
    f"{file} {chroma_subsampling_combination} {quant_or_ones}"
)


plt.subplot(4, 2, 1)
plt.imshow(img)
plt.subplot(4, 2, 3)
plt.title("Y")
plt.imshow(img[:, :, 0], cmap=plt.cm.gray)
plt.subplot(4, 2, 5)
plt.title("Cr")
plt.imshow(img[:, :, 1], cmap=plt.cm.gray)
plt.subplot(4, 2, 7)
plt.title("Cb")
plt.imshow(img[:, :, 2], cmap=plt.cm.gray)
plt.subplot(4, 2, 2)
plt.imshow(decompressed[:, :])
plt.subplot(4, 2, 4)
plt.title("Y")
plt.imshow(decompressed[:, :, 0], cmap=plt.cm.gray)
plt.subplot(4, 2, 6)
plt.title("Cr")
plt.imshow(decompressed[:, :, 1], cmap=plt.cm.gray)
plt.subplot(4, 2, 8)
plt.title("Cb")
plt.imshow(decompressed[:, :, 2], cmap=plt.cm.gray)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.savefig(
    f"{file[:-4]}_{chroma_subsampling_combination.replace(':','')}_{quant_or_ones.replace(' ','')},{x,x2,y,y2}.png"
)
plt.show()