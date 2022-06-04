from cv2 import blur
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2
import random
from skimage import metrics
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

img1 = cv2.imread(
    'C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\images\\img1_orig.jpg')
img2 = cv2.imread(
    'C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\images\\img2_orig.jpg')
img3 = cv2.imread(
    'C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\images\\img3_orig.jpg')
# bgr2rgb
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# def mse(K, I):
#     m, n = K.shape[0], K.shape[1]
#     mse = 1 / m * n * np.sum((K - I) ** 2)
#     return mse

def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    # img1 and 2 should have same shape
    num_pix = img1.shape[0] * img1.shape[1]
    err = summed / num_pix
    return err


def mse(K, I):
    m, n = K.shape[0], K.shape[1]
    mse = 1 / m * n * np.sum((K - I) ** 2)
    return mse


def psnr(K, I):
    psnr = 10 * np.log10(255 ** 2 / MSE(K, I))
    return psnr


def ssim(K, I):
    return metrics.structural_similarity(K, I, channel_axis=-1)


def image_fidelity(K, I):
    return 1 - np.sum((K - I) ** 2) / np.sum(K * I)


def jpg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)

    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(decimg)
    return decimg


def blur_img(img, maskSize):
    median = cv2.medianBlur(img, maskSize)
    return median


def noise_img(img, noise_type, samples_size):
    if noise_type == "speckle":
        gauss = np.random.normal(0, samples_size, img.size)
        gauss = gauss.reshape(
            img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
        return img + img * gauss
    if noise_type == "gaussian":
        gauss = np.random.normal(0, samples_size, img.size)
        gauss = gauss.reshape(
            img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
        return cv2.add(img, gauss)
    if noise_type == "exponential":
        exponential = np.random.exponential(samples_size, img.size)
        exponential = exponential.reshape(
            img.shape[0], img.shape[1], img.shape[2]
        ).astype("uint8")
        return cv2.add(img, exponential)
    if noise_type == "rayleigh":
        rayleigh = np.random.rayleigh(samples_size, img.size)
        rayleigh = rayleigh.reshape(img.shape[0], img.shape[1], img.shape[2]).astype(
            "uint8"
        )
        return cv2.add(img, rayleigh)
    if noise_type == "uniform":
        uniform = np.random.uniform(0, samples_size, img.size)
        uniform = uniform.reshape(img.shape[0], img.shape[1], img.shape[2]).astype(
            "uint8"
        )
        return cv2.add(img, uniform)


def create_blurred_images(img):
    dir = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\blur"
    f = open("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\blur\\dane.txt", "w")
    os.chdir(dir)
    arr = [3, 5, 7, 9, 11, 15, 19, 25, 31, 41]
    for blur in arr:
        filename = f"blurred_maskSize-{blur}.jpg"
        distort = blur_img(img, blur)
        cv2.imwrite(filename, distort)
        data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename, MSE(
            img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
        f.write(data_to_write)


def create_compressed_images(img):
    dir = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg"
    arr = [10, 20, 25, 30, 40, 45, 50, 60, 70, 80]
    f = open("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg\\dane.txt", "w")
    os.chdir(dir)
    for quality in arr:
        filename = f"compressed_{quality}.jpg"
        distort = jpg_compress(img, quality)
        cv2.imwrite(filename, distort)
        data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename, MSE(
            img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
        f.write(data_to_write)
        quality += 10


def create_noise_images(img):
    dir = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\noise"
    f = open("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\noise\\dane.txt", "w")
    os.chdir(dir)
    noise_types = ["gaussian", "rayleigh", "uniform", "exponential", "speckle"]
    noise_str = []
    for types in noise_types:
        if(types in ['gaussian', 'speckle']):
            noise_str = [0.4, 0.8]
        else:
            noise_str = [20, 40]

        for strength in noise_str:
            filename = f"noisy_{types}-{strength}.jpg"
            distort = noise_img(img, types, strength)
            cv2.imwrite(filename, distort)
            data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename, MSE(
                img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
            f.write(data_to_write)


def merge_miary_ocen():
    blur = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\blur\\dane.txt"
    kompresja = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg\\dane.txt"
    noise = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\noise\\dane.txt"

    f = open(
        "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\merge.txt", "w")
    f.write("\"MSE\",\"PSNR\",\"SSIM\",\"FIDELITY\"\n")
    with open(noise) as noise_file:
        for line in noise_file:
            f.write(line)
    with open(blur) as blur_file:
        for line in blur_file:
            f.write(line)
    with open(kompresja) as kompresja_file:
        for line in kompresja_file:
            f.write(line)


# create_noise_images(img3)
# create_blurred_images(img1)
# create_compressed_images(img2)
# merge_miary_ocen()

def plot_mos(pairs_all_separately, pairs_one_subject, pairs_mean):
    fig, axs = plt.subplots(3,3)
    # make 48 markers in array
    markers=[".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    j = 0
    k = 0

    fig.suptitle('MOS')

    for i in range(pairs_all_separately[0][:, 0].size):
        axs[0,0].scatter(pairs_all_separately[0][:, 0][i], pairs_all_separately[0][:, 1][i].astype(float), marker=markers[j], color=colors[k])
        axs[1,0].scatter(pairs_all_separately[1][:, 0][i], pairs_all_separately[1][:, 1][i].astype(float), marker=markers[j+1], color=colors[k])
        axs[2,0].scatter(pairs_all_separately[2][:, 0][i], pairs_all_separately[2][:, 1][i].astype(float), marker=markers[j+2], color=colors[k])
        k = k+1
        j = j+3
        if(k > len(colors)-1):
            k = 0
        if(j > len(markers)-3):
            j = 0

    for i in range(pairs_one_subject[0][:, 0].size):
        axs[0,1].scatter(pairs_one_subject[0][:, 0][i], pairs_one_subject[0][:, 1][i].astype(float), marker=markers[j], color=colors[k])
        axs[1,1].scatter(pairs_one_subject[1][:, 0][i], pairs_one_subject[1][:, 1][i].astype(float), marker=markers[j+1], color=colors[k])
        axs[2,1].scatter(pairs_one_subject[2][:, 0][i], pairs_one_subject[2][:, 1][i].astype(float), marker=markers[j+2], color=colors[k])
        k = k+1
        j = j+3
        if(k > len(colors)-1):
            k = 0
        if(j > len(markers)-3):
            j = 0
    
    for i in range(pairs_mean[0][:, 0].size):
        axs[0,2].scatter(pairs_mean[0][:, 0][i], pairs_mean[0][:, 1][i].astype(float), marker=markers[j], color=colors[k])
        axs[1,2].scatter(pairs_mean[1][:, 0][i], pairs_mean[1][:, 1][i].astype(float), marker=markers[j+1], color=colors[k])
        axs[2,2].scatter(pairs_mean[2][:, 0][i], pairs_mean[2][:, 1][i].astype(float), marker=markers[j+2], color=colors[k])
        k = k+1
        j = j+3
        if(k > len(colors)-1):
            k = 0
        if(j > len(markers)-3):
            j = 0

    axs[0,0].set_title('Mos - wszystkie oceny - Szum')
    axs[1,0].set_title('Mos - wszystkie oceny - Rozmycie')
    axs[2,0].set_title('Mos - wszystkie oceny - Kompresja')
    axs[0,1].set_title('Mos - zagreg. dla użytk. - Szum')
    axs[1,1].set_title('Mos - zagreg. dla użytk. - Rozmycie')
    axs[2,1].set_title('Mos - zagreg. dla użytk. - Kompresja')
    axs[0,2].set_title('Mos - zagreg. - Szum')
    axs[1,2].set_title('Mos - zagreg. - Rozmycie')
    axs[2,2].set_title('Mos - zagreg. - Kompresja')

    fig.set_size_inches(16.,16.)
    filename = "MOS_" + ".png"

    #save
    plt.savefig(filename)

def retRegressionPoints(df):
    x = df[:, 0]
    y = df[:, 1]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    x1 = np.amin(x)
    x2 = np.amax(x)

    pred_y=model.predict(np.array([x1,x2]).reshape(-1, 1))

    return x, y, x1, x2, pred_y

def plot_mos_regression(pairs_all_separately, pairs_one_subject, pairs_mean, norm):
    _all1 = retRegressionPoints(pairs_all_separately[0])
    _all2 = retRegressionPoints(pairs_all_separately[1])
    _all3 = retRegressionPoints(pairs_all_separately[2])

    _one1 = retRegressionPoints(pairs_one_subject[0])
    _one2 = retRegressionPoints(pairs_one_subject[1])
    _one3 = retRegressionPoints(pairs_one_subject[2])

    _mean1 = retRegressionPoints(pairs_mean[0])
    _mean2 = retRegressionPoints(pairs_mean[1])
    _mean3 = retRegressionPoints(pairs_mean[2])

    ### plot regression ###
    fig, axs = plt.subplots(3,3)
    fig.suptitle("Regresja: " + norm, fontsize=16)
    axs[0,0].plot(_all1[0], _all1[1], 'ro')
    axs[0,0].plot(np.array([_all1[2], _all1[3]]), _all1[4], '-k')
    axs[0,0].set_title('Regresja - wszystkie oceny - Szum')
    axs[0,1].plot(_one1[0], _one1[1], 'ro')
    axs[0,1].plot(np.array([_one1[2], _one1[3]]), _one1[4], '-k')
    axs[0,1].set_title('Regresja - zagreg. dla użytk. - Szum')
    axs[0,2].plot(_mean1[0], _mean1[1], 'ro')
    axs[0,2].plot(np.array([_mean1[2], _mean1[3]]), _mean1[4], '-k')
    axs[0,2].set_title('Regresja - zagreg. - Szum')

    axs[1,0].plot(_all2[0], _all2[1], 'ro')
    axs[1,0].plot(np.array([_all2[2], _all2[3]]), _all2[4], '-k')
    axs[1,0].set_title('Regresja - wszystkie oceny - Rozmycie')
    axs[1,1].plot(_one2[0], _one2[1], 'ro')
    axs[1,1].plot(np.array([_one2[2], _one2[3]]), _one2[4], '-k')
    axs[1,1].set_title('Regresja - zagreg. dla użytk. - Rozmycie')
    axs[1,2].plot(_mean2[0], _mean2[1], 'ro')
    axs[1,2].plot(np.array([_mean2[2], _mean2[3]]), _mean2[4], '-k')
    axs[1,2].set_title('Regresja - zagreg. - Rozmycie')

    axs[2,0].plot(_all3[0], _all3[1], 'ro')
    axs[2,0].plot(np.array([_all3[2], _all3[3]]), _all3[4], '-k')
    axs[2,0].set_title('Regresja - wszystkie oceny - Kompresja')
    axs[2,1].plot(_one3[0], _one3[1], 'ro')
    axs[2,1].plot(np.array([_one3[2], _one3[3]]), _one3[4], '-k')
    axs[2,1].set_title('Regresja - zagreg. dla użytk. - Kompresja')
    axs[2,2].plot(_mean3[0], _mean3[1], 'ro')
    axs[2,2].plot(np.array([_mean3[2], _mean3[3]]), _mean3[4], '-k')
    axs[2,2].set_title('Regresja - zagreg. - Kompresja')

    fig.set_size_inches(16.,16.)
    filename = "Regresja_" + norm + ".png"
    #save
    plt.savefig(filename)
    ############

# def plot_mos_regression(pairs_all_separately, pairs_one_subject, pairs_mean, norm):
#     markers=[".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
#     colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
#     j = 0
#     k = 0

#     _all1 = retRegressionPoints(pairs_all_separately[0])
#     _all2 = retRegressionPoints(pairs_all_separately[1])
#     _all3 = retRegressionPoints(pairs_all_separately[2])

#     _one1 = retRegressionPoints(pairs_one_subject[0])
#     _one2 = retRegressionPoints(pairs_one_subject[1])
#     _one3 = retRegressionPoints(pairs_one_subject[2])

#     _mean1 = retRegressionPoints(pairs_mean[0])
#     _mean2 = retRegressionPoints(pairs_mean[1])
#     _mean3 = retRegressionPoints(pairs_mean[2])

#     fig, axs = plt.subplots(3,3)
#     fig.suptitle("Regresja: " + norm, fontsize=16)

#     for i in range(len(_all1)):
#         axs[0,0].scatter(_all1[0][i], _all1[1][i], marker=markers[j], color=colors[k])
#         axs[1,0].scatter(_all2[0][i], _all2[1][i], marker=markers[j+1], color=colors[k])
#         axs[2,0].scatter(_all3[0][i], _all3[1][i], marker=markers[j+2], color=colors[k])
#         k = k+1
#         j = j+3
#         if(k > len(colors)-1):
#             k = 0
#         if(j > len(markers)-3):
#             j = 0

#     for i in range(len(_one1)):
#         axs[0,1].scatter(_one1[0][i], _one1[1][i], marker=markers[j], color=colors[k])
#         axs[1,1].scatter(_one2[0][i], _one2[1][i], marker=markers[j+1], color=colors[k])
#         axs[2,1].scatter(_one3[0][i], _one3[1][i], marker=markers[j+2], color=colors[k])
#         k = k+1
#         j = j+3
#         if(k > len(colors)-1):
#             k = 0
#         if(j > len(markers)-3):
#             j = 0
    
#     for i in range(len(_mean1)):
#         axs[0,2].scatter(_mean1[0][i], _mean1[1][i], marker=markers[j], color=colors[k])
#         axs[1,2].scatter(_mean2[0][i], _mean2[1][i], marker=markers[j+1], color=colors[k])
#         axs[2,2].scatter(_mean3[0][i], _mean3[1][i], marker=markers[j+2], color=colors[k])
#         k = k+1
#         j = j+3
#         if(k > len(colors)-1):
#             k = 0
#         if(j > len(markers)-3):
#             j = 0

#     plt.show()
    
def plotCorrelationMatrix(df):
    ilosc_badanych = 16
    pairs_all_separately = [[], [], []]
    pairs_one_subject = [[], [], []]
    pairs_mean = [[], [], []]
    
    for i in range(10):
        pairs_mean[0].append([df.iloc[i, ilosc_badanych*3:(ilosc_badanych*3)+4].to_numpy(), df["MEAN"][i]])
    
    for i in range(10, 20):
        pairs_mean[1].append([df.iloc[i, ilosc_badanych*3:(ilosc_badanych*3)+4].to_numpy(), df["MEAN"][i]])
    
    for i in range(20, 30):
        pairs_mean[2].append([df.iloc[i, ilosc_badanych*3:(ilosc_badanych*3)+4].to_numpy(), df["MEAN"][i]])

    # pairs_mean[0] = np.array(pairs_mean[0])
    # pairs_mean[1] = np.array(pairs_mean[1])
    # pairs_mean[2] = np.array(pairs_mean[2])
    print(type(pairs_mean[0]))

    corr_matrix1 = np.corrcoef(pairs_mean[0]).round(decimals=2)
    corr_matrix2 = np.corrcoef(pairs_mean[1]).round(decimals=2)
    corr_matrix3 = np.corrcoef(pairs_mean[2]).round(decimals=2)

    # print(corr_matrix1)

    return pairs_mean

def make_pairs_regression(df, norm):
    ilosc_badanych = 16
    pairs_all_separately = [[], [], []]
    pairs_one_subject = [[], [], []]
    pairs_mean = [[], [], []]

    for i in range(10):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            norm_val = df[norm][i]
            pairs_all_separately[0].append([norm_val,user_values.iloc[i, j]])

    for i in range(10, 20):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            norm_val = df[norm][i]
            pairs_all_separately[1].append([norm_val,user_values.iloc[i, j]])
    
    for i in range(20, 30):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            norm_val = df[norm][i]
            pairs_all_separately[2].append([norm_val,user_values.iloc[i, j]])

    for i in range(10):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            norm_val = df[norm][i]
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[0].append([norm_val,tmp_arr.mean()])

    for i in range(10, 20):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            norm_val = df[norm][i]
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[1].append([norm_val,tmp_arr.mean()])

    for i in range(20, 30):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            norm_val = df[norm][i]
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[2].append([norm_val,tmp_arr.mean()])

    for i in range(10):
        norm_val = df[norm][i]
        pairs_mean[0].append([norm_val, df["MEAN"][i]])
    
    for i in range(10, 20):
        norm_val = df[norm][i]
        pairs_mean[1].append([norm_val, df["MEAN"][i]])
    
    for i in range(20, 30):
        norm_val = df[norm][i]
        pairs_mean[2].append([norm_val, df["MEAN"][i]])

    pairs_all_separately[0] = np.array(pairs_all_separately[0])
    pairs_all_separately[1] = np.array(pairs_all_separately[1])
    pairs_all_separately[2] = np.array(pairs_all_separately[2])
    pairs_mean[0] = np.array(pairs_mean[0])
    pairs_mean[1] = np.array(pairs_mean[1])
    pairs_mean[2] = np.array(pairs_mean[2])
    pairs_one_subject[0] = np.array(pairs_one_subject[0])
    pairs_one_subject[1] = np.array(pairs_one_subject[1])
    pairs_one_subject[2] = np.array(pairs_one_subject[2])

    return pairs_mean, pairs_all_separately, pairs_one_subject


def make_pairs(df):
    ilosc_badanych = 16
    pairs_all_separately = [[], [], []]
    pairs_one_subject = [[], [], []]
    pairs_mean = [[], [], []]

    for i in range(10):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            imgname = "img"+str(i)
            pairs_all_separately[0].append([imgname,user_values.iloc[i, j]])

    for i in range(10, 20):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            imgname = "img"+str(i)
            pairs_all_separately[1].append([imgname,user_values.iloc[i, j]])
    
    for i in range(20, 30):
        user_values = df.iloc[:, :ilosc_badanych*3]
        for j in range(user_values.shape[1]):
            imgname = "img"+str(i)
            pairs_all_separately[2].append([imgname,user_values.iloc[i, j]])

    for i in range(10):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            imgname = "img"+str(i)
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[0].append([imgname,tmp_arr.mean()])

    for i in range(10, 20):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            imgname = "img"+str(i)
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[1].append([imgname,tmp_arr.mean()])

    for i in range(20, 30):
        user_values = data_array.iloc[:, :ilosc_badanych*3]
        for j in range(ilosc_badanych):
            row = user_values.iloc[i]
            imgname = "img"+str(i)
            badany1 = 'badany' + str(j+1) + '_' + str(0)
            badany2 = 'badany' + str(j+1) + '_' + str(1)
            badany3 = 'badany' + str(j+1) + '_' + str(2)
            tmp_arr = row[[badany1,badany2,badany3]]
            pairs_one_subject[2].append([imgname,tmp_arr.mean()])

    for i in range(10):
        imgname = "img"+str(i)
        pairs_mean[0].append([imgname, df["MEAN"][i]])

    for i in range(10, 20):
        imgname = "img"+str(i)
        pairs_mean[1].append([imgname, df["MEAN"][i]])

    for i in range(20, 30):
        imgname = "img"+str(i)
        pairs_mean[2].append([imgname, df["MEAN"][i]])

    pairs_all_separately[0] = np.array(pairs_all_separately[0])
    pairs_all_separately[1] = np.array(pairs_all_separately[1])
    pairs_all_separately[2] = np.array(pairs_all_separately[2])
    pairs_mean[0] = np.array(pairs_mean[0])
    pairs_mean[1] = np.array(pairs_mean[1])
    pairs_mean[2] = np.array(pairs_mean[2])
    pairs_one_subject[0] = np.array(pairs_one_subject[0])
    pairs_one_subject[1] = np.array(pairs_one_subject[1])
    pairs_one_subject[2] = np.array(pairs_one_subject[2])
    

    return pairs_mean, pairs_all_separately, pairs_one_subject


os.chdir("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10")
data_array = pd.read_csv("jakosc_obrazow.csv").T
merge_array = pd.read_csv("merge.txt")

# data_array = makeManyRevievs(data_array)
data_array["MEAN"] = data_array.mean(axis=1)
data_array['MSE'] = merge_array['MSE'].tolist()
data_array['PSNR'] = merge_array['PSNR'].tolist()
data_array['SSIM'] = merge_array['SSIM'].tolist()
data_array['FIDELITY'] = merge_array['FIDELITY'].tolist()

norms = data_array.iloc[:, -4:].columns
pairs_mean_mos, pairs_all_separately_mos, pairs_one_subject_mos = make_pairs(data_array)
plot_mos(pairs_all_separately_mos, pairs_one_subject_mos, pairs_mean_mos)
for norm in norms:
    pairs_mean_regression, pairs_all_separately_regression, pairs_one_subject_regression = make_pairs_regression(data_array, norm)
    plot_mos_regression(pairs_all_separately_regression, pairs_one_subject_regression, pairs_mean_regression, norm)
    

# plotCorrelationMatrix(data_array)
