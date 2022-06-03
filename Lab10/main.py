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

img1 = cv2.imread('images\\img1_orig.jpg')
img2 = cv2.imread('images\\img2_orig.jpg')
img3 = cv2.imread('images\\img3_orig.jpg')
# bgr2rgb
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# def mse(K, I):
#     m, n = K.shape[0], K.shape[1]
#     mse = 1 / m * n * np.sum((K - I) ** 2)
#     return mse
    
def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
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
        gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
        return img + img * gauss
    if noise_type == "gaussian":
        gauss = np.random.normal(0, samples_size, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype("uint8")
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
        data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename,MSE(img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
        f.write(data_to_write)

def create_compressed_images(img):
    dir = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg"
    arr = [10,20,25,30,40,45,50,60,70,80]
    f = open("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg\\dane.txt", "w")
    os.chdir(dir)
    for quality in arr:
        filename = f"compressed_{quality}.jpg"
        distort = jpg_compress(img, quality)
        cv2.imwrite(filename, distort)
        data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename,MSE(img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
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
            data_to_write = "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" % (filename,MSE(img, distort), psnr(img, distort), ssim(img, distort), image_fidelity(img, distort))
            f.write(data_to_write)


def merge_miary_ocen():
    blur = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\blur\\dane.txt"
    kompresja = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\kompresjajpg\\dane.txt"
    noise = "C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\badania\\noise\\dane.txt"

    f = open("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\merge.txt", "w")
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

def fakeDataFrame(dataframe):
    new_dataframe = dataframe.copy(deep=True)

    for col in new_dataframe:
        for i in range(new_dataframe[col].size):
            chance = random.random()
            if(chance < 0.2):
                if(new_dataframe[col][i] == 5):
                    new_dataframe[col][i] = new_dataframe[col][i]-1
                elif(new_dataframe[col][i] == 0):
                    new_dataframe[col][i] = new_dataframe[col][i]+1
                else:
                    if(random.random() > 0.5):
                        new_dataframe[col][i] = new_dataframe[col][i]-1
                    else:
                        new_dataframe[col][i] = new_dataframe[col][i]+1

    return new_dataframe

def makeManyRevievs(dataToChange):
    
    new_data1 = fakeDataFrame(dataToChange)
    new_data2 = fakeDataFrame(dataToChange)

    new_data1.columns = new_data1.columns.str.replace('0', '1')
    new_data2.columns = new_data2.columns.str.replace('0', '2')

    new_data = pd.concat([new_data1, new_data2], axis=1)
    new_data = pd.concat([dataToChange, new_data], axis=1)

    new_data.to_csv("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\many_revievs.csv", index=False)
    return new_data



# create_noise_images(img3)
# create_blurred_images(img1)
# create_compressed_images(img2)
# merge_miary_ocen()

def make_pairs(df, norm):
    pairs_all_separately = [[] for i in range(df.shape[0])]
    pairs_one_subject = [[] for i in range(df.shape[0])]
    pairs_mean = []

    for i in range(df.shape[0]):
        user_values = df.iloc[:, :48]
        for j in range(user_values.shape[1]):
            norm_val = df[norm].values[i]
            pairs_all_separately[i].append([norm_val,user_values.iloc[i, j]])

#     for i in range(df.shape[0]):
#         user_values = df.iloc[:, :48]
#         for j in range(user_values.shape[1]):
#             norm_val = df[norm].values[i]
#             pairs_all_separately[i].append([norm_val,user_values.iloc[i, j]])

    # for i in range(df.shape[0]):
    #     norm_val = df[norm][i]
    #     pairs_mean.append([norm_val, df["MEAN"][i]])

    for i in range(df.shape[0]):
        norm_val = df[norm][i]
        imgname = "img"+str(i+1)
        pairs_mean.append([df["MEAN"][i], imgname])

    return np.array(pairs_all_separately), np.array(pairs_one_subject), np.array(pairs_mean)


os.chdir("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10")
data_array = pd.read_csv("jakosc_obrazow.csv").T
merge_array = pd.read_csv("merge.txt")
a = pd.read_csv("a.txt")
data_array = makeManyRevievs(data_array)
data_array["MEAN"] = data_array.mean(axis=1)
data_array['MSE'] = merge_array['MSE'].tolist()
data_array['PSNR'] = merge_array['PSNR'].tolist()
data_array['SSIM'] = merge_array['SSIM'].tolist()
data_array['FIDELITY'] = merge_array['FIDELITY'].tolist()

pairs_all_separately, pairs_one_subject, pairs_mean = make_pairs(data_array, "MSE")

# model = LinearRegression()
# model.fit(pairs_mean[:,0],pairs_mean[:,1])

y = pairs_mean[:,0].astype(float)
x = pairs_mean[:,1]
# print(y)
# print(pairs_mean[:,0])
# print(y)
plt.plot(x, y, 'o')
plt.show()
# print(type(y[0]))
# x = ["Apple1", "Banana2", "Cherry3"]
# y = [5,2,3]

# plt.plot(x, y)

plt.show()