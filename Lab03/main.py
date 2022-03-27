import cv2
import pyaudio
import matplotlib.pyplot as plt
import numpy as np

# img1 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0001.jpg')
# img2 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0002.jpg')
# img3 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0003.jpg')
# img4 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0004.jpg')
# img5 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0005.jpg')
# img6 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0006.jpg')
# img7 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0007.jpg')
# img8 = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab03\\SM_Lab03\\0008.tif')

# if (len(OG.shape)<3): 
#     ### grayscale
# else:
#     ### RGB


img_out = np.zeros((3,3,3), dtype=np.uint8)
img= np.zeros((3,3,3),dtype=np.uint8)
img[1,1,:]=255

print(img)



