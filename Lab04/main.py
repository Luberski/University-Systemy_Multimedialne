import cv2
import pyaudio
import matplotlib.pyplot as plt
import numpy as np

#np.linspace(0,1,x - ilość wartości).reshape(x,1) - wrzucamy do colorfita
#littering losowy - wartości losowe - np.random.rand(img.shape)

img = plt.imread('C:\\Users\\Srat Lord\\Projects\\Systemy_Multimedialne\\Lab04\\img\\0001.jpg')

img = img/255 #zamiana na float

palette = np.array([
    [0.,0.,0.],          #black
    [0.,1.,1.],          #aqua
    [0.,0.,1.],          #blue
    [1.,0.,1.],          #fuchsia
    [0.,0.5,0.],        #green
    [0.5,0.5,0.5],    #grey
    [0.,1.,0.],          #lime
    [0.5,0.,0.],        #maroon
    [0.,0.,0.5],        #navy
    [0.5,0.5,0.],      #olive
    [0.5,0.,0.5],      #purple
    [1.,0.,0.],          #red
    [0.75,0.75,0.75], #silver
    [0.,0.5,0.5],      #teal
    [1.,1.,1.],          #white
    [1.,1.,0.]          #yellow
])

def colorFit(color, palette):
    return palette[np.argmin(np.linalg.norm(color - palette, axis=1))]

def random_dithering():
    if(img.ndim >= 3):
        exit()
    

def m2_dithering():
    pass

def floyd_steinberg():
    pass

# print(colorFit(0.4, img))

# x = np.linspace(0, 1, 10).reshape(10,1)
# print(colorFit(x,palette))

print(img.ndim)

