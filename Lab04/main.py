import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('img/0008.png')

if(img.shape[1] > 3):
    img = img[:,:,0:3]

if(img.dtype != 'float32'):
    img = img/255

def generate_grayscale_palette(nc):
    palette = np.zeros((nc,3))
    for i in range(nc):
        palette[i, :] = i*(1.0/(nc-1))
    
    return palette

palette4b = np.array([
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]).astype('float32')

palette8b = np.array([
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
]).astype('float32')

# n = 1 -> 2x2 (2 colors per pixel (1 bit))
# n = 2 -> 4x4 (4 colors per pixel (2 bits))
# n = 4 -> 8x8 (8 colors per pixel (4 bits))
# n = 8 -> 16x16 (16 colors per pixel (8 bits))
# ...
def bayer(n):
    if n == 1:
        return np.array([[0,2],[3,1]]).astype('float32')/((2*n)**2)
    M = np.array(((2*n)**2)*bayer(int(n/2))).astype('float32')
    return np.concatenate((np.concatenate((M, M+2), axis=1), np.concatenate((M+3, M+1), axis=1)), axis=0).astype('float32')/((2*n)**2)

def transform_bayer_to_preprocessed_matrix(bayer, n):
    return ((  (bayer*((2*n)**2)) +1)    /   ((2*n)**2)) - 0.5


def colorFit(color, palette):
    return palette[np.argmin(np.linalg.norm(color - palette, axis=1))]

def random_dittering(img):
    img_out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            if(r < img[i,j,0]):
                img_out[i,j] = 255
            else:
                img_out[i,j] = 0
    
    return img_out

def threshold_mapping(img, palette):
    img_out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_out[i,j] = colorFit(img[i,j], palette)
    return img_out

# ordered dithering
def ordered_dithering(img, nc, palette):
    bayer_matrix = bayer(nc)
    img_out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_out[i,j] = colorFit(img[i,j] + transform_bayer_to_preprocessed_matrix(bayer(nc),nc)[i%(nc*2),j%(nc*2)], palette)
    return img_out

# floyd steinberg dithering
def floyd_steinberg_dithering(img, palette):
    img_out = np.zeros(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            old_pixel = img[i,j].copy()
            new_pixel = colorFit(old_pixel, palette)
            img_out[i,j] = new_pixel
            quant_error = old_pixel - new_pixel
            if j + 1 < w:
                img[i, j + 1] += quant_error * 7 / 16 # prawo
            if (i + 1 < h) and (j + 1 < w):
                img[i + 1, j + 1] += quant_error * 1 / 16 # prawy dolny róg
            if i + 1 < h:
                img[i + 1, j] += quant_error * 5 / 16 # dół
            if (j - 1 >= 0) and (i + 1 < h): 
                img[i + 1, j - 1] += quant_error * 3 / 16 # lewy dolny róg

    return img_out
    
gray1 = generate_grayscale_palette(2)
gray2 = generate_grayscale_palette(4)
gray4 = generate_grayscale_palette(8)

# plot dithering
fig, axs = plt.subplots(1,5)
fig.suptitle('Image 0008.png grayscale', fontsize=16)
axs[0].imshow(img)
axs[0].set_title('Original')

# axs[1].imshow(threshold_mapping(img, palette4b))
# axs[1].set_title('Threshold mapping')
# axs[2].imshow(ordered_dithering(img, int(palette4b.shape[0]/2), palette4b))
# axs[2].set_title('Ordered dithering')
# axs[3].imshow(floyd_steinberg_dithering(img, palette4b))
# axs[3].set_title('Floyd-Steinberg dithering')

# axs[1].imshow(threshold_mapping(img, palette8b))
# axs[1].set_title('Threshold mapping')
# axs[2].imshow(ordered_dithering(img, int(palette8b.shape[0]/2), palette8b))
# axs[2].set_title('Ordered dithering')
# axs[3].imshow(floyd_steinberg_dithering(img, palette8b))
# axs[3].set_title('Floyd-Steinberg dithering')

axs[1].imshow(threshold_mapping(img, gray4))
axs[1].set_title('Threshold mapping 4 bit')
axs[2].imshow(random_dittering(img))
axs[2].set_title('Random dithering 4 bit')
axs[3].imshow(ordered_dithering(img, int(gray4.shape[0]/2), gray4))
axs[3].set_title('Ordered dithering 4 bit')
axs[4].imshow(floyd_steinberg_dithering(img, gray4))
axs[4].set_title('Floyd-Steinberg dithering 4 bit')


plt.show()