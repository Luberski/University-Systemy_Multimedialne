from cProfile import label
from copy import copy
from re import X
from tracemalloc import start
from xml.etree.ElementTree import tostring
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write

A = 87.6

def quantize(x, bits):
    x = copy(x)
    min = 0
    max = 0
    typ = x.dtype

    if np.issubdtype(typ, np.floating):
        min = -1
        max = 1
    else:
        min = np.iinfo(typ).min
        max = np.iinfo(typ).max

    x = x.astype(np.float32)
    x = (x - min) / (max - min)
    x = (np.round(x * (2 ** bits - 1))) / (2 ** bits - 1)
    x = x * (max - min) + min

    return np.round(x.astype(typ), 2)

def convert_float32_to_int(x):
    x = copy(x)
    fmin = -1
    fmax = 1
    min = np.iinfo(np.int16).min
    max = np.iinfo(np.int16).max
    x = (x - fmin) / (fmax - fmin)
    x = x * (max - min) + min
    x = np.round(x).astype(np.int16)

    return x

def convert_int_to_float32(x):
    x = copy(x)
    type = x.dtype
    fmin = -1
    fmax = 1
    min = np.iinfo(type).min
    max = np.iinfo(type).max
    x = x.astype(np.float32)
    x = (x - min) / (max - min)
    x = x * (fmax - fmin) + fmin

    return x

x = np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)

def A_Law_compress(x, bits):
    x = copy(x)
    for i in range(len(x)):
        if np.abs(x[i]) < (1/A):
            x[i] = np.sign(x[i])*((A*np.abs(x[i]))/(1+np.log(A)))
        elif(np.abs(x[i]) >= (1/A) and np.abs(x[i]) <= 1):
            x[i] = np.sign(x[i])*((1+np.log(A*np.abs(x[i]))) / (1+np.log(A)))

    if(bits != 0):
        x = quantize(x, bits)

    return x

def A_Law_decompress(y):
    y = copy(y)
    for i in range(len(y)):
        if np.abs(y[i]) < (1/(1+np.log(A))):
             y[i] = np.sign(y[i]) * (np.abs(y[i]) * (1 + np.log(A))) / (A)
        elif(np.abs(y[i]) >= (1/(1+np.log(A))) and np.abs(y[i]) <= 1):
            y[i] = np.sign(y[i]) * (np.exp(np.abs(y[i]) * (1 + np.log(A)) - 1)) / (A)

    return y

X = [15,16,20,14,5,10,15,13,11,7,10,11,20,1,23]

def DPCM_no_pred_encode(x, bits):
    x = copy(x)
    type = x.dtype
    X_prime = np.zeros(len(x)).astype(type)
    e = 0
    Y = np.zeros(len(x)).astype(type)

    for i in range(len(x)):
        Y_prime = x[i] - e
        Y[i] = quantize(Y_prime, bits)
        X_prime[i] = Y[i] + e
        e = X_prime[i]
        print("Y_prime: ", Y_prime)
        print("Y: ", Y[i])
        print("X_prime: ", X_prime[i])
        print("e: ", e)
        print("\n")

    return Y.astype(type)

print(DPCM_no_pred_encode(x, 8))

def DPCM_no_pred_decode(y):
    y = copy(y)
    type = y.dtype
    X_prime = np.zeros(len(y)).astype(type)
    e = 0

    for i in range(len(y)):
        if(i==0):
            X_prime[i] = y[i]
        else:
            X_prime[i] = y[i] + X_prime[i-1]

    return X_prime



# fig, axs = plt.subplots(1,2)
# fig.suptitle('Kompresja a-law', fontsize=16)
# fig.legend()
# axs[0].plot(x, A_Law_compress(x, 8), label="Sygnał po kompresji A-law po kwantyzacji do 8-bitów")
# axs[0].plot(x, A_Law_compress(x, 8), label="Sygnał po kompresji A-law bez kwantyzacji")
# axs[0].set_title('Krzywa kompresji')
# axs[0].legend(loc='upper left')

# axs[1].plot(x,x, label="Sygnał oryginalny")
# axs[1].plot(x,A_Law_decompress(A_Law_compress(x, 8)), label="Sygnał po dekompresji A-law po kwantyzacji do 8-bitów")
# axs[1].plot(x,A_Law_decompress(A_Law_compress(x, 0)), label="Sygnał po dekompresji A-law bez kwantyzacji")
# axs[1].set_title('Krzywa dekompresji')
# axs[1].legend(loc='upper left')

# plt.show()

# fig, axs = plt.subplots(3,1)
# fig.suptitle('Kompresje sinusa', fontsize=16)

# axs[0].plot(x, y)
# axs[0].set_title('Sygnał oryginalny')

# axs[1].plot(x, A_Law_compress(y, 8))
# axs[1].set_title('Kompresja A-law')

# dpcm_data = convert_float32_to_int(y)
# dpcm_compressed = DPCM_no_pred_encode(dpcm_data)
# dpcm_float = convert_int_to_float32(dpcm_compressed)
# axs[2].plot(x, dpcm_float)
# axs[2].set_title('Kompresja DPCM')

# plt.show()

# dpcm_decompressed = DPCM_no_pred_decode(dpcm_compressed)
# dpcm_float = convert_int_to_float32(dpcm_decompressed)
# plt.plot(x, y, label="Sygnał oryginalny")
# plt.plot(x, A_Law_decompress(A_Law_compress(y, 8)), label="Sygnał po dekompresji z A-law")
# plt.plot(x, dpcm_float, label="Sygnał po dekompresji z DPCM")
# plt.legend(loc='upper left')

# plt.show()

# soundNames = ["sin_60Hz", "sin_440Hz", "sin_8000Hz", "sin_combined", "sing_low1", "sing_low2", "sing_high1", "sing_high2", "sing_medium1", "sing_medium2"]
soundNames = ["sing_low1", "sing_high1", "sing_medium1"]
for soundName in soundNames:
    soundFile_name = "SM_Lab05/"+soundName+".wav"
    input_data = read(soundFile_name)
    audio = input_data[1]

    # for i in range(2,9):
    #     fig, axs = plt.subplots(4,1)
    #     audio_float = convert_int_to_float32(audio)
    #     data = A_Law_compress(audio_float, i)
    #     data_decompressed = A_Law_decompress(data)
    #     title = "Kompresja A-law "+str(i)+" bit " + soundName
    #     fig.suptitle(title, fontsize=16)

    #     axs[0].plot(audio)
    #     axs[0].set_title('Sygnał oryginalny')

    #     axs[1].plot(data)
    #     axs[1].set_title('Krzywa kompresji')

    #     axs[2].plot(data_decompressed)
    #     axs[2].set_title('Krzywa dekompresji')

    #     axs[3].plot(quantize(audio, i))
    #     axs[3].set_title('Czysta kwantyzacja')

    #     filename = "A-law_"+str(i)+"_bit_"+soundName+".png"
    #     fig.set_size_inches(30.,18.)
    #     plt.savefig(filename, bbox_inches='tight')



    fig, axs = plt.subplots(3,1)
    data = DPCM_no_pred_encode(audio, 8)
    data_decompressed = DPCM_no_pred_decode(data)
    # title = "Kompresja DPCM " + soundName
    # fig.suptitle(title, fontsize=16)

    # axs[0].plot(audio)
    # axs[0].set_title('Sygnał oryginalny')

    # axs[1].plot(data)
    # axs[1].set_title('Krzywa kompresji')

    # axs[2].plot(data_decompressed)
    # axs[2].set_title('Krzywa dekompresji')

    # filename = "DPCM_"+soundName+".wav"
    # fig.set_size_inches(30.,18.)
    # plt.savefig(filename, bbox_inches='tight')
    
    # samplerate = 44100; fs = 100
    # t = np.linspace(0., 1., samplerate)
    # amplitude = np.iinfo(np.int16).max
    # obj = wave.open('sound.wav','wb')


    # fig, axs = plt.subplots(4,2)
    # audio_float = convert_int_to_float32(audio[0:1000])
    # Adata = A_Law_compress(audio_float, 8)
    # Adata_decompressed = A_Law_decompress(Adata)
    # title = "Kompresja A-law 8 bit " + soundName
    # fig.suptitle(title, fontsize=16)
    # Ddata = DPCM_no_pred_encode(audio[0:1000])
    # Ddata_decompressed = DPCM_no_pred_decode(Ddata)
    # title = "Kompresja DPCM " + soundName
    # fig.suptitle(title, fontsize=16)

    # axs[1,0].plot(audio[0:1000])
    # axs[1,0].set_title('Sygnał oryginalny')

    # axs[0,1].plot(convert_int_to_float32(Ddata))
    # axs[0,1].set_title('Krzywa kompresji DPCM')

    # axs[1,1].plot(Adata)
    # axs[1,1].set_title('Krzywa kompresji A_Law')

    # axs[2,1].plot(convert_int_to_float32(Ddata_decompressed))
    # axs[2,1].set_title('Krzywa dekompresji DPCM')

    # axs[3,1].plot(Adata_decompressed)
    # axs[3,1].set_title('Krzywa dekompresji A_Law')

    # filename = "DPCMvsAlaw"+soundName+".png"
    # fig.set_size_inches(30.,18.)
    # plt.savefig(filename, bbox_inches='tight')