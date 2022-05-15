from cProfile import label
from copy import copy
from re import X
from tracemalloc import start
from xml.etree.ElementTree import tostring
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
from sqlalchemy import null

A = 87.6

def convert_float32_to_int(x, type=None):
    x = copy(x)
    fmin = -1
    fmax = 1
    min = 0
    max = 0
    
    if(type is None):
        min = np.iinfo(np.int16).min
        max = np.iinfo(np.int16).max
        type = np.int16
    else:
        min = np.iinfo(type).min
        max = np.iinfo(type).max

    x = (x - fmin) / (fmax - fmin)
    x = x * (max - min) + min

    return x.astype(type)

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



def DPCM_no_pred_encode(x, bits):
    x = copy(x)
    type = x.dtype
    X_prime = np.zeros(len(x)).astype(type)
    Y_prime = np.int16(0).astype(type)
    e = np.int16(0).astype(type)
    Y = np.zeros(len(x)).astype(type)

    for i in range(1, len(x)):


        Y_prime = x[i] - e
        Y[i] = quantize(Y_prime, bits)
        X_prime[i] = Y[i] + e
        e = X_prime[i]
        
    
    # Y = Y[3:]
    return Y.astype(type)

def DPCM_no_pred_decode(y):
    type = y.dtype
    y = copy(y)
    X_prime = np.zeros(len(y)).astype(type)
    e = 0

    for i in range(len(y)):
        if(i==0):
            X_prime[i] = y[i]
        else:
            X_prime[i] = y[i] + X_prime[i-1]

    return X_prime.astype(type)



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
    x = (x * (max - min)) + min
    # print("quant: ",x)
    # print("quant dtype: ", typ)
    # print("quant min: ", min, "quant max: ", max)
    return np.round(x.astype(typ), 2)

fig, axs = plt.subplots(1,2)
fig.suptitle('Kompresja a-law', fontsize=16)
fig.legend()
axs[0].plot(x, A_Law_compress(x, 8), label="Sygnał po kompresji A-law po kwantyzacji do 8-bitów")
axs[0].plot(x, A_Law_compress(x, 8), label="Sygnał po kompresji A-law bez kwantyzacji")
axs[0].set_title('Krzywa kompresji')
axs[0].legend(loc='upper left')

axs[1].plot(x,x, label="Sygnał oryginalny")
axs[1].plot(x,A_Law_decompress(A_Law_compress(x, 8)), label="Sygnał po dekompresji A-law po kwantyzacji do 8-bitów")
axs[1].plot(x,A_Law_decompress(A_Law_compress(x, 0)), label="Sygnał po dekompresji A-law bez kwantyzacji")
axs[1].set_title('Krzywa dekompresji')
axs[1].legend(loc='upper left')

plt.show()

fig, axs = plt.subplots(5,1)
fig.suptitle('Kompresje sinusa', fontsize=16)

axs[0].plot(x, y)
axs[0].set_title('Sygnał oryginalny')
alawww = A_Law_compress(y, 8)
axs[1].plot(x, alawww)
axs[1].set_title('Kompresja A-law')

dpcm_compressed = DPCM_no_pred_encode(y,8)
dpcm_decompressed = DPCM_no_pred_decode(dpcm_compressed)

axs[2].plot(x, dpcm_compressed)
axs[2].set_title('Kompresja DPCM')

axs[3].plot(x, A_Law_decompress(alawww))
axs[3].set_title('Dekompresja A-law')

axs[4].plot(x, dpcm_decompressed)
axs[4].set_title('Dekompresja DPCM')

plt.show()

plt.plot(x, y, label="Sygnał oryginalny")
plt.plot(x, A_Law_decompress(A_Law_compress(y, 8)), label="Sygnał po dekompresji z A-law")
plt.plot(x, dpcm_decompressed, label="Sygnał po dekompresji z DPCM")
plt.legend(loc='upper left')

plt.show()

# soundNames = ["sin_60Hz", "sin_440Hz", "sin_8000Hz", "sin_combined", "sing_low1", "sing_low2", "sing_high1", "sing_high2", "sing_medium1", "sing_medium2"]
# soundNames = ["sing_high1", "sing_high2", "sing_medium1", "sing_medium2"]
# for soundName in soundNames:
#     soundFile_name = "SM_Lab05/"+soundName+".wav"
#     samplerate, audio = read(soundFile_name)
#     length = audio.shape[0] / samplerate
#     time = np.linspace(0., length, audio.shape[0])
#     audio_type = audio.dtype
#     timeDPCM = time
#     print(soundName)
#     for i in range(2,9):
#         audio_float = convert_int_to_float32(audio)
#         A_compressed = A_Law_compress(audio_float, i)
#         A_decompressed = A_Law_decompress(A_compressed)
#         DPCM_compressed = DPCM_no_pred_encode(audio, i)
#         DPCM_decompressed = DPCM_no_pred_decode(DPCM_compressed)

#         fig, axs = plt.subplots(5,1)
#         fig.suptitle('Kompresje dla '+str(i)+" bit", fontsize=16)

#         axs[0].plot(time, audio)
#         axs[0].set_title('Sygnał oryginalny')

#         axs[1].plot(timeDPCM, DPCM_compressed)
#         axs[1].set_title('Krzywa kompresji dpcm')

#         axs[2].plot(timeDPCM, DPCM_decompressed)
#         axs[2].set_title('Krzywa dekompresji dpcm')

#         axs[3].plot(time, A_compressed)
#         axs[3].set_title('Krzywa kompresji alaw')

#         axs[4].plot(time, A_decompressed)
#         axs[4].set_title('Krzywa dekompresji alaw')

#         dpcm =  "DPCM_"+str(i)+"_bit_"+soundName+".wav"
#         alaw =  "A-law_"+str(i)+"_bit_"+soundName+".wav"
#         c = "kompresja_"+str(i)+"_bit_"+soundName+".png"

#         write(dpcm, samplerate, DPCM_decompressed.astype(audio_type))
#         write(alaw, samplerate, convert_float32_to_int(A_decompressed, audio_type))
#         print(i)
#         plt.gcf().set_size_inches(30, 25)
#         plt.savefig(c, bbox_inches='tight')
