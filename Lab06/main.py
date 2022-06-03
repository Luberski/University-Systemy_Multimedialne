import numpy as np
import matplotlib.pyplot as plt

A = 87.6

x=np.linspace(-1,1,1000, dtype=np.float32)
a1 = np.round(np.linspace(0,255,255,dtype=np.uint8))
a2 = np.round(np.linspace(np.iinfo(np.int32).min,np.iinfo(np.int32).max,1000,dtype=np.int32))

def colorFit(color, palette):
    return palette[np.argmin(np.linalg.norm(color - palette, axis=1))]

def A_Law_compress(x):
    for i in range(len(x)):
        if abs(x[i]) < (1/A):
            x[i] = np.sign(x[i])*(A*abs(x[i]))/(1+np.log(A))
        elif(abs(x[i]) >= 1/A and abs(x[i]) <= 1):
            x[i] = np.sign(x[i])*(1+np.log(A*abs(x[i]))) / (1+np.log(A))

    return x

def A_Law_decompress(y):
    for i in range(len(y)):
        if abs(y[i]) < (1/(1+np.log(A))):
            y[i] = np.sign(y[i])*(abs(y[i])*(1+np.log(A)))/A
        elif(abs(y[i]) >= (1/(1+np.log(A))) and abs(y[i]) <= 1):
            y[i] = np.sign(y[i])*(np.exp(abs(y[i])*(1+np.log(A)))-1)/A

    return y

def quantize(data, n):
    min = 0
    max = 0
    typ = data.dtype
    if(np.issubdtype(typ,np.floating)):
        min = -1
        max = 1
    else:
        min = np.iinfo(typ).min
        max = np.iinfo(typ).max

    data = data.astype(np.float32)
    data = (data-min)/(max-min)
    data = (np.round(data*(2**n-1)))/(2**n-1)
    data = data*(max-min) + min
    return np.round(data.astype(typ),2)

print(quantize(a1, 2))

y = A_Law_compress(x)
y = quantize(y, 8)
# y = A_Law_decompress(y)
plt.plot(x, y)
plt.show()

# print(np.iinfo(np.int32).max)
# print(np.issubdtype(a1.dtype,np.integer))
# print(np.dtype(a2[0]))