import numpy as np
import matplotlib.pyplot as plt

# np.issubdtype(x.dtype,np.integer)
# np.issubdtype(x.dtype,np.floating)

# R=np.random.rand(5,5)
# Z=zeros(R.shape)
# idx1=R<0.25
# Z[idx1]=1
# print(R)
# print(Z)

A = 87.6

x=np.linspace(-1,1,1000)
a1 = np.round(np.linspace(0,255,255,dtype=np.uint8))
a2 = np.round(np.linspace(np.iinfo(np.int32).min,np.iinfo(np.int32).max,1000,dtype=np.int32))

def A_Law_compress(x):
    for i in range(len(x)):
        if abs(x[i]) < (1/A):
            x[i] = np.sign(x[i])*(A*abs(x[i]))/(1+np.log(A))
        else:
            x[i] = np.sign(x[i])*(1+np.log(A*abs(x[i]))) / (1+np.log(A))

    return x

def A_Law_decompress(y):
    for i in range(len(y)):
        if abs(y[i]) < (1/(1+np.log(A))):
            y[i] = np.sign(y[i])*((np.log(A)+1)/(A))
        else:
            y[i] = np.sign(y[i])*(1+np.log(A*abs(y[i]))) / (1+np.log(A))

    return y

def quantize(x, n):
    min = 0
    max = 0

    if(np.issubdtype(x.dtype,np.floating)):
        min = -1
        max = 1
    elif(np.issubdtype(x.dtype,np.integer)):
        min = np.iinfo(np.int32).min
        max = np.iinfo(np.int32).max

    x = (x-min)/(max-min)
    x = np.round(x*(n-1))
    x = x/n
    x = x*(max-min) + min
    return x

# print(quantize(x, 8))
print(A_Law(x))

y = A_Law(x)
x = np.linspace(0, 1000, 1000)

plt.plot(x, y)
plt.show()
