# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
def bayer(n):
    if n == 1:
        return np.array([[0,2],[3,1]])
    M = np.array(bayer(n/2))
    return np.concatenate((np.concatenate(((((2*n))*M), (((2*n))*M)+2), axis=1), np.concatenate(((((2*n))*M)+3, (((2*n))*M)+1), axis=1)), axis=0)

def transform_bayer_to_preprocessed_matrix(bayer, n):
    return ((bayer+1)/((2*n)**2))-0.5

bayes2b = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
])

# %%
n = 1

M = np.array(bayer(n/2), "\n")

# m1con = np.concatenate((M, M+2), axis=1)
# m2con = np.concatenate((M+3, M+1), axis=1)
# print(m1con)
# print(m2con)

# print("bayes2b:\n", bayes2b, "\n")
# print("bayes2b2*n:\n",bayes2b/((2*n)**2), "\n")
# print("bayes2b2*n**2:\n",bayes2b/((n)**2), "\n")


# mcon2 = np.concatenate((np.concatenate(((((n))*M), (((n))*M)+2), axis=1), np.concatenate(((((n))*M)+3, (((n))*M)+1), axis=1)), axis=0)
mbay = bayer(n)
# print("mcon2*n:\n", mcon, "\n")
print("mcon2*n**2:\n", mcon2, "\n")
print("mcon2*n**2:\n", mbay, "\n")
print(transform_bayer_to_preprocessed_matrix(mcon2, n))
print(transform_bayer_to_preprocessed_matrix(mbay, n))


