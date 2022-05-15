from cProfile import label
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

y = np.array([0],dtype=np.int32)

min = np.iinfo(np.int32).min
max = np.iinfo(np.int32).max
fmin = -1
fmax = 1

# # Konwersja na int
# y = (y - fmin) / (fmax - fmin)
# y = y * (max - min) + min
# print(y)
# y = np.round(y).astype(np.int16)

# konwersja na float
y = y.astype(np.float32)
y = (y - min) / (max - min)
print(y)
y = y * (fmax - fmin) + fmin