import numpy as np
import matplotlib.pyplot as plt
import sys

# Test strumieniowa
t1 = np.array([1,1,1,1,2,1,1,1,1,2,1,1,1,1])
t9 = np.array([1,1,1,1,1,2,2,2,3,4,5,6,6,6,6,1])
t2 = np.array([1,2,3,1,2,3,1,2,3])
t3 = np.array([5,1,5,1,5,5,1,1,5,5,1,1,5])
t4 = np.array([-1,-1,-1,-5,-5,-3,-4,-2,1,2,2,1])
t5 = np.zeros((1,520))
t6 = np.arange(0,521,1)

# Test 4tree & strumieniowa
t7 = np.eye(7)
t8 = np.dstack([np.eye(7),np.eye(7),np.eye(7)])

print(t2)

def encode_rle(arr):
    out = arr.flatten()
    counter = 0
    symbols = []
    # result = np.array([]).astype(np.uint8)
    result = np.array([])
    for i in range(len(out)):
        if out[i] == out[i - 1]:
            counter += 1
        else:
            symbols.append(counter)
            symbols.append(out[i - 1])
            counter = 1
    symbols.append(counter)
    symbols.append(out[-1])
    result = np.array(symbols)
    return result

def decode_rle(arr):
    decoded = np.array([])
    for i in range(0, len(arr), 2):
        # decoded = np.append(decoded, np.ones(arr[i], dtype=np.uint8) * arr[i + 1])
        decoded = np.append(decoded, np.ones(arr[i]) * arr[i + 1])
    return decoded

encode_rle(t9)