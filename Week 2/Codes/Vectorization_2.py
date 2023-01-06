import numpy as np
import time 

# Declaring features and parameters:
x = np.linspace(1, 10, 10000000)
w = np.linspace(6, 15, 10000000)
b = 1

# Using Summation to Calculate the Multiple Linear Regression:
f_wb = 0
n = len(x)
start = time.time()
if len(x) == len(w):
    for j in range(n):
        f_wb = f_wb+x[j]*w[j]
    f_wb = f_wb+b
    end = time.time()
    print(f"Model Result Using Summation: {f_wb}")
    duration = time.strftime('%H:%M:%S', time.gmtime(end-start))
    print(f"Time that This Method Took: {duration}")
else:
    print('The Vectors Have Different Amount of Values!')

# Using Vectorization:
if len(x) == len(w):
    start = time.time()
    f_wb = np.dot(x, w)+b
    end = time.time()
    print(f"Model Result Using Vectorization: {f_wb}")
    duration = time.strftime('%H:%M:%S', time.gmtime(end-start))
    print(f"Time that Vectorization Took: {duration}")
else:
    print('The Vectors Have Different Amount of Values!')