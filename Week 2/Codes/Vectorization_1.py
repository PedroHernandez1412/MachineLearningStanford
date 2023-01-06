import numpy as np

# Declaring features and parameters:
x = np.array([1.0, 2.5, -3.3])
w = np.array ([10, 20, 30])
b = 4

# Manually Inputing the Features:
f_wb = x[0]*w[0]+x[1]*w[1]+x[2]*w[2]+b
print(f"Model Result Using Inputing Manually: {f_wb}")

# Summation Method:
f_wb = 0
n = len(x)
for j in range(n):
    f_wb = f_wb+w[j]*x[j]
f_wb = f_wb+b
print(f"Model Result Using Summation: {f_wb}")

# Using Vectorization:
f_wb = np.dot(x, w)+b
print(f"Model Result Using Vectorization: {f_wb}")