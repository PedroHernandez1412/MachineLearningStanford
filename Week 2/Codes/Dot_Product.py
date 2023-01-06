import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 4])

def Dot_Product(a, b):
    dp = 0
    if len(a) == len(b):
        for j in range(len(a)):
            dp = a[j] * b[j] + dp
        return dp
    else:
        return print("Vectors have different shapes!")

dp = Dot_Product(a, b);      print(f"Result of the Dot Product Between Vectors: {dp}")