import numpy as np
import matplotlib.pyplot as plt

# Parameters:
w= -0.5         #Change
b=0             #Desconsidering 

# Training set:
x_train = np.array([1, 2, 3])
y_train = np.array([1, 2, 3])
print(f"Input Features: {x_train}")
print(f"Output Targets: {y_train}")

# Number of training examples:
m = len(x_train)
print(f"Size of the Training Set: {m}")

# Linear regression or Model Function:
x_domain = np.linspace(0, 3, num=30) # Evenly spaced numbers
f_wb = w*x_domain + b

# Plot with varying marker format and color:
plt.title('f(w)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_domain, f_wb, s=0.75)
plt.scatter(x_train, y_train, marker='x', c='r')
plt.show()

# Cost Function:
Error = 0
print('--------------------------------------------------')
for i in range(m):
    f_wb = w*x_train[i]+b
    print(f"Index: {i}")
    print(f"input: {x_train[i]}")
    print(f"Output: {y_train[i]}")
    print(f"Prediction: {f_wb}")
    Error_i = (f_wb-y_train[i])**2
    Error = Error+Error_i
    print(f"SDError: {Error_i}")
    print(f"TotalError: {Error}")
    print('--------------------------------------------------')

J = (1/(2*m))*(Error)
print(f"Cost Function Value: {J}")