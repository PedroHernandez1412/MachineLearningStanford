import numpy as np
import matplotlib.pyplot as plt

# Cost Function:
def CostFunction(x, y, w, b, m):
    Error = 0
    for i in range(m):
        f_wb = w*x[i]+b
        Error_i = (f_wb-y[i])**2
        Error = Error+Error_i
    J = ((1)/(2*m))*(Error)
    return J

# Training set:
x_train = np.array([1, 2, 3])
y_train = np.array([1, 2, 3])
print(f"Input Features: {x_train}")
print(f"Output Targets: {y_train}")

# Parameters:
w = -0.5
b = 0

# Size of the Training Sample:
m = len(x_train)
print(f"Size of the Training Set: {m}")

# Plot the Tendence of The Linear Regression/Model Function:
x_domain = np.linspace(0, 3, num=30)
f_wb_model = w*x_domain+b
plt.scatter(x_train, y_train, marker='x', c='r')
plt.scatter(x_domain, f_wb_model, s=0.75)
plt.title('f(w)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Calculating multiple value for Cost Functions:
J_list = []
w_list = []
while w < 3:
    w_list.append(w)
    J = CostFunction(x_train, y_train, w, b, m)
    J_list.append(J)
    w += 0.5
print(f"Parameters Values choosed: {w_list}")
print(f"Cost Function Values: {J_list}")

# Ploting the Correlation Graphic Between Parameter W and Cost Function:
plt.scatter(w_list, J_list, c='r')
plt.title('J(w)')
plt.xlabel('w')
plt.ylabel('J')
plt.show()