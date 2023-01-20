import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)                                                        # Decimal Places Shown by the Numbers in the Array

# Function that Computes the Derivative of the Cost Function in Respect of the Parameters w and b:
def DerivativeCF(X, y, w, b):
    m = len(X)
    n = len(X[0])
    e = math.e
    dJdw = np.zeros(n)
    dJdb = 0
    for i in range(m):
        z = np.dot(w, X[i, :]) + b
        f_wb = (1) / (1 + e**(-z))
        for j in range(n):
            dJdw[j] = dJdw[j] + (f_wb - y[i]) * (X[i, j])
        dJdb = dJdb + (f_wb - y[i])
    dJdw = (dJdw) / (m)
    dJdb = (dJdb) / (m)
    return dJdw, dJdb

# Function that Computes the Value of the Logistic Cost Function:
def CostFunction(X, y, w, b):
    m = len(X)
    J = 0
    for i in range(m):
        z = np.dot(w, X[i, :]) + b
        f_wb = (1) / (1+(math.e)**(-z))
        J += -y[i] * math.log(f_wb) - (1 - y[i]) * math.log(1 - f_wb)
    J = (J) / (m)
    return (J)
    
# Training Set:
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
print(f"Matrix with the Input Features Values: \n{X_train}")
print(f"Vector with the Output Targets Values: \n{y_train}")

# Dimensions of the Training Set:
m = len(X_train)
n = len(X_train[0, :])
print(f"Dimension of the Input Features Matrix: \n{[m, n]}")

# Plotting the Training Set:
fig,ax = plt.subplots(1, 1, figsize=(5,5))
ax.axis([0, max(X_train[:,0]+0.5), 0, max(X_train[:,1])+0.5])
ax.set_xlabel('$x_0$', fontsize=12)
ax.set_ylabel('$x_1$')
plt.title('Training Set Plot')
for i in range(m):
    if y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='o', c='b')
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='x', c='r')
plt.show()

# Initial Guess for the Parameters:
w = np.zeros(n)
b = 0

# Defining a Value for the Learning Rate:
alpha = 0.1

# Applying the Gradient Descent Method for Logistic Regression:
iterations = 2000
J_history = []
for i in range(iterations):
    dJdw, dJdb = DerivativeCF(X_train, y_train, w, b)
    w = w - alpha * dJdw
    b = b - alpha * dJdb
    J = CostFunction(X_train, y_train, w, b)
    J_history.append(J)
print(f"Final Value of the Parameter w: \n{w}")
print(f"Final Value of the Parameter b: \n{b:.2f}")
print(f"Minimized Value of the Cost Function: \n{J:.2e}")

# Plotting the Learning Curve:
curve_domain = np.linspace(1, iterations, iterations)
plt.scatter(curve_domain, J_history, c='b')
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.show()

# Plotting the Predictions of the Model: 0.5 being the boundary between the values
f_wb_results = []
for i in range(m):
    z = np.dot(w, X_train[i, :]) + b
    f_wb = (1) / (1 + (math.e)**(-z))
    f_wb_results.append(f_wb)
    if f_wb >= 0.5:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='x', c='r')
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='o', c='b')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Predictions')
plt.show()
print(f"Predictions Using the Logistical Regression Model: \n{f_wb_results}")