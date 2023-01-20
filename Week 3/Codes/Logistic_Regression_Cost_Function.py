import math
import matplotlib.pyplot as plt
import numpy as np

# Function that Computes the Value of the Logistic Cost Function:
def CostFunction(X, y, w, b):
    J = 0
    m = len(X)
    for i in range(m):
        z = np.dot(w, X[i, :]) + b
        f_wb = (1) / (1 + (math.e)**(-z))
        J += -y[i] * math.log(f_wb) - (1 - y[i]) * math.log(1 - f_wb)
    J = (J) / (m)
    return J        

# Training set:
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
print(f"Matrix with the Input Features Values: \n{X_train}")
print(f"Vector with the Output Targets: \n{y_train}")

# Dimensions of the Training Set:
m = len(X_train)
n = len(X_train[0, :])
print(f"Dimensions of the Input Features Matrix: \n{m, n}")

# Guessing the Regression Parameters:
w_tmp = np.array([1, 1])
b_tmp = -3
print(f"Initial Guess of the Parameter w: \n{w_tmp}")
print(f"Initial Guess of the Parameter b: \n{b_tmp}")

# Computing the Cost Function for the Logisit Regression Model:
J = CostFunction(X_train, y_train, w_tmp, b_tmp)
print(f"Value Computed for the Cost Function: \n{J:.2f}")

# Plotting the output labels:
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.axis([0, max(X_train[:, 0]) + 0.5, 0, max(X_train[:, 1]) + 0.5])
ax.set_xlabel('$x_0$', fontsize=12)
ax.set_ylabel('$x_1$', fontsize=12)
plt.title('Binary Logistic Problem')
for i in range(m):
    if y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='o', c='b')
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='x', c='r')
plt.show()

# Plotting the Decision Boundary:
db_aux = np.arange(0, m)
db = -b_tmp - db_aux
b_tmp_2 = -4
db_2 = -b_tmp_2 - db_aux
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.axis([0, max(X_train[:, 0]) + 0.5, 0, max(X_train[:, 1]) + 0.5])
ax.set_xlabel('$x_0$', fontsize=12)
ax.set_ylabel('$x_1$', fontsize=12)
plt.title('Decision Boundary')
plt.plot(db_aux, db, c='m', label='$b$ = -3')
plt.plot(db_aux, db_2, c='orange', label='$b$ = -4')
for i in range(m):
    if y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='o', c='b')
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], marker='x', c='r')
plt.legend()
plt.show()