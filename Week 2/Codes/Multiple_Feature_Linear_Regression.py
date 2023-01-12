import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)    # Number of Decimal Numbers Show by the Numbers in the Array

# Compute Cost Function:
def CostFunction(X, y, w, b):
    m = len(X)
    Sum_Error = 0
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        Sum_Error_i = (f_wb - y[i])**2
        Sum_Error = Sum_Error + Sum_Error_i
    J = ((1) / (2 * m)) * Sum_Error
    return J

# Compute the Derivative of The Cost Function in Respect of the Parameters:
def DerivativeCF(X, y, w, b):
    m = len(X)
    n = len(X[0])
    dJdw = np.zeros((n,))
    dJdb = 0
    for i in range(m):
        f_wb = np.dot(w, X[i, :]) + b
        for j in range(n):
            dJdw[j] = (f_wb - y[i]) * X[i, j] + dJdw[j]
        dJdb = (f_wb - y[i]) + dJdb
    dJdw = dJdw / m
    dJdb = dJdb / m
    return dJdw, dJdb

# Make a Predicition:


# Training Data:
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Size of the Training Set for Each Feature:
m = len(X_train)

# Number of Features:
n = len(X_train[0])

# Parameters Initial Guess:
w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
b = 785.181136799408

# Learning Rate:
alpha  = 1e-7

# Applying Gradient Descent Method
J_history = []
interactions = 10
for i in range(interactions):
    dJdw, dJdb = DerivativeCF(X_train, y_train, w, b)
    w = w - alpha * dJdw
    b = b - alpha * dJdb
    J = CostFunction(X_train, y_train, w, b)
    J_history.append(J)
print(f"Final Value of the Parameter w: \n{w}")
print(f"Final Value of the Parameter b: \n{b:.2f}")
print(f"Final Value of the Cost Function: \n{J_history[interactions-1]}")

# Final Value of The Predctions:
f_wb_history = []
for i in range(m):
    f_wb = np.dot(w, X_train[i, :]) + b
    f_wb_history.append(f_wb)
print(f"Original Values of the Output Targets: \n{y_train}")
print(f"Final Results of the Predictions: \n{f_wb_history}")

# Prediction:
x_prediction = np.array([1200, 3, 1, 40])
f_wb = np.dot(w, x_prediction) + b
print(f"Value of the Prediciton: \n{f_wb:.0f}")

# Learning Curve:
x_domain = np.linspace(1, interactions, interactions)
plt.scatter(x_domain, J_history)
plt.show()