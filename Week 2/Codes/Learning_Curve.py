import matplotlib.pyplot as plt
import numpy as np

# Compute the Value of the Cost Function:
def CostFunction(x, y, w, b):
    m = len(x)
    Sum_Error = 0
    for i in range(m):
        f_wb = w * x[i] + b
        Error_i = (f_wb - y[i])**2
        Sum_Error = Sum_Error + Error_i
    J = ((1) / (2 * m)) * Sum_Error
    return J

# Compute the Derivative of the Cost Function:
def DerivativeCF(x, y, w, b):
    m = len(x)
    Sum_dJdw = 0
    Sum_dJdb = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dJdw_i = (f_wb - y[i]) * x[i]
        dJdb_i = (f_wb - y[i])
        Sum_dJdw = Sum_dJdw + dJdw_i
        Sum_dJdb = Sum_dJdb + dJdb_i
    return Sum_dJdw, Sum_dJdb

# Training Set:
x_train = np.array([1, 2, 3])        # Input Features
y_train = np.array([1, 2, 3])        # Output Targets

# Parameters:
w = 2                               # Initial Guess of the Parameter
b = 0                               # Initial Guesss of the Parameter

# Linear Regression Model:
x_domain = np.linspace(1, 3, 30)    # Array to make a line
f_wb_plot = w * x_domain + b

# Plotting the model:
plt.scatter(x_train, y_train, marker='x', c='r')
plt.scatter(x_domain, f_wb_plot, s=1.5, c='orange')

# Training Set Size:
m = len(x_train)

# Learning Rate:
alpha = 1e-2

# Gradient Descent
interactions = 15
J_history = []
for i in range (interactions):
    dJdw, dJdb = DerivativeCF(x_train, y_train, w, b)
    w = w - alpha * dJdw
    b = b - alpha * dJdb
    J_history.append(CostFunction(x_train, y_train, w, b))

# Linear Regression Model:
x_domain = np.linspace(1, 3, 30)    # Array to make a line
f_wb_plot = w * x_domain + b

# Model Adjusted with Gradient Descent:
plt.scatter(x_domain, f_wb_plot, s=1.5, c='green')
plt.title('Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Learning Curve:
x_lc_domain = np.linspace(1, interactions, interactions)
plt.scatter(x_lc_domain, J_history)
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Cost Function (J)')
plt.show()