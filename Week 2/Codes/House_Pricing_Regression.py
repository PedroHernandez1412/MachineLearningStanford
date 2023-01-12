import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(precision=2)                                                        # Decimal Places Shown by the Numbers in the Array

# Function That Applies Z-Score Normalization as a Method of Feature Scaling:
def ZScoreNormalization(X):
    x_mean_results = []                                                                 # List to Alocate the Mean Value for Each Feature
    x_sd_results = []                                                                   # List to Alocate the Standard Deviation for Each Feature
    m = len(X)
    n = len(X[0])
    X_norm = np.zeros([m, n])                                                           # Matrix Filled with Zeros to Alocate the Values of the Features Scaled
    for j in range(n):
        x_mean = X[:, j].mean()
        x_sd = X[:, j].std()
        for i in range(m):
            X_norm[i, j] = (X[i, j] - x_mean) / (x_sd)
        x_mean_results.append(x_mean)
        x_sd_results.append(x_sd)
    return X_norm, x_mean_results, x_sd_results

# Function to Scale the Features Using the Maximum Value:
def FSMax(X):
    x_max_results = []                                                                  # List to Alocate the Maximum Value for Each Feature
    m = len(X)
    n = len(X[0])
    X_norm = np.zeros([m, n])                                                           # Matrix Filled with Zeros to Alocate the Values of the Features Scaled
    for j in range(n):
        x_max = X[:, j].max()
        for i in range(m):
            X_norm[i, j] = (X[i, j]) / (x_max)
        x_max_results.append(x_max)
    return X_norm, x_max_results

# Function To Scale the Features Using the Mean Method:
def FSMean(X):
    x_mean_results, x_max_results, x_min_results = [], [], []
    m = len(X)
    n = len(X[0])
    X_norm = np.zeros([m, n])
    for j in range(n):
        x_mean = X[:, j].mean()
        x_max = X[:, j].max()
        x_min = X[:, j].min()
        for i in range(m):
            X_norm[i, j] = (X[i, j] - x_mean) / (x_max - x_min)
        x_mean_results.append(x_mean)
        x_max_results.append(x_max)
        x_min_results.append(x_min)
    return X_norm, x_mean_results, x_max_results, x_min_results

# Function that Computes the Value of the Cost Function:
def CostFunction(X, y, w, b):
    m = len(X)
    Sum_Error = 0
    for i in range(m):
        f_wb = np.dot(w, X[i, :]) + b
        Error_i = (f_wb - y[i])**2
        Sum_Error = Sum_Error + Error_i
    J = ((1) / (2 * m)) * (Sum_Error)
    return J

# Function that Computes the Value of the Derivative of the Cost Function:
def CFDerivative(X, y, w, b):
    m = len(X)
    n = len(X[0])
    dJdw = np.zeros(n,)
    dJdb = 0
    for i in range(m):
        f_wb = np.dot(w, X[i, :]) + b
        for j in range(n):
            dJdw[j] = dJdw[j] + (f_wb - y[i]) * X[i, j]
        dJdb = dJdb + (f_wb - y[i])
    dJdw = ((1) / (m)) * (dJdw)
    dJdb = ((1) / (m)) * (dJdb)
    return dJdw, dJdb

# Read Dataset:
dataset_path = r'F:\Pedro\2023\Courses\Supervised Machine Learning Regression and Classification\Week 2\Codes\House_Pricing_Data.xlsx'
house_price_df = pd.read_excel(dataset_path)
print(f"Original dataset: \n{house_price_df.head(10)}")

# Training Data:
X_train = house_price_df[['size(sqft)', 'bedrooms', 'floors', 'age']].to_numpy()        # Input Features Matrice
y_train = house_price_df['price'].to_numpy()                                            # Output Target Vector
y_train = y_train.ravel()                                                               # Output Target as an One Dimensional Array
print(f"Matrix with The Input Features Value: \n{X_train[0:10:1, :]}")
print(f"Vector Containing the Output Targeets: \n{y_train[0:10:1]}")

# Dimensions:
m = len(X_train)                                                                        # Size of the Training Set
n = len(X_train[0])                                                                     # Number of Features
print(f"Dimensions of The Features Matrix: \n{m, n}")

# Applying Z-Score Normalization:
X_norm_Z, x_mean, x_sd = ZScoreNormalization(X_train)
print(f"Matrix with The Features Scaled Using Z-Score Normalization: \n{X_norm_Z[0:10:1, :]}")
print(f"Vector Containing the Mean of the Features: \n{x_mean}")
print(f"Vector Containing the Standard Deviation of the Features: \n{x_sd}")

# Feature Scaling dividing the Fatures by its maximum:
X_norm_max, x_max = FSMax(X_train)
print(f"Matrix with The Features Scaled Based on Their Maximum Value: \n{X_norm_max[0:10:1, :]}")
print(f"Vector Contatining The Maximum of the Features: \n{x_max}")

# Feature Scaling using the mean and the extreme values:
X_norm_mean, x_mean, x_max, x_min = FSMean(X_train)
print(f"Matrix with Features Scaled Using the Mean, Max and Min: \n{X_norm_mean[0:10:1, :]}")
print(f"Vector Contatining The Minimum of the Features: \n{x_min}")

# Plotting the Relation Between the Features Scaled:
fig, ax = plt.subplots(1,3, figsize=(12, 3))
plot_titles = ['Z-Score Normalization', 'Feature Scaling - Maximum', 'Feature Scaling - Mean']
X_norm_list = [X_norm_Z, X_norm_max, X_norm_mean]
for i in range(3):
    ax[i].scatter(X_norm_list[i][:, 0], X_norm_list[i][:, 3], c='orange')
    ax[i].set_title(plot_titles[i])
    ax[i].set_xlabel('Size')
    ax[i].set_ylabel('Age')
plt.show()

# Initial Guess of The Parameters:
w = np.array([1, 1, 1, 1])
b = 1
print(f"Initial Guess of the Parameter w: \n{w}")
print(f"Initial Guess of the Parameter b: \n{b}")

# Initial Value of the Cost Function:
J = CostFunction(X_train, y_train, w, b)
print(f"Initial Value of The Cost Function: \n{J:.2e}")

# Value of The Derivatives:
dJdw, dJdb = CFDerivative(X_train, y_train, w, b)
print(f"Value of the Derivatives of the Cost Function: \n{dJdw, dJdb}")

# Choosing the Learning Rate:
alpha = 1e-7

# Applying the Gradient Descent Method:
J_history = []
interactions = 15
for i in range(interactions):
    dJdw, dJdb = CFDerivative(X_train, y_train, w, b)
    w = w - alpha * dJdw
    b = b - alpha * dJdb
    J = CostFunction(X_train, y_train, w, b)
    J_history.append(J)
print(f"Final Value of the Parameter w: \n{w}")
print(f"Final Value of the Parameter b: \n{b:.2f}")
print(f"Final Value of the Cost Function: \n{J:.2e}")

# Learning Curve:
x_domain = np.linspace(1, interactions, interactions)
plt.title('Learning Curve')
plt.xlabel('Interactions')
plt.ylabel('Cost Function (J)')
plt.scatter(x_domain, J_history, c='blue')
plt.show()

# Output Target vs Prediction:
f_wb_prediction_list = []
for i in range(m):
    f_wb = np.dot(w, X_train[i, :]) + b
    f_wb_prediction_list.append(f_wb)
line = np.linspace(1, 900, 50)
plt.scatter(y_train, f_wb_prediction_list, c='r')
plt.scatter(line, line, c='blue', s=30)
plt.title('Target Outputs vs Prediction')
plt.xlabel('Target Values')
plt.ylabel('Prediction')
plt.show()