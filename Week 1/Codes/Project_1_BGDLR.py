#Batch Gradient Descent Linear Regression:

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_path = r'F:\Pedro\2023\Courses\Supervised Machine Learning Regression and Classification\Week 1\Codes\Project_1_Car_Data.xlsx'

def CostFunction(x_train, y_train, w, b):
    m = len(x_train)
    Error = 0
    for i in range(m):
        f_wb = w*x_train[i]+b
        Error_i = (f_wb-y_train[i])**2
        Error = Error+Error_i
    J = ((1)/(2*m))*(Error)
    return J

def CalculateDerivative(x_train, y_train, w, b):
    dJdw = 0
    dJdb = 0
    for i in range(m):
        f_wb = w*x_train[i]+b
        dJdw_i = (f_wb-y_train[i])*x_train[i]
        dJdw = dJdw+dJdw_i
        dJdb_i = f_wb-y_train[i]
        dJdb = dJdb+dJdb_i
    return dJdw, dJdb

# Reading the dataset:
car_dataset = pd.read_excel(dataset_path)
dataframe = pd.DataFrame(car_dataset)
filter_car = dataframe[dataframe['Car_Name'].str.contains('corolla altis', na=False)]
print(filter_car)

# Training Set:
x_train = filter_car[['Kms_Driven']].to_numpy().reshape(-1)
y_train = filter_car[['Selling_Price']].to_numpy().reshape(-1)

# Scaling the Features and Outputs of The Training Set:
x_max = np.max(x_train)
y_max = np.max(y_train)
x_scaled = (x_train)/(x_max)
y_scaled = (y_train)/(y_max)

# Number of training sets:
m = len(x_scaled)
print(f"Quantity of Samples in the Dataset: {m}")

# Parameters Value Initial Guess:
w = 0
b = 0
alpha = 1e-2

# Gradient Descent Method:
J = CostFunction(x_scaled, y_scaled, w, b)
print(f"Initial Value of The Cost Function: {J}")

interactions = 10000
for i in range(interactions):
    dJdw, dJdb = CalculateDerivative(x_scaled, y_scaled, w, b)
    temp_w = w-alpha*dJdw
    temp_b = b-alpha*dJdb
    J = CostFunction(x_scaled, y_scaled, temp_w, temp_b)
    w = temp_w
    b = temp_b
    if i% math.ceil(interactions/100) == 0:
        print(f"Iteration {i:4}: Cost {J:0.2e}",
                  f"dJdw: {dJdw: 0.3e}, dJdb: {dJdb: 0.3e}",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

print(f"Parameter w Found by Gradient Descent: {w: 0.3e}")
print(f"Parameter b Found by Gradient Descent: {b: 0.3e}")

# Plotting Linear Regression:
x_domain = np.linspace(0, max(x_scaled), num=100)

# Reescaling the Model:
f_wb_model = w*x_domain+b

# Ploting the dataset and the linear regression:
plt.title('Car Price')
plt.xlabel('Total Distance in KM')
plt.ylabel('Price in 1000s of dollars')
plt.scatter(x_scaled, y_scaled, marker='x', c='r')
plt.scatter(x_domain, f_wb_model, s=3)
plt.show()