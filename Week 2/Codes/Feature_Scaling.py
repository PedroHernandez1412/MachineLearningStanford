import numpy as np

def FeatureScalingMax(x):
    x_max = np.max(x)
    x_scaled = x / x_max
    return x_scaled

def FeatureScalingMean(x):
    x_mean = np.mean(x)
    x_max = np.amax(x)
    x_min = np.amin(x)
    x_scaled = (x - x_mean) / (x_max - x_min)
    return x_scaled

def FeatureScalingSD(x):
    x_mean = np.mean(x)
    x_sd = np.std(x)
    x_scaled = (x - x_mean) / (x_sd)
    return x_scaled

x = [np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([10, 11, 13])]
print(x)

scaled_list = []
for x_i in x:
    x_scaled = FeatureScalingSD(x_i)
    scaled_list.append(x_scaled)

print(scaled_list)