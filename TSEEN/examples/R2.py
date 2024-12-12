import numpy as np
import matplotlib.pyplot as plt

def r2_score(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - y_true_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2

def pearson_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

    correlation = numerator / denominator
    
    return correlation

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# 从.npy文件中加载数据
y_true = (np.load("work_dirs/custom_exp/saved/trues.npy"))*1e-9  # 实际值
y_pred = (np.load("work_dirs/custom_exp/saved/preds.npy"))*1e-9 # 预测值

# 计算 R2
r2 = r2_score(y_true, y_pred)
pc = pearson_correlation(y_true, y_pred)
mse_value = mse(y_true, y_pred)
mae_value = mae(y_true, y_pred)
rmse_value = rmse(y_true, y_pred)

print("MSE:", mse_value)
print("MAE:", mae_value)
print("RMSE:", rmse_value)
print("R-squared score:", r2)
print("pearson_correlation:", pc)

