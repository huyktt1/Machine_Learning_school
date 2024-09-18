# WEEEK 1:
1) CODE MACHINE LEARNING :
# problem 1:
import numpy as np


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Trọng số (theta):", theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print("Dự đoán:", y_predict)
------------------------------------
Trọng số (theta): [[4.22215108]
 [2.96846751]]
Dự đoán: [[ 4.22215108]
 [10.1590861 ]]

# problem 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/content/linear.csv')

X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)

X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
y_predict = X_b.dot(theta_best)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Data points")
plt.plot(X, y_predict, "r-", label="Prediction (Linear Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Linear Regression Fit (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# problem 3:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/linear.csv')

X = data['Diện tích'].values.reshape(-1, 1)
y = data['Giá'].values.reshape(-1, 1)

# Create polynomial features (X^2)
X_poly = np.c_[np.ones((X.shape[0], 1)), X, X**2]  # Adding bias (1), X, and X^2

# Compute theta (weights) using the normal equation
theta_best = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

# Predict the values using the parabolic model
y_predict = X_poly.dot(theta_best)

# Define the RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Calculate RMSE
loss_rmse = rmse(y, y_predict)
print(f"RMSE: {loss_rmse}")

# Plot the data points and the prediction curve (parabolic fit)
plt.figure(figsize=(8, 6))
plt.plot(X, y, "b.", label="Data points")
plt.plot(X, y_predict, "r-", label="Prediction (Parabolic Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Parabolic Regression Fit (RMSE: {loss_rmse:.4f})")
plt.legend()
plt.grid(True)
plt.show()
------------------------------------------------------------------------------------------------------------------------
