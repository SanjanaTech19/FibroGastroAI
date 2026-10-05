import numpy as np
import matplotlib.pyplot as plt

# 1. Input Data
X = np.array([1.2, 2.5, 3.8, 5.0, 7.2])
Y = np.array([10.1, 11.2, 12.5, 13.8, 15.0])

# 2. Fit Linear Regression using OLS Normal Equation: theta = (X^T * X)^(-1) * X^T * Y
X_b = np.c_[np.ones(len(X)), X]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y

# 3. Predict for new data points
X_new = np.array([6.0, 8.5])
Y_pred = np.c_[np.ones(len(X_new)), X_new] @ theta

# 4. Plot Regression Line & Predictions
plt.scatter(X, Y, color='blue', label='Clinical Data')
plt.plot(X_new, Y_pred, 'r--', label=f'Fit: Y = {theta[1]:.2f}X + {theta[0]:.2f}')
plt.xlabel('Bilirubin (mg/dl)')
plt.ylabel('Prothrombin Time (s)')
plt.legend()
plt.title('Shortened Linear Regression Prediction')
plt.show()

print(f"Intercept: {theta[0]:.2f}, Slope: {theta[1]:.2f}")
print(f"Predictions for X={X_new}: {Y_pred.round(2)}")
