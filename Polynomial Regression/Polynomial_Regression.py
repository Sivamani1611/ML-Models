import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def initialize_parameters(degree):
    beta = np.zeros(degree + 1)
    return beta

def create_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = X.flatten() ** i
    return X_poly

def predict(X, beta, degree):
    X_poly = create_polynomial_features(X, degree)
    return np.dot(X_poly, beta)

def compute_cost(y, y_pred, beta, lambda_reg=0.01):
    n = len(y)
    cost = (1 / (2 * n)) * np.sum((y - y_pred)**2)
    reg_term = (lambda_reg / (2 * n)) * np.sum(beta[1:] ** 2)
    return cost + reg_term

def gradient_descent(X, y, beta, learning_rate, degree, lambda_reg=0.01):
    n = len(y)
    X_poly = create_polynomial_features(X, degree)
    y_pred = np.dot(X_poly, beta)
    gradients = -(1 / n) * np.dot(X_poly.T, (y - y_pred))
    gradients[1:] += (lambda_reg / n) * beta[1:]
    beta -= learning_rate * gradients
    return beta

def polynomial_regression(X, y, degree, learning_rate=0.001, epochs=1000, lambda_reg=0.01):

    beta = initialize_parameters(degree)

    for i in range(epochs):
        beta = gradient_descent(X, y, beta, learning_rate, degree, lambda_reg)
        y_pred = predict(X, beta, degree)
        cost = compute_cost(y, y_pred, beta, lambda_reg)

        if np.isnan(cost) or np.any(np.isnan(beta)):
            print(f"Terminating: NaN detected at epoch {i}. Check learning rate or data.")
            return beta

    return beta

def plot_results(X, y, beta, degree):
    plt.scatter(X, y, color='blue', label='Data Points')
    x_values = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_values = predict(x_values, beta, degree)
    plt.plot(x_values, y_values, color='red', label=f'Polynomial Fit (Degree {degree})')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'Polynomial_Regression_Degree {degree}_plot.png')
    plt.show()

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = X.flatten().reshape(-1, 1)

X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

learning_rate = 0.001
epochs = 1000
lambda_reg = 0.01
for degree in range(1, 6):
    beta = polynomial_regression(X, y, degree, learning_rate, epochs, lambda_reg)
    if np.any(np.isnan(beta)):
        print(f"Aborted fitting for degree {degree} due to NaN values.")
        break
    print(f"Degree {degree}: Coefficients: {beta}")
    plot_results(X, y, beta, degree)
