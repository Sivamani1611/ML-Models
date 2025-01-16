import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def initialize_parameters():
    # Initial coefficients of parameters
    beta_0 = 0
    beta_1 = 1
    return beta_0,beta_1

def predict(beta_0,beta_1, X):
    # Computing predictions for the input X
    return beta_0 + beta_1 * X

def compute_cost(y, y_pred):
    # Calculate Mean Squared Error
    n = len(y)
    return (1/n) * np.sum((y - y_pred) ** 2)

def gradient_descent(X, y, beta_0, beta_1 ,learning_rate):
    # Applying one step of gradient descent and updating the coefficients
    n = len(y)
    y_pred = predict(beta_0, beta_1, X)

    # gradient values
    gd_beta_0 = -(2 / n) * np.sum(y - y_pred)
    gd_beta_1 = -(2 / n) * np.sum((y - y_pred) * X)

    # updating the coefficients
    beta_0 = beta_0 - learning_rate * gd_beta_0
    beta_1 = beta_1 - learning_rate * gd_beta_1

    return beta_0, beta_1

def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    beta_0, beta_1 = initialize_parameters()

    for i in range(epochs):
        beta_0, beta_1 = gradient_descent(X, y, beta_0, beta_1, learning_rate)

        if i % 100 == 0:
            y_pred = predict(beta_0, beta_1, X)
            cost = compute_cost(y, y_pred)
            print(f"Epoch {i}, Cost: {cost:.4f}, beta_0: {beta_0:.4f}, beta_1: {beta_1:.4f}")

    return beta_0, beta_1

def automatic_hyperparameter_tuning(X, y):
    learning_rate = 0.1
    epochs = 1000
    prev_cost = float('inf')

    for _ in range(5):
        beta_0, beta_1 = linear_regression(X, y, learning_rate, epochs)
        y_pred = predict(beta_0, beta_1, X)
        cost = compute_cost(y, y_pred)

        if cost < prev_cost:
            prev_cost = cost
            break
        else:
            learning_rate = learning_rate / 2
    return learning_rate, beta_0, beta_1

def plot_results(X, y, beta_0, beta_1):
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, predict(beta_0, beta_1, X), color='red', label='Regression Line')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    #plt.savefig('linear_regression_plot.png')
    plt.show()


X, y = make_regression(n_samples = 100, n_features = 1, noise = 10, random_state = 42)
X = X.flatten()

print("Tuning of Hyperparameters")
learning_rate, beta_0, beta_1 = automatic_hyperparameter_tuning(X, y)

print(f"\nOptimized Learning Rate: {learning_rate:.4f}")
print(f"Trained Coefficients:\n beta_0 (Intercept) = {beta_0:.4f}\n beta_1 (Slope) = {beta_1:.4f}")

plot_results(X, y, beta_0, beta_1)
