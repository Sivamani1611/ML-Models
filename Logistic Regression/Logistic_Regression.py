import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def initialize_parameters():
    beta_0 = 0
    beta_1 = 0
    return beta_0, beta_1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(beta_0, beta_1, X):
    z = beta_0 + np.dot(X, beta_1)
    return sigmoid(z)

def compute_cost(y, y_pred):
    n = len(y)
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cost = -(1/n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

def gradient_descent(X, y, beta_0, beta_1, learning_rate):
    n = len(y)
    y_pred = predict(beta_0, beta_1, X)

    gd_beta_0 = -(1/n) * np.sum(y - y_pred)
    gd_beta_1 = -(1/n) * np.dot((y - y_pred), X)
    
    beta_0 = beta_0 - learning_rate * gd_beta_0
    beta_1 = beta_1 - learning_rate * gd_beta_1

    return beta_0, beta_1

def logistic_regression(X, y, learning_rate = 0.01, epochs = 1000):
    beta_0, beta_1 = initialize_parameters()

    for i in range(epochs):
        beta_0, beta_1 = gradient_descent(X, y, beta_0, beta_1, learning_rate)

        if i % 100 == 0:
            y_pred = predict(beta_0, beta_1, X)
            cost = compute_cost(y, y_pred)
            print(f"Epoch {i}, Cost: {cost:.4f}, beta_0: {beta_0:.4f}, beta_1 : {beta_1:.4f}")

    return beta_0, beta_1

def automatic_hyperparameter_turning(X, y):
    learning_rate = 0.01
    epochs = 1000
    prev_cost = float('inf')

    for _ in range(5):
        beta_0, beta_1 = logistic_regression(X, y, learning_rate, epochs)
        y_pred = predict(beta_0, beta_1, X)
        cost = compute_cost(y, y_pred)

        if cost < prev_cost:
            prev_cost = cost
            break
        else:
            learning_rate = learning_rate / 2
    return learning_rate, beta_0, beta_1

def plot_results(X, y, beta_0, beta_1):
    plt.scatter(X, y, color='blue', label='Data Points')
    x_values = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_values = predict(beta_0, beta_1, x_values)
    plt.plot(x_values, y_values, color='red', label='Decision Boundary')

    plt.title('Logistic Regression')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    #plt.savefig('logistic_regression_plot.png')
    plt.show()

X, y = make_classification(n_samples = 100, n_features = 1, n_informative = 1, n_redundant = 0, n_clusters_per_class = 1, random_state = 42)
X = X.flatten()

print("Turning of Hyperparameters")
learning_rate , beta_0, beta_1 = automatic_hyperparameter_turning(X, y)

print(f"\nOptimizing Learning Rate: {learning_rate:.4f}")
print(f"Trained Coefficient:\n beta_0 (Intercept) = {beta_0:.4f}, beta_1 (Slope) = {beta_1:.4f}")

plot_results(X, y, beta_0, beta_1)
