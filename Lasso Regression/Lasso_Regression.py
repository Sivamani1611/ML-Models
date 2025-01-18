import numpy as np
import matplotlib.pyplot as plt

def lasso_regression(X, y, alpha=0.1, learning_rate=0.01, max_iter=1000):

    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []

    for i in range(max_iter):

        y_pred = np.dot(X, weights) + bias

        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2) + alpha * np.sum(np.abs(weights))
        losses.append(loss)

        dw = (1/m) * np.dot(X.T, (y_pred - y)) + alpha * np.sign(weights)
        db = (1/m) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return weights, bias, losses

np.random.seed(42)
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = 3*X[:, 0] - 2*X[:, 1] + 1*X[:, 2] + 4 + np.random.randn(100) * 0.5  # Linear relation with some noise

alpha = 0.1
weights, bias, losses = lasso_regression(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)

print(f"Trained weights: {weights}")
print(f"Trained bias: {bias}")

plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss", color="red")
plt.title("Loss Function Convergence for Lasso Regression")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("Lasso_Regression_Loss_Function.png")
plt.show()


y_pred = np.dot(X, weights) + bias

plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred, color="blue", label="Predictions")
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", label="Ideal Predictions")
plt.title("Lasso Regression Predictions vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)
plt.legend()
plt.savefig("Lasso_Regression.png")
plt.show()
