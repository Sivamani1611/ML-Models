import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(50), -1 * np.ones(50)])
    return X, y

def train_svm(X, y, lr=0.001, epochs=1000, C=10.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    losses = []

    for epoch in range(epochs):
        hinge_loss = 1 - y * (np.dot(X, w) + b)
        hinge_loss[hinge_loss < 0] = 0
        loss = 0.5 * np.dot(w, w) + C * np.mean(hinge_loss)
        losses.append(loss)

        for i in range(n):
            if hinge_loss[i] > 0:
                w -= lr * (w - C * y[i] * X[i])
                b -= lr * (-C * y[i])
            else:
                w -= lr * w
    return w, b, losses

def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.8, edgecolor='k')

    predictions = np.dot(X, w) + b
    margin_violations = (y * predictions) < 1
    plt.scatter(X[margin_violations, 0], X[margin_violations, 1], edgecolors='yellow', 
                facecolors='none', s=100, label="Margin Violations")

    x_min, x_max = plt.xlim()
    y_min = -(w[0] * x_min + b) / w[1]
    y_max = -(w[0] * x_max + b) / w[1]
    plt.plot([x_min, x_max], [y_min, y_max], 'k-', label="Decision Boundary")

    margin_pos = -(w[0] * x_min + b - 1) / w[1]
    margin_neg = -(w[0] * x_min + b + 1) / w[1]
    plt.plot([x_min, x_max], [margin_pos, margin_neg], 'g--', label="Margins")
    plt.plot([x_min, x_max], [margin_neg, margin_pos], 'g--')

    plt.legend()
    plt.title("SVM Decision Boundary with Margin Violations")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    plt.savefig('svm_decision_boundary.png')
    plt.close()

X, y = generate_data()
w, b, losses = train_svm(X, y, lr=0.001, epochs=1500, C=10.0)

plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('SVM_training_loss.png')
plt.show()
plt.close()

plot_decision_boundary(X, y, w, b)
