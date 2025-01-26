import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

def gaussian_pdf(x, mu, cov):
    d = len(x)
    diff = x - mu
    return (1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))) * np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)

def gmm_em(data, K, max_iters=100, tol=1e-4):
    N, d = data.shape
    responsibilities = np.zeros((N, K))
    
    np.random.seed(42)
    mu = np.random.rand(K, d)
    cov = np.array([np.eye(d)] * K)
    pi = np.ones(K) / K
    
    for iteration in range(max_iters):
        for i in range(N):
            denom = 0
            for k in range(K):
                denom += pi[k] * gaussian_pdf(data[i], mu[k], cov[k])
            for k in range(K):
                responsibilities[i, k] = (pi[k] * gaussian_pdf(data[i], mu[k], cov[k])) / denom
        
        for k in range(K):
            Nk = np.sum(responsibilities[:, k])
            pi[k] = Nk / N
            mu[k] = np.sum(responsibilities[:, k][:, np.newaxis] * data, axis=0) / Nk
            diff = data - mu[k]
            cov[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk
        
        log_likelihood = 0
        for i in range(N):
            temp = 0
            for k in range(K):
                temp += pi[k] * gaussian_pdf(data[i], mu[k], cov[k])
            log_likelihood += np.log(temp)
        
        if iteration > 0 and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
    
    return mu, cov, pi, responsibilities, log_likelihood

def calculate_aic(log_likelihood, K, d):
    num_params = K * d + K * d * d + K  # Number of parameters (means + covariances + mixing coefficients)
    return 2 * num_params - 2 * log_likelihood

K_values = [2, 3, 4, 5]
max_iters_values = [100, 200]
tol_values = [1e-3, 1e-4]

best_aic = np.inf
best_model = None
best_K = None
best_max_iters = None
best_tol = None

for K in K_values:
    for max_iters in max_iters_values:
        for tol in tol_values:
            mu, cov, pi, responsibilities, log_likelihood = gmm_em(data, K, max_iters=max_iters, tol=tol)
            aic = calculate_aic(log_likelihood, K, data.shape[1])
            print(f"Evaluating K={K}, max_iters={max_iters}, tol={tol}, AIC={aic}")
            if aic < best_aic:
                best_aic = aic
                best_model = (mu, cov, pi, responsibilities)
                best_K = K
                best_max_iters = max_iters
                best_tol = tol

print(f"Best Model: K={best_K}, max_iters={best_max_iters}, tol={best_tol}, AIC={best_aic}")

mu, cov, pi, responsibilities = best_model

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(yy.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        Z[i, j] = np.argmax([pi[k] * gaussian_pdf(point, mu[k], cov[k]) for k in range(best_K)])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c=np.argmax(responsibilities, axis=1), cmap='viridis', s=10)
plt.scatter(mu[:, 0], mu[:, 1], c='red', marker='x', s=200, label="Centroids")
plt.title(f"Gaussian Mixture Model (GMM) - Best Model (K={best_K})")
plt.legend()
plt.show()
