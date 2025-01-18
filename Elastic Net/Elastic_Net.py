import numpy as np
import matplotlib.pyplot as plt

# cost function with both L1 (Lasso) and L2 (Ridge) regularization
def compute_cost(X, y, theta, lambda_1, lambda_2):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    
    l1_penalty = lambda_1 * np.sum(np.abs(theta[1:]))
    l2_penalty = (lambda_2 / 2) * np.sum(theta[1:] ** 2)
    
    cost += l1_penalty + l2_penalty
    return cost

def compute_gradient(X, y, theta, lambda_1, lambda_2):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    
    gradient = (1 / m) * X.T.dot(errors)

    l1_gradient = lambda_1 * np.sign(theta[1:])
    l2_gradient = lambda_2 * theta[1:]
    gradient[1:] += l1_gradient + l2_gradient
    return gradient

def gradient_descent(X, y, theta, learning_rate, iterations, lambda_1, lambda_2):
    cost_history = []
    
    for _ in range(iterations):
        gradient = compute_gradient(X, y, theta, lambda_1, lambda_2)
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta, lambda_1, lambda_2))
        
    return theta, cost_history

def grid_search(X, y, lambda_1_values, lambda_2_values, learning_rate, iterations):
    best_theta = None
    best_cost = float('inf')
    best_lambda_1 = 0
    best_lambda_2 = 0
    cost_histories = {}

    for lambda_1 in lambda_1_values:
        for lambda_2 in lambda_2_values:
            theta = np.zeros(X.shape[1])
            theta_optimal, cost_history = gradient_descent(X, y, theta, learning_rate, iterations, lambda_1, lambda_2)
            final_cost = cost_history[-1]
            
            cost_histories[(lambda_1, lambda_2)] = cost_history
            
            if final_cost < best_cost:
                best_cost = final_cost
                best_theta = theta_optimal
                best_lambda_1 = lambda_1
                best_lambda_2 = lambda_2
    
    return best_theta, best_lambda_1, best_lambda_2, cost_histories

def plot_cost_history(cost_histories, lambda_1_values, lambda_2_values):
    plt.figure(figsize=(10, 6))
    
    for lambda_1 in lambda_1_values:
        for lambda_2 in lambda_2_values:
            plt.plot(cost_histories[(lambda_1, lambda_2)], label=f"λ1={lambda_1}, λ2={lambda_2}")
    
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence for Different Hyperparameters')
    plt.legend()
    plt.savefig('Elastic_Net.png')
    plt.show()


np.random.seed(42)
X = np.random.rand(100, 3)
X = np.c_[np.ones(X.shape[0]), X]
y = X.dot(np.array([5, 3, 2, 1])) + np.random.randn(100) * 0.5

lambda_1_values = [0.01, 0.1, 1]
lambda_2_values = [0.01, 0.1, 1]
learning_rate = 0.01
iterations = 1000

best_theta, best_lambda_1, best_lambda_2, cost_histories = grid_search(X, y, lambda_1_values, lambda_2_values, learning_rate, iterations)

print("Optimal theta values:", best_theta)
print("Best λ1:", best_lambda_1)
print("Best λ2:", best_lambda_2)

plot_cost_history(cost_histories, lambda_1_values, lambda_2_values)
