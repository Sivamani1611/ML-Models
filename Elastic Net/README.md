# Elastic Net Regression with Hyperparameter Tuning and Visualization

This repository implements an **Elastic Net Regression** model using mathematical formulas and showcases its application with hyperparameter tuning. The model is built from scratch, without using high-level libraries like `scikit-learn`. The script includes grid search for hyperparameter optimization and visualization of cost convergence.

## Overview

- **Elastic Net Regression**: A linear regression model with combined L1 (Lasso) and L2 (Ridge) regularization. It balances feature selection and coefficient shrinkage to prevent overfitting and handle multicollinearity.
- **Hyperparameter Tuning**: The script employs a grid search over a range of regularization parameters (λ₁ for L1 and λ₂ for L2 penalties).
- **Visualization**: The cost function's convergence is visualized for different combinations of hyperparameters, providing insight into model performance.

---

## Mathematical Formulas

### 1. Elastic Net Cost Function
The Elastic Net cost function combines the Mean Squared Error (MSE) and regularization terms:

J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \frac{\lambda_2}{2} \sum_{j=1}^{n} \theta_j^2

Where:
- \( m \): Number of data points.
- \( λ₁ \): L1 regularization parameter (Lasso).
- \( λ₂ \): L2 regularization parameter (Ridge).
- \( \theta_j \): Coefficients of the model (excluding \( \theta_0 \)).

---

### 2. Gradient for Elastic Net
The gradient for updating parameters during optimization includes contributions from both L1 and L2 regularization:

\[
\text{Gradient} = \frac{1}{m} X^T (h_\theta(X) - y) + λ₁ \cdot \text{sign}(\theta) + λ₂ \cdot \theta
\]

Where:
- \( \text{sign}(\theta) \): The sign of each parameter \( \theta_j \).

---

## Script Workflow

1. **Data Generation**:
   - A synthetic dataset with 3 features and Gaussian noise is created for regression tasks.

2. **Model Implementation**:
   - **Cost Function**: Computes the MSE and adds L1 and L2 penalties.
   - **Gradient Calculation**: Incorporates L1 and L2 regularization into the gradient.
   - **Gradient Descent**: Optimizes the model parameters by minimizing the cost function.

3. **Hyperparameter Tuning**:
   - A grid search is performed over a range of λ₁ and λ₂ values to find the best combination.

4. **Visualization**:
   - The script generates plots showing the cost function's convergence for different hyperparameter values.

---

## Parameters

| Parameter         | Description                                         | Default Value     |
|-------------------|-----------------------------------------------------|-------------------|
| `lambda_1_values` | List of L1 regularization strengths (λ₁)            | `[0.01, 0.1, 1]`  |
| `lambda_2_values` | List of L2 regularization strengths (λ₂)            | `[0.01, 0.1, 1]`  |
| `learning_rate`   | Learning rate for gradient descent                  | `0.01`            |
| `iterations`      | Number of iterations for gradient descent           | `1000`            |
| `X`               | Input feature matrix                                | Synthetic data    |
| `y`               | Target variable                                     | Synthetic data    |

---

## Visualization

### Cost Function Convergence
The script generates a plot showing how the cost function converges over iterations for different combinations of λ₁ and λ₂. This helps in analyzing the effect of regularization on the model's performance.

### Example Plot:
![Elastic Net Convergence](https://placeholder.url/elastic-net-convergence.png)

---

## Output
```
Optimal theta values: [4.8355205  2.63715634 2.05520892 1.63712866]
Best λ1: 0.01
Best λ2: 0.01
```
