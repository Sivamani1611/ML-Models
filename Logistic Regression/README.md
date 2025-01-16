# Logistic Regression with Gradient Descent

This repository implements a simple **Logistic Regression** model from scratch using **Gradient Descent** for optimization. The goal is to predict a binary target variable (`y`) based on input features (`X`) by finding the optimal model parameters that minimize the cost function (Binary Cross-Entropy Loss).

## Overview

- **Logistic Regression**: A statistical method for binary classification problems. It models the relationship between a dependent variable (`y`) and one or more independent variables (`X`) using the sigmoid function.
- **Gradient Descent**: An iterative optimization algorithm used to minimize the cost function by adjusting model parameters (weights and bias).
- **Binary Cross-Entropy Loss**: A common cost function used for classification problems. It measures the difference between the true labels and the predicted probabilities.

## Mathematical Formulas

### 1. Logistic Regression Model
The logistic regression model assumes the relationship between the independent variables (`X`) and the dependent variable (`y`) can be described using the sigmoid function:

$$
z = \beta_0 + \beta_1 \cdot X
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- `ŷ` is the predicted probability of the positive class.
- `z` is the linear combination of inputs and weights.
- `σ(z)` is the sigmoid function.

### 2. Cost Function: Binary Cross-Entropy Loss
The cost function used in logistic regression is the Binary Cross-Entropy Loss:

$$
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]
$$

Where:
- `n` is the number of data points.
- `yᵢ` is the true label (0 or 1).
- `ŷᵢ` is the predicted probability for the positive class.

### 3. Gradient Descent Update Rule
The gradient descent algorithm updates the parameters \( \beta_0 \) and \( \beta_1 \) iteratively to minimize the cost function. The update rules are as follows:

$$
\beta_0 = \beta_0 - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

$$
\beta_1 = \beta_1 - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot X_i
$$

Where:
- `η` is the **learning rate**.
- `β₀` and `β₁` are the model parameters (bias and weight).
- `Xᵢ` is the value of the feature for the `i-th` data point.

## Script Workflow

1. **Data Generation**:
   - The script generates a synthetic binary classification dataset using `make_classification` from scikit-learn with 100 samples and 1 feature.

2. **Initialize Parameters**:
   - The model parameters \( \beta_0 \) (bias) and \( \beta_1 \) (weights) are initialized to zero.

3. **Prediction**:
   - The `predict` function calculates the predicted probabilities of the positive class using the sigmoid function:
     $$
     \hat{y} = \sigma(\beta_0 + \beta_1 \cdot X)
     $$

4. **Cost Function**:
   - The `compute_cost` function calculates the Binary Cross-Entropy Loss between the true labels `y` and the predicted probabilities `\hat{y}`.

5. **Gradient Descent**:
   - The `gradient_descent` function performs one step of gradient descent. It calculates the gradients of the cost function with respect to \( \beta_0 \) and \( \beta_1 \) and updates the parameters to reduce the cost.

6. **Model Training**:
   - The `logistic_regression` function runs gradient descent for a specified number of epochs (iterations). During each epoch, the model parameters are updated, and the cost is printed every 100 epochs to monitor convergence.

7. **Visualization**:
   - After training, the script plots the decision boundary and the original data points using `matplotlib`.

## Output

```bash
Turning of Hyperparameters
Epoch 0, Cost: 0.6909, beta_0: -0.0001, beta_1 : 0.0048
Epoch 100, Cost: 0.5139, beta_0: -0.0095, beta_1 : 0.4244
Epoch 200, Cost: 0.4070, beta_0: -0.0170, beta_1 : 0.7508
Epoch 300, Cost: 0.3381, beta_0: -0.0221, beta_1 : 1.0129
Epoch 400, Cost: 0.2909, beta_0: -0.0248, beta_1 : 1.2299
Epoch 500, Cost: 0.2569, beta_0: -0.0257, beta_1 : 1.4141
Epoch 600, Cost: 0.2314, beta_0: -0.0251, beta_1 : 1.5737
Epoch 700, Cost: 0.2117, beta_0: -0.0235, beta_1 : 1.7141
Epoch 800, Cost: 0.1960, beta_0: -0.0210, beta_1 : 1.8394
Epoch 900, Cost: 0.1832, beta_0: -0.0179, beta_1 : 1.9523

Optimizing Learning Rate: 0.0100
Trained Coefficient:
 beta_0 (Intercept) = -0.0145, beta_1 (Slope) = 2.0541
```

## Visualization
The plot shows the data points (blue and orange) and the decision boundary (red line) predicted by the logistic regression model.
