# Linear Regression with Gradient Descent

This repository implements a simple **Linear Regression** model from scratch using **Gradient Descent** for optimization. The goal is to predict a continuous target variable (`y`) based on a single feature (`X`) by finding the best-fit line using the least square error method.

## Overview

- **Linear Regression**: A statistical method for modeling the relationship between a dependent variable (`y`) and an independent variable (`X`).
- **Gradient Descent**: An iterative optimization algorithm used to minimize the cost function (Mean Squared Error, MSE in this case) by adjusting model parameters (intercept and slope).
- **Mean Squared Error (MSE)**: A common cost function used for regression problems. It measures the average squared difference between the predicted and actual values.

## Mathematical Formulas

### 1. Linear Regression Model
The linear regression model assumes the relationship between the independent variable (`X`) and dependent variable (`y`) can be described by a linear equation:

$$
y = \beta_0 + \beta_1 \cdot X
$$

Where:
- `y` is the predicted value.
- `β₀` is the intercept (bias term).
- `β₁` is the slope (coefficient for the feature).
- `X` is the input feature.

### 2. Cost Function: Mean Squared Error (MSE)
The cost function used in this implementation is the Mean Squared Error (MSE), which calculates the average squared difference between the true values (`y`) and the predicted values (`y_pred`):

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - y_{\text{pred}_i})^2
$$

Where:
Where:
- `n` is the number of data points.
- `yᵢ` is the actual value.
- $$`y_predᵢ`$$ is the predicted value.

### 3. Gradient Descent Update Rule
The gradient descent algorithm updates the parameters \( \beta_0 \) and \( \beta_1 \) iteratively to minimize the cost function. The update rules are as follows:

$$
\beta_0 = \beta_0 - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (y_i - y_{\text{pred}_i})
$$

$$
\beta_1 = \beta_1 - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (y_i - y_{\text{pred}_i}) \cdot X_i
$$

Where:
- `η` is the **learning rate**.
- `β₀` and `β₁` are the model parameters (intercept and slope).
- `Xᵢ` is the value of the feature for the `i-th` data point.

## Script Workflow

1. **Data Generation**:
   - The script generates a synthetic dataset using `make_regression` from scikit-learn with 100 samples and 1 feature, and noise is added to make the data more realistic.

2. **Initialize Parameters**:
   - The model parameters \( \beta_0 \) (intercept) and \( \beta_1 \) (slope) are initialized to zero and one, respectively.

3. **Prediction**:
   - The `predict` function calculates the predicted values of `y` using the formula:
     $$
     y_{\text{pred}} = \beta_0 + \beta_1 \cdot X
     $$

4. **Cost Function**:
   - The `compute_cost` function calculates the Mean Squared Error (MSE) between the actual values `y` and the predicted values `y_pred`.

5. **Gradient Descent**:
   - The `gradient_descent` function performs one step of gradient descent. It calculates the gradients of the cost function with respect to \( \beta_0 \) and \( \beta_1 \) and updates the parameters to reduce the cost.

6. **Model Training**:
   - The `linear_regression` function runs gradient descent for a specified number of epochs (iterations). During each epoch, the model parameters are updated, and the cost is printed every 100 epochs to monitor convergence.

7. **Hyperparameter Tuning**:
   - The `automatic_hyperparameter_tuning` function automatically adjusts the learning rate to find the value that minimizes the cost function more effectively.

8. **Visualization**:
   - After training, the script plots the original data points and the fitted regression line using `matplotlib`.

## Output

```bash
Tuning of Hyperparameters
Epoch 0, Cost: 1156.1838, beta_0: -0.6691, beta_1: 8.1630
Epoch 100, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 200, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 300, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 400, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 500, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 600, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 700, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 800, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372
Epoch 900, Cost: 78.0543, beta_0: 1.1651, beta_1: 44.4372

Optimized Learning Rate: 0.1000
Trained Coefficients:
 beta_0 (Intercept) = 1.1651
 beta_1 (Slope) = 44.4372
```
