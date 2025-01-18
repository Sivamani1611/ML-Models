# Lasso Regression Visualization

This repository implements **Lasso Regression** and visualizes its performance on noisy data. The script demonstrates how Lasso Regression works, applying both the **L1 regularization** technique to prevent overfitting and visualizing the optimization process through loss function convergence.

## Overview

- **Lasso Regression**: A linear model that fits a linear relationship between a dependent variable (`y`) and an independent variable (`X`), while applying **L1 regularization** to reduce the magnitude of model coefficients, resulting in sparse solutions.
- **Visualization**: The script generates visualizations showing the loss function's convergence over iterations and compares the predicted vs actual values for the fitted model.

## Mathematical Formulas

### 1. Lasso Regression Model
The Lasso regression model is expressed as:

$$
\hat{y} = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \dots + \beta_d \cdot X_d
$$

Where:
- `ŷ`: Predicted value.
- `βᵢ`: Coefficients for the linear terms.
- `Xᵢ`: Input feature.

### 2. Lasso Regression Loss Function
Lasso regression minimizes the following cost function:

$$
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{d} |\beta_j|
$$

Where:
- `n`: Number of data points.
- `α`: Regularization parameter that controls the penalty for large coefficients.
- `βᵢ`: Coefficients of the linear terms.
- The first term is the Mean Squared Error (MSE) and the second term is the **L1 regularization**, encouraging sparsity.

## Script Workflow

1. **Data Generation**:
   - The dataset is generated using a linear relationship with added Gaussian noise.

2. **Model Training**:
   - Lasso regression is trained using **Gradient Descent** to minimize the loss function that includes L1 regularization.

3. **Visualization**:
   - The script generates plots showing the loss function's convergence over iterations and compares the predicted vs actual values of the model.

4. **Error Handling**:
   - The script ensures proper handling of edge cases, such as invalid values during data processing.

## Parameters

| Parameter        | Description                                           | Default Value     |
|------------------|-------------------------------------------------------|-------------------|
| `alpha`          | Regularization strength (controls L1 penalty)         | `0.1`             |
| `learning_rate`  | Step size for gradient descent                        | `0.01`            |
| `max_iter`       | Number of iterations for gradient descent             | `1000`            |
| `m`              | Number of data points                                 | `100`             |
| `n`              | Number of features                                    | `3`               |

## Visualization

The script generates the following plots:
1. **Loss Function Convergence**: A plot showing how the loss decreases over iterations as the gradient descent algorithm optimizes the Lasso model.
2. **Predictions vs Actual Values**: A scatter plot comparing the actual values against the predicted values from the trained model.

### Example Plots:
- **Loss Function Convergence**: The red line showing the loss value at each iteration of gradient descent.
- **Predictions vs Actual**: A scatter plot of the predicted vs actual values, where a perfect fit would lie along the red line.

![Lasso Regression Loss Convergence](https://raw.githubusercontent.com/Sivamani1611/Lasso-Regression/Lasso_Regression_Loss_Function.png)

## Output

```
Iteration 0: Loss = 17.10017587125422
Iteration 100: Loss = 2.371747189038191
Iteration 200: Loss = 0.8938882498819256
Iteration 300: Loss = 0.7094283884879866
Iteration 400: Loss = 0.6794596885495756
Iteration 500: Loss = 0.6733035583542335
Iteration 600: Loss = 0.6718305647450943
Iteration 700: Loss = 0.6714485212333456
Iteration 800: Loss = 0.6713454537854403
Iteration 900: Loss = 0.6713171014183309
Trained weights: [ 2.80073079 -1.93887906  0.85930179]
Trained bias: 4.0935950285085605
```
