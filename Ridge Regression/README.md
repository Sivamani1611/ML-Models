# Regression Visualization with Polynomial Ridge

This repository implements **Polynomial Ridge Regression** and **Linear Regression** models and visualizes their performance on noisy data. The script demonstrates how to apply linear and polynomial ridge regression to fit curves, using both regularization and polynomial transformations.

## Overview

- **Linear Regression**: A model that fits a linear relationship between a dependent variable (`y`) and an independent variable (`X`).
- **Polynomial Ridge Regression**: A model that applies polynomial transformations to `X` and then fits a ridge regression (regularized linear regression) model.
- **Visualization**: The script generates visual comparisons of the fitted models against the ideal function and noisy data.

## Mathematical Formulas

### 1. Linear Regression Model
The linear regression model is expressed as:

$$
\hat{y} = \beta_0 + \beta_1 \cdot X
$$

Where:
- `ŷ`: Predicted value.
- `βᵢ`: Coefficients for the linear terms.
- `X`: Input feature.

### 2. Polynomial Ridge Regression Model
In polynomial ridge regression, we first transform `X` into polynomial features and then apply ridge regression:

$$
\hat{y} = \beta_0 + \beta_1 \cdot X + \beta_2 \cdot X^2 + \dots + \beta_d \cdot X^d
$$

Where:
- `βᵢ`: Coefficients for the polynomial terms.
- `d`: Degree of the polynomial.
- The regularization term is included to penalize large coefficients, preventing overfitting.

### 3. Ridge Regression Regularization
Ridge regression minimizes the following cost function:

$$
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \frac{\alpha}{2n} \sum_{j=1}^{d} \beta_j^2
$$

Where:
- `n`: Number of data points.
- `α`: Regularization parameter that controls the penalty for large coefficients.
- `βᵢ`: Coefficients of the polynomial terms.
- The first term is the Mean Squared Error (MSE) and the second term is the L2 regularization.

## Script Workflow

1. **Data Generation**:
   - Noisy data is generated using a custom function: \( y = X^2 \cdot \sin(X) \cdot \exp\left(\frac{-X}{10}\right) \), with added Gaussian noise.

2. **Feature Transformation**:
   - The input `X` is scaled and optionally transformed into polynomial features up to a specified degree.

3. **Model Training**:
   - Linear regression is trained on the scaled input data.
   - Polynomial ridge regression is trained on polynomial features, with varying degrees and regularization strengths (`α`).

4. **Visualization**:
   - The script generates plots comparing the ideal function, noisy data, linear regression, and polynomial ridge regression fits.

5. **Error Handling**:
   - The script ensures proper handling of edge cases, such as invalid values during data processing.

## Parameters

| Parameter         | Description                                         | Default Value     |
|-------------------|-----------------------------------------------------|-------------------|
| `x_min`           | Minimum value for `X`                               | `1`               |
| `x_max`           | Maximum value for `X`                               | `10`              |
| `n_points`        | Number of data points generated                     | `41`              |
| `noise_mean`      | Mean of Gaussian noise                             | `0`               |
| `noise_sd`        | Standard deviation of Gaussian noise               | `2`               |
| `ridge_alpha`     | Tuple of regularization strengths (α) for ridge regression | `[0.001, 0.01, 0.1]` |
| `degree_max`      | Maximum degree for polynomial features              | `8`               |

## Visualization

The generated plots show the comparison of the ideal function, noisy data, and the fits of both linear and polynomial ridge regression. 

### Example Plots:
- **Linear Regression**: A simple linear fit to the noisy data.
- **Polynomial Ridge Regression**: A polynomial regression with regularization, showing how different degrees and α values affect the model fit.

![Linear vs Polynomial Ridge Comparison](https://raw.githubusercontent.com/Sivamani1611/Mathematics-Driven-Machine-Learning/refs/heads/main/Ridge%20Regression/comparison_alpha_0.01.png)
