# Polynomial Regression with Gradient Descent

This repository implements a **Polynomial Regression** model from scratch using **Gradient Descent** for optimization. The script demonstrates how to fit a polynomial curve to data, compute the cost function, and update coefficients iteratively.

## Overview

- **Polynomial Regression**: A regression technique that models the relationship between a dependent variable (`y`) and an independent variable (`X`) as a polynomial of degree `d`.
- **Gradient Descent**: An iterative optimization algorithm used to minimize the cost function by adjusting the coefficients (`β`).
- **Regularization**: L2 regularization is included to prevent overfitting and penalize large coefficients.

## Mathematical Formulas

### 1. Polynomial Regression Model
The polynomial regression model is expressed as:

$$
\hat{y} = \beta_0 + \beta_1 \cdot X + \beta_2 \cdot X^2 + \dots + \beta_d \cdot X^d
$$

Where:
- `ŷ`: Predicted value.
- `βᵢ`: Coefficient for the `i`-th term of the polynomial.
- `X`: Input feature.
- `d`: Degree of the polynomial.

### 2. Cost Function: Mean Squared Error (MSE)
The cost function minimizes the difference between predicted and actual values:

$$
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \frac{\lambda}{2n} \sum_{j=1}^{d} \beta_j^2
$$

Where:
- `n`: Number of data points.
- `λ`: Regularization parameter.
- `βᵢ`: Coefficients of the polynomial terms.
- The first term is the Mean Squared Error (MSE) and the second term is the L2 regularization.

### 3. Gradient Descent Update Rule
The coefficients are updated iteratively as follows:

$$
\beta_j = \beta_j - \eta \cdot \frac{\partial J}{\partial \beta_j}
$$

Where:
- `η`: Learning rate.
- `∂J / ∂βᵢ`: Gradient of the cost function with respect to `βᵢ`.

$$
∂J / ∂βᵢ = (1 / n) Σ (yi - ŷi) * Xᵢ + (λ / n) βᵢ
$$
Where:
`Xᵢ` is the `i`-th feature of the input data.


## Script Workflow

1. **Data Generation**:
   - Synthetic data is created using `make_regression` from `scikit-learn`.

2. **Feature Transformation**:
   - The input `X` is transformed into polynomial features up to a specified degree.

3. **Gradient Descent**:
   - Coefficients are initialized and iteratively updated using gradient descent.

4. **Cost Computation**:
   - The `compute_cost` function calculates the Mean Squared Error with L2 regularization.

5. **Model Training**:
   - The script trains the polynomial model for degrees 1 through 5 and prints the coefficients.

6. **Visualization**:
   - The script plots the polynomial fit against the original data for each degree.

7. **Error Handling**:
   - The script detects and terminates fitting if numerical instability (e.g., NaN values) occurs.

## Parameters

| Parameter         | Description                                     | Default Value |
|-------------------|-------------------------------------------------|---------------|
| `degree`          | Degree of the polynomial                       | `1 to 5`      |
| `learning_rate`   | Step size for gradient descent                  | `0.001`       |
| `epochs`          | Number of iterations for gradient descent       | `1000`        |
| `lambda_reg`      | Regularization strength (L2 regularization)     | `0.01`        |

## Visualization
The plots show the polynomial regression fit for degrees 1 to 5. Example:

![Polynomial Regression Fit](https://via.placeholder.com/600x400)

## Example Output

```bash
Degree 1: Coefficients: [-1.62147689 26.92895273]
Degree 2: Coefficients: [-0.4793349  26.72877946 -1.90998976]
Degree 3: Coefficients: [-1.57252209 12.46909576  2.09888222  7.76598048]
Degree 4: Coefficients: [-1.90294131 11.79606176 -2.10926464  8.89543575  1.41235451]
Degree 5: Coefficients: [-1.42536453 12.53130405 -1.29253234 13.43825114  0.21366811 -1.22475268]
```
