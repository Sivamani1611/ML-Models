# Support Vector Machine (SVM)

This repository implements a **Support Vector Machine (SVM)** classifier from scratch using **NumPy** for mathematical computations and **Matplotlib** for visualization. The model is trained using a hinge-loss-based optimization to separate two classes while maximizing the margin.

## Overview

- **Support Vector Machine**: A supervised machine learning algorithm used for classification tasks. It aims to find the hyperplane that best separates the data into two classes.
- **Hinge Loss**: A loss function used in SVMs to penalize data points inside the margin or on the wrong side of the decision boundary.
- **Gradient Descent**: An optimization algorithm used to minimize the hinge loss and adjust the model parameters (weights and bias).

## Mathematical Formulas

### 1. Decision Function
The decision boundary in SVM is represented by the following equation:

$$
y_{\text{pred}} = w \cdot X + b
$$

Where:
- `w`: Weight vector (defines the orientation of the hyperplane).
- `b`: Bias term (defines the position of the hyperplane).
- `X`: Input features.

### 2. Hinge Loss
To ensure maximum margin classification, the SVM minimizes the following hinge loss:

$$
L = \frac{1}{2} \|w\|^2 + C \cdot \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \cdot (w \cdot X_i + b))
$$

Where:
- `\|w\|^2`: Regularization term to ensure smoothness.
- `C`: Regularization parameter controlling the trade-off between maximizing margin and minimizing misclassifications.
- `y_i`: True label (+1 or -1).
- `X_i`: Feature vector for the i-th data point.

### 3. Gradient Updates
The parameters `w` and `b` are updated iteratively using gradient descent as follows:

For points violating the margin (`y_i \cdot (w \cdot X_i + b) < 1`):

$$
w = w - \eta \cdot (w - C \cdot y_i \cdot X_i)
$$

$$
b = b + \eta \cdot C \cdot y_i
$$

For points correctly classified:

$$
w = w - \eta \cdot w
$$

Where:
- `\eta`: Learning rate.

## Script Workflow

1. **Data Generation**:
   - Synthetic data is created using `make_classification` from `sklearn`.
   - The data is split into two classes, with Gaussian noise added to make the problem realistic.

2. **Model Training**:
   - Initializes weights (`w`) and bias (`b`) to zero.
   - Iteratively minimizes the hinge loss using gradient descent.
   - Tracks the loss value during training to observe convergence.

3. **Model Evaluation**:
   - After training, the model is evaluated on the test set to calculate accuracy and performance.

4. **Visualization**:
   - The script plots the decision boundary, margin lines, and data points.
   - Highlights points violating the margin condition for further analysis.

## Parameters

| Parameter         | Description                                    | Default Value |
|-------------------|------------------------------------------------|---------------|
| `learning_rate`   | Step size for gradient descent                 | `0.001`       |
| `C`               | Regularization parameter                       | `1`           |
| `epochs`          | Number of iterations for gradient descent      | `1000`        |


