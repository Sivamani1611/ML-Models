# Gaussian Mixture Model (GMM)

This repository implements a **Gaussian Mixture Model (GMM)** from scratch using **NumPy** for mathematical computations and **Matplotlib** for visualization. The model is trained using the **Expectation-Maximization (EM)** algorithm to fit a mixture of Gaussian distributions to the data. Hyperparameters such as the number of components (K), the number of iterations, and the convergence tolerance are tuned to select the best model.

## Overview

- **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes all data points are generated from a mixture of several Gaussian distributions with unknown parameters.
- **Expectation-Maximization (EM)**: An iterative algorithm used to find the maximum likelihood estimates of parameters in probabilistic models with latent variables (in this case, the assignment of data points to clusters).
- **Hyperparameter Tuning**: The process of optimizing hyperparameters such as the number of components, max iterations, and convergence tolerance to improve model performance.

## Mathematical Formulas

### 1. Gaussian Probability Density Function (PDF)
The probability density function of a Gaussian distribution is given by:

$$
\mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
$$

Where:
- `x`: Input data point.
- `\mu`: Mean of the Gaussian distribution.
- `\Sigma`: Covariance matrix of the Gaussian distribution.
- `d`: Dimension of the data.

### 2. Expectation-Maximization (EM) Algorithm

- **E-step**: Compute the responsibilities (probabilities that each data point belongs to each Gaussian component).
  
  $$ \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)} $$

- **M-step**: Update the parameters (means, covariances, and mixing coefficients) using the responsibilities computed in the E-step.
  
  $$ \mu_k = \frac{\sum_{i=1}^N \gamma_{ik} x_i}{\sum_{i=1}^N \gamma_{ik}} $$

  $$ \Sigma_k = \frac{\sum_{i=1}^N \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^N \gamma_{ik}} $$

  $$ \pi_k = \frac{1}{N} \sum_{i=1}^N \gamma_{ik} $$

### 3. AIC (Akaike Information Criterion)
The Akaike Information Criterion (AIC) is used for model selection and is calculated as:

$$
\text{AIC} = 2d - 2\log(\hat{L})
$$

Where:
- `d`: Number of parameters in the model (means, covariances, and mixing coefficients).
- `\hat{L}`: Log-likelihood of the model.

## Script Workflow

1. **Data Generation**:
   - Synthetic data is created using `make_blobs` from **sklearn**.
   - The data consists of multiple Gaussian clusters with noise added to make the problem realistic.

2. **Model Training**:
   - Initializes the means, covariances, and mixing coefficients.
   - Iteratively performs the Expectation-Maximization (EM) algorithm.
   - Computes responsibilities in the E-step and updates parameters in the M-step.

3. **Hyperparameter Tuning**:
   - Performs a grid search over different values of `K` (number of Gaussian components), `max_iters` (maximum iterations), and `tol` (convergence tolerance).
   - For each combination, the AIC is calculated to select the best model based on the lowest AIC value.

4. **Model Evaluation**:
   - After training, the model is evaluated based on the log-likelihood and AIC score.
   - The best model parameters are selected for visualization.

5. **Visualization**:
   - Plots the decision boundaries for the best model, showing the cluster areas and the Gaussian centroids.
   - Visualizes the clusters and centroids on the dataset.

## Parameters

| Parameter         | Description                                    | Default Value |
|-------------------|------------------------------------------------|---------------|
| `K`               | Number of Gaussian components (clusters)       | Varies        |
| `max_iters`       | Maximum number of iterations for the EM algorithm | `100`         |
| `tol`             | Convergence tolerance (log-likelihood change threshold) | `1e-4`       |
| `learning_rate`   | Step size for gradient descent (if needed)     | `0.001`       |
| `epochs`          | Number of iterations for model training        | `1000`        |

## Hyperparameter Tuning Process

- **K**: Number of Gaussian components (clusters).
- **max_iters**: Maximum iterations for the Expectation-Maximization algorithm.
- **tol**: Convergence tolerance for the algorithm.
- The script performs a grid search over different values of these parameters and selects the best combination based on the **Akaike Information Criterion (AIC)**.

## Results

- The model that minimizes the AIC value is selected as the best model.
- The final model parameters are visualized, showing the cluster regions and centroids.

## Visualizations

- **Decision Boundaries**: Shows the decision boundaries between different clusters.
- **Cluster Areas**: Visualizes how the data points are divided among different Gaussian components.
- **Centroids**: Displays the centroids of the Gaussian components, marked as red crosses.


## Expected Output
```
Evaluating K=2, max_iters=100, tol=0.001, AIC=3096.2008235681283
Evaluating K=2, max_iters=100, tol=0.0001, AIC=3096.2008235681283
Evaluating K=2, max_iters=200, tol=0.001, AIC=3096.2008235681283
Evaluating K=2, max_iters=200, tol=0.0001, AIC=3096.2008235681283
Evaluating K=3, max_iters=100, tol=0.001, AIC=2707.462509232812
Evaluating K=3, max_iters=100, tol=0.0001, AIC=2707.462509232812
Evaluating K=3, max_iters=200, tol=0.001, AIC=2707.462509232812
Evaluating K=3, max_iters=200, tol=0.0001, AIC=2707.462509232812
Evaluating K=4, max_iters=100, tol=0.001, AIC=2549.15743633181
Evaluating K=4, max_iters=100, tol=0.0001, AIC=2549.1571283745047
Evaluating K=4, max_iters=200, tol=0.001, AIC=2549.15743633181
Evaluating K=4, max_iters=200, tol=0.0001, AIC=2549.1571283745047
Evaluating K=5, max_iters=100, tol=0.001, AIC=2548.8796282070343
Evaluating K=5, max_iters=100, tol=0.0001, AIC=2548.875125936263
Evaluating K=5, max_iters=200, tol=0.001, AIC=2548.8796282070343
Evaluating K=5, max_iters=200, tol=0.0001, AIC=2548.875125936263
Best Model: K=5, max_iters=100, tol=0.0001, AIC=2548.875125936263
```
