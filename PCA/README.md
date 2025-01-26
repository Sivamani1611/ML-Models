# Principal Component Analysis (PCA) with Wine Quality Dataset

This repository implements **Principal Component Analysis (PCA)** using the **Wine Quality Dataset** from `sklearn`. The dataset is preprocessed, and PCA is applied to reduce the dimensionality while retaining 95% of the variance. The code also provides visualizations for both 2D and 3D projections of the dataset after PCA.

## Overview

- **Principal Component Analysis (PCA)**: A statistical technique used for dimensionality reduction, which transforms the data into a set of orthogonal components.
- **Wine Quality Dataset**: A popular dataset used to demonstrate classification and clustering techniques, containing chemical properties of wine and their quality ratings.
- **Explained Variance**: PCA finds a lower-dimensional representation of the data that retains the maximum variance from the original features.
- **Preprocessing**: The dataset is scaled using either **StandardScaler** or **MinMaxScaler** to normalize the features before applying PCA.

## Workflow

1. **Data Loading and Preprocessing**:
   - The Wine Quality dataset is loaded from `sklearn.datasets.load_wine`.
   - Data is scaled using either `StandardScaler` or `MinMaxScaler`.

2. **Principal Component Analysis (PCA)**:
   - PCA is applied to the scaled data to reduce the dimensionality while retaining 95% of the variance.
   - The optimal number of principal components is determined based on the cumulative explained variance.

3. **Visualization**:
   - **Cumulative Explained Variance**: A plot showing the cumulative explained variance as a function of the number of principal components.
   - **2D PCA Projection**: A 2D scatter plot of the dataset projected onto the first two principal components.
   - **3D PCA Projection**: A 3D scatter plot of the dataset projected onto the first three principal components (if applicable).

## Mathematical Formulas

### PCA

The **PCA** algorithm finds the orthogonal directions (principal components) that maximize the variance of the data. This is achieved through the following steps:

1. **Standardize the Data**: The dataset is first centered and scaled.
2. **Compute the Covariance Matrix**: The covariance matrix captures the relationships between different features.
3. **Eigenvalue Decomposition**: Eigenvectors and eigenvalues are computed from the covariance matrix.
4. **Projection**: Data is projected onto the selected principal components.

### Explained Variance

The explained variance ratio for each principal component is given by:

$$
\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum \lambda_i}
$$

Where:
- \( \lambda_i \) is the eigenvalue corresponding to the \( i \)-th principal component.

## Hyperparameters

| Parameter           | Description                                | Default Value |
|---------------------|--------------------------------------------|---------------|
| `scale_type`        | Method used to scale data (`standard` or `minmax`) | `standard`    |
| `target_variance`   | The target variance to retain after PCA    | `0.95`        |

## Outputs

- **Cumulative Explained Variance**: A plot showing how the variance is retained as we increase the number of principal components.
- **2D PCA Projection**: A scatter plot of the data projected onto the first two principal components.
- **3D PCA Projection**: A scatter plot of the data projected onto the first three principal components (if applicable).
- The plots are saved as image files:
  - `cumulative_explained_variance.png`
  - `2d_pca_projection.png`
  - `3d_pca_projection.png`

## Requirements

- **NumPy**: For matrix operations and mathematical computations.
- **Pandas**: For handling the data.
- **Matplotlib**: For plotting the results.
- **Scikit-learn**: For data preprocessing, PCA implementation, and dataset loading.



## Output

- **Cumulative Explained Variance Plot**: Displays how much variance is explained by each additional principal component.
- **2D PCA Projection Plot**: Shows a 2D scatter plot of the dataset after projection onto the first two principal components.
- **3D PCA Projection Plot**: Shows a 3D scatter plot of the dataset after projection onto the first three principal components.
