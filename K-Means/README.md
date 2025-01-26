# K-means Clustering

This repository implements a **K-means clustering** algorithm from scratch using **NumPy** for mathematical computations and **Matplotlib** for visualization. The model is trained to partition a dataset into `K` clusters by minimizing the within-cluster variance and iteratively updating the cluster centroids.

## Overview

- **K-means Clustering**: A popular unsupervised machine learning algorithm used for partitioning a dataset into `K` clusters, where each data point belongs to the cluster whose centroid is nearest.
- **Centroid Update**: The algorithm iteratively updates the centroids by calculating the mean of the data points in each cluster.
- **Euclidean Distance**: The distance metric used to assign data points to the nearest centroid and to update centroids.

## Mathematical Formulas

### 1. Centroid Calculation
The centroid of a cluster is computed as the mean of the points assigned to the cluster:

$$
c_k = \frac{1}{n_k} \sum_{i=1}^{n_k} x_i
$$

Where:
- `c_k`: Centroid of cluster `k`.
- `n_k`: Number of data points in cluster `k`.
- `x_i`: Data points assigned to cluster `k`.

### 2. Euclidean Distance
The distance between a data point `x` and a centroid `c` is calculated as:

$$
d(x, c) = \sqrt{\sum_{i=1}^{d} (x_i - c_i)^2}
$$

Where:
- `x`: Data point.
- `c`: Centroid.
- `d`: Dimensionality of the data.

### 3. K-means Objective Function
The objective function for K-means aims to minimize the within-cluster variance, which is the sum of squared distances from each point to its assigned centroid:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} (x_i - c_k)^2
$$

Where:
- `C_k`: Points in cluster `k`.
- `c_k`: Centroid of cluster `k`.

## Script Workflow

1. **Data Generation**:
   - Synthetic data is created using `make_blobs` from `sklearn` to simulate a clustering problem with multiple centers and Gaussian noise.

2. **Model Training**:
   - Randomly initializes `K` centroids from the dataset.
   - Iteratively assigns each data point to the nearest centroid.
   - Recalculates the centroids as the mean of the points in each cluster.
   - Continues until convergence (no change in centroids).

3. **Model Evaluation**:
   - The algorithm doesn't require a test set but can be evaluated based on the final clustering accuracy, silhouette score, or visual inspection.

4. **Visualization**:
   - The script plots the clusters, centroids, and decision boundaries for each cluster.
   - It also visualizes the movement of centroids over iterations to show how the algorithm converges.

## Parameters

| Parameter         | Description                                        | Default Value |
|-------------------|----------------------------------------------------|---------------|
| `K`               | Number of clusters                                 | `4`           |
| `max_iters`       | Maximum number of iterations for the algorithm     | `100`         |
| `epsilon`         | Convergence tolerance (for centroid movement)     | `1e-4`        |

## Saved Plots

The following plots are saved during the execution of the script:

- **`cluster_areas.png`**: Visualization of decision boundaries with cluster areas.
- **`initial_centroids.png`**: Plot showing the initial centroids before training.
- **`centroids_after_1st_update.png`**: Plot showing the centroids after the first update.
- **`final_clusters_centroids.png`**: Plot showing the final clusters and centroids after convergence.
- **`centroids_movement.png`**: Animation of centroid movement during the iterations.

## Dependencies

- `NumPy` for numerical operations.
- `Matplotlib` for plotting.
- `sklearn.datasets` for synthetic data generation.

