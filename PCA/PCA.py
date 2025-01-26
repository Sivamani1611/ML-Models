import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

def load_and_preprocess_data(scale_type='standard'):
    data = load_wine()
    X = data.data
    y = data.target
    
    if scale_type == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def apply_pca(X, target_variance=0.95):
    pca = PCA(n_components=None)
    pca.fit(X)
    
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.savefig("cumulative_explained_variance.png")
    
    n_components = np.argmax(explained_variance_ratio >= target_variance) + 1
    print(f"Optimal number of components to retain {target_variance * 100}% variance: {n_components}")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca

def visualize_2d_projection(X_pca, labels=None):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Wine Class')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    
    plt.title("2D PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("2d_pca_projection.png")

def visualize_3d_projection(X_pca, labels=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', alpha=0.6)
        fig.colorbar(scatter, label='Wine Class')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6)
    
    ax.set_title("3D PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig("3d_pca_projection.png")

X_scaled, y = load_and_preprocess_data(scale_type='standard')
    
X_pca, pca_model = apply_pca(X_scaled, target_variance=0.95)
    
visualize_2d_projection(X_pca, labels=y)
if X_pca.shape[1] > 2:
    visualize_3d_projection(X_pca, labels=y)

