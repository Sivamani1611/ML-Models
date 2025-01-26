import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

K = 4
centroids = data[np.random.choice(data.shape[0], K, replace=False)]

def euclidean_distance(x, c):
    return np.sqrt(np.sum((x - c) ** 2))

def assign_clusters(data, centroids):
    clusters = []
    for x in data:
        distances = [euclidean_distance(x, c) for c in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(data, clusters, K):
    new_centroids = np.zeros((K, data.shape[1]))
    for i in range(K):
        points_in_cluster = data[clusters == i]
        new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

def kmeans(data, K, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    all_centroids = [centroids]
    
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, K)
        all_centroids.append(new_centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, clusters, all_centroids

final_centroids, final_clusters, all_centroids = kmeans(data, K)

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = np.array([assign_clusters(np.c_[x, y], final_centroids)[0] for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c=final_clusters, cmap='viridis', label="Data Points")
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='x', s=200, c='red', label="Final Centroids")
plt.title("K-means Clustering with Cluster Areas")
plt.legend()
plt.savefig('cluster_areas.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, label="Data Points")
plt.scatter(all_centroids[0][:, 0], all_centroids[0][:, 1], c='red', marker='x', s=200, label="Initial Centroids")
plt.title("Initial Centroids")
plt.legend()
plt.savefig('initial_centroids.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, label="Data Points")
plt.scatter(all_centroids[1][:, 0], all_centroids[1][:, 1], c='blue', marker='x', s=200, label="Centroids After 1st Update")
plt.title("Centroids After 1st Update")
plt.legend()
plt.show()
plt.savefig('centroids_after_1st_update.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=final_clusters, cmap='viridis', label="Clusters")
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='x', s=200, c='red', label="Final Centroids")
plt.title("Final Clusters and Centroids")
plt.legend()
plt.savefig('final_clusters_centroids.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
for i in range(len(all_centroids)):
    plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5)
    plt.scatter(all_centroids[i][:, 0], all_centroids[i][:, 1], c='red', marker='x', s=200)
    plt.title(f"Centroids Movement (Iteration {i+1})")
    plt.pause(0.5)
    plt.clf()
plt.savefig('centroids_movement.png')
plt.close()
