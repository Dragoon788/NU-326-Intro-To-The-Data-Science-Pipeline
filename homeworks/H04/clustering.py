import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from typing import Callable

np.random.seed(2024)

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.
    
    Arguments:
        x : np.ndarray
            First vector.
        y : np.ndarray
            Second vector.
    Returns:
        float : Euclidean distance.
    """
    return (np.sqrt(np.sum(np.square(x-y))))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance between two vectors.
    
    Arguments:
        x : np.ndarray
            First vector.
        y : np.ndarray
            Second vector.
    
    Returns:
        float : Cosine distance.

    """
    return 1-(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance between two vectors.
    
    Arguments:
        x : np.ndarray
            First vector.
        y : np.ndarray
            Second vector.

    Returns:
        float : Manhattan distance.
    """
    return np.sum(np.abs(x-y))

def pairwise_distance(X: np.ndarray, Y: np.ndarray, distance_method: Callable) -> np.ndarray:
    """
    Compute pairwise distance between two matrices, using a provided distance method.

    NOTE: This is pairwise, so you need to compute the distance
    between EACH sample in X and EACH sample in Y. Your output
    should have shape (n_samples_1, n_samples_2).

    Arguments:
        X : np.ndarray
            Input data of shape (n_samples_1, n_features).
        Y : np.ndarray
            Input data of shape (n_samples_2, n_features).

    Returns: 
        np.ndarray of pairwise distances of shape (n_samples_1, n_samples_2).
    """
    distances = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            # 1. Calculate the distance between the test sample and the training sample.
            dist = distance_method(X[i], Y[j])
            
            # 2. Append the (distance, label) tuple to the distances list.
            distances[i,j] = dist

    return distances


def k_means(X: np.ndarray, k: int, initial_centroids: np.ndarray, distance_method: Callable, max_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering algorithm.

    NOTE: You should use pairwise_distance function to compute distances.

    NOTE: Your loop should break when the labels do not change.

    Arguments:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of clusters.
        initial_centroids (np.ndarray): Initial centroids of shape (k, n_features).
        distance_method (Callable): Distance method to use.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        tuple: Final centroids of shape (k, n_features) and labels of shape (n_samples,).
    """
 
    # Set initial labels to -1.
    labels = np.zeros(X.shape[0]) - 1

    # Copy initial centroids to avoid modifying the original array.
    centroids = initial_centroids.copy()

    # Iterate until convergence or max_iter.
    for _ in range(max_iter):
        # 1. Compute distances between each sample and each centroid.
        distances = pairwise_distance(X, centroids, distance_method)
        
        # 2. Assign each sample to the closest centroid.
        nearest_centroid = np.argmin(distances, axis=1)
        
        
        # 3. Exit the loop if labels do not change from previous iteration.
        if np.array_equal(nearest_centroid, labels):
            break

        # 4. Update labels.
        labels = nearest_centroid


        # 5. Update k centroids to be the mean of all labeled samples.
        for i in range(k):
            group_k = X[labels == i]
            if len(group_k) > 0:
                centroids[i] = np.mean(group_k, axis = 0)

    # 6. Return final centroids and labels.
    return centroids, labels



def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN clustering algorithm from sklearn.

    NOTE: Take a look at the DBScan class in sklearn. There is
    a field that you can access for the derived labels. 

    Arguments:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
        np.ndarray: Cluster labels of shape (n_samples,).
    """
    ans = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return ans.labels_


def local_silhouette_score(X: np.ndarray, labels: np.ndarray, metric: str) -> float:
    """Compute silhouette score.
    
    NOTE: You should use the sklearn package for this, 
    and ensure you use the metric argument.

    Arguments:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels of shape (n_samples,).
        metric (str): Metric to use for distance computation.
    
    Returns:
        float: Silhouette score.
    """
    # print(labels)
    return silhouette_score(X, labels, metric=metric)