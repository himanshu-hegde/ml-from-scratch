import numpy as np
import pandas as pd
import scipy as sp
import pprint
from pprint import pprint
from dataclasses import dataclass, field
from typing import Union, List, Optional, Callable, Literal, Tuple, Dict
from functools import reduce
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt

np.random.seed(0)


@dataclass
class Dataset:
    """
    A class with multiple datasets.

    """

    def credit_card_dataset(
        self,
        data_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Python VS Code\\ML From Scratch\\Datasets\\clustering credit card dataset\\CC GENERAL.csv",
        scale_bool: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieves and preprocesses the credit card dataset.

        :param data_path: The path to the credit card dataset CSV file.
        :return: The preprocessed credit card dataset with training and testing sets.
        """

        data = pd.read_csv(data_path)

        # Step 2: Data preprocessing
        # Drop CUST_ID as it's not relevant for clustering
        data = data.drop(columns=["CUST_ID"])

        # Handle missing values by imputing with the column mean
        data = data.fillna(data.mean())

        if scale_bool:
            # Scale the data for clustering
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            return scaled_data

        return data


@dataclass
class KMeans:
    """
    A class for K-Means clustering.

    2 methods - Random Init and K-Means++

    """

    n_clusters: int = 3
    max_iter: int = 1000

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _assign_best_centroid(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assigns each data point to the closest centroid.

        Args:
            X (np.ndarray): Input data - shape = (n_samples, n_features).
            centroids (np.ndarray): Array of centroids- shape = (n_clusters, n_features).

        Returns:
            np.ndarray: Array of cluster assignments- shape = (1, n_samples).
        """

        distance_to_centroid_matrix = np.array(
            [self._euclidean_distance(X, centroid) for centroid in centroids]
        )  # shape = (n_clusters, n_samples)
        cluster_assignments = np.argmin(distance_to_centroid_matrix, axis=0)

        return cluster_assignments

    def fit_randomized(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the K-Means model using randomized initialization.

        Args:
            X (np.ndarray): Input data - shape = (n_samples, n_features).

        Returns:
            np.ndarray: Array of cluster assignments.
        """
        # Step 1: Randomly select K centroids from the data points
        n_samples, n_features = X.shape[0], X.shape[1]
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        # print(self.centroids)

        for n in range(self.max_iter):

            old_centroids = self.centroids.copy()
            # Distance between each data point and each centroid to assign cluster per data point
            self.cluster_labels = self._assign_best_centroid(X, self.centroids)

            # Mean of each cluster
            for k in range(self.n_clusters):
                if self.cluster_labels[self.cluster_labels == k].shape[0] > 0:
                    self.centroids[k] = np.mean(X[self.cluster_labels == k], axis=0)

            if np.all(old_centroids == self.centroids):
                print(f" Convergence achieved after {n} iterations")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster label for each data point

        Args:
            X (np.ndarray): Input data - shape = (n_samples, n_features).

        Returns:
            np.ndarray: Array of cluster assignments.
        """
        return self._assign_best_centroid(X, self.centroids)

    def intertia(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the inertia for the K-Means model.

        Args:
            X (np.ndarray): Input data - shape = (n_samples, n_features).

        Returns:
            np.ndarray: Inertia value.
        """

        labels = self._assign_best_centroid(X, self.centroids)
        inertia = 0
        for n in range(self.n_clusters):
            if labels[labels == n].shape[0] > 0:
                inertia = inertia = np.sum((X[labels == n] - self.centroids[n]) ** 2)

        return inertia


if __name__ == "__main__":

    dataset = Dataset().credit_card_dataset()
    print(dataset.shape)

    kmeans_fit = KMeans().fit_randomized(dataset)
    # print(kmeans_fit)

    k_values = []
    inertia_values = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k)
        kmeans_fit = kmeans.fit_randomized(dataset)
        inertia_values.append(kmeans_fit.intertia(dataset))
        k_values.append(k)

    plt.figure(figsize=(12, 5))
    plt.plot(k_values, inertia_values, marker="o")
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid()
    plt.show()
