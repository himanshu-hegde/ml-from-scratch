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

    def _transform_data(
        self, df: pd.DataFrame, exclude_cols: list = None
    ) -> Union[pd.DataFrame, ColumnTransformer]:
        """
        Transform the data by one-hot encoding object columns and scaling numeric columns.

        Parameters:
        - df (DataFrame): The data to transform.
        - exclude_cols (list, optional): A list of columns to exclude from transformation. Defaults to None.

        Returns:
        - transformed_df (DataFrame): The transformed data.
        """
        # Get the column types
        object_cols = df.select_dtypes(include=["object"]).columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        # Exclude the specified columns
        if exclude_cols:
            object_cols = [col for col in object_cols if col not in exclude_cols]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Create the transformers
        onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        scaler = StandardScaler()

        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", onehot_encoder, object_cols),
                ("scaler", scaler, numeric_cols),
            ],
            remainder="passthrough",
        )

        # Use the preprocessor to transform the data
        transformed_df = preprocessor.fit_transform(df)

        # Convert the transformed data to a DataFrame
        transformed_df = pd.DataFrame(
            transformed_df, columns=preprocessor.get_feature_names_out()
        )

        return transformed_df, preprocessor

    def heart_classication_dataset(
        self,
        data_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Python VS Code\\ML From Scratch\\Datasets\\heart_classification.csv",
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        """
        Loads the heart classification dataset from a CSV file, preprocesses it,
        and splits it into training and testing sets.

        :param data_path: The path to the heart classification CSV file.
                            Defaults to a specific file path.
        :return: A tuple containing the preprocessed training and testing datasets.
        """

        heart_classification = pd.read_csv(data_path)
        heart_classification = heart_classification.drop(columns=["Unnamed: 0"])
        heart_classification.columns = [
            col.lower() for col in heart_classification.columns
        ]

        heart_classification["ahd"] = heart_classification["ahd"].apply(
            lambda col: 1 if col == "Yes" else 0
        )
        heart_classification = heart_classification.dropna()

        # Split the data into training and testing sets
        heart_classification_train, heart_classification_test = train_test_split(
            heart_classification, test_size=0.2, random_state=42, shuffle=True
        )
        heart_classification_train, preprocessor_obj = self._transform_data(
            heart_classification_train, exclude_cols=["ahd"]
        )
        heart_classification_test = pd.DataFrame(
            preprocessor_obj.transform(heart_classification_test),
            columns=preprocessor_obj.get_feature_names_out(),
        )

        return heart_classification_train, heart_classification_test


@dataclass
class PCA:

    n_components: int = None
    pca_components: np.ndarray = None
    explained_variance: np.ndarray = None

    def _power_method(
        self, X_cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the Power Method to find the eigenvalues and eigenvectors.

        Args:
        - X_cov (np.ndarray): The covariance matrix of the data to fit the model to.

        Returns:
        - np.ndarray: The eigenvalues.
        - np.ndarray: The eigenvectors.
        """

        n_features = X_cov.shape[0]
        print(f"iter = {max_iter}")

        vector = np.random.rand(n_features)
        vector = vector / np.linalg.norm(vector)

        # Initialise eigenvalues and eigenvectors - iterate and find the eigenvalues and eigenvectors until they converge. This means multiplying the cov matrix by the eigenvectors(initially random) will give us the eigenvalues.

        for i in range(max_iter):
            prev_vector = (
                vector.copy()
            )  # store the previous vector to be able to check if it has converged
            vector = np.dot(X_cov, prev_vector)
            vector = vector / np.linalg.norm(vector)

            # check if the eigenvectors have converged
            if np.abs(np.dot(prev_vector, vector)) > 1 - tol:
                break

        # Compute Rayleigh quotient: λ = v^T A v/(v^T v) - denomiator is 1 since v is a unit vector(has been normalised)
        eigen_values = np.dot(np.dot(vector, X_cov), vector)

        return eigen_values, vector

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model to the data.

        We will use the Power Method to find the eigenvalues and eigenvectors.

        Args:
        - X (np.ndarray): The data to fit the model to.

        Returns:
        - np.ndarray: The transformed data.
        """

        # centre the data
        X_centered = X - np.mean(X, axis=0)

        n_samples, n_features = X.shape

        # covariance matrix
        X_cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Define n_components
        self.n_components = (
            n_features if self.n_components is None else self.n_components
        )

        self.pca_components = np.zeros((n_features, self.n_components))
        self.explained_variance = np.zeros(self.n_components)

        remaining_matrix = X_cov.copy()
        # Start the power method  - each iteration gives the higher eigenvalue. We then subtract it from the matrix (resulting in a remaining matrix), and repeat.
        for i in range(self.n_components):

            eigenvalues, eigenvectors = self._power_method(remaining_matrix)

            # Store the eigenvectors and eigenvalues of each iteration
            self.pca_components[:, i] = eigenvectors
            self.explained_variance[i] = eigenvalues

            # Subtract variance from each iteration from the remaining matrix ( A_{i+1} = A_i - λᵢvᵢvᵢᵀ) - using np.outer, as shape of eigenvectors is (n_features,). if we reshape it to (n_features,1), we can use np.dot as well.
            remaining_matrix = remaining_matrix - eigenvalues * np.outer(
                eigenvectors, eigenvectors
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted PCA model.

        Args:
        - X (np.ndarray): The data to transform.

        Returns:
        - np.ndarray: The transformed data.
        """
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.pca_components)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data using the fitted PCA model.

        Args:
        - X (np.ndarray): The data to inverse transform.

        Returns:
        - np.ndarray: The inverse transformed data.
        """
        return np.dot(X, self.pca_components) + np.mean(X, axis=0)


if __name__ == "__main__":
    model_used = "decision_tree"

    df_train, df_test = Dataset().heart_classication_dataset()

    # test pca
    pca = PCA(n_components=None)
    fit_pca = pca.fit(df_train)
    transformed_train = fit_pca.transform(df_train)

    # # Calculate variance ratios
    total_var = np.sum(fit_pca.explained_variance)
    var_ratio = fit_pca.explained_variance / total_var

    cum_var_ratio = np.cumsum(var_ratio)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    components = range(1, len(var_ratio) + 1)

    # Scree plot
    ax1.plot(components, var_ratio, "bo-", linewidth=2)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Scree Plot")
    ax1.grid(True)

    # Add value labels
    for i, v in enumerate(var_ratio):
        ax1.text(i + 1, v, f"{v:.2%}", ha="center", va="bottom")

    # Cumulative variance plot
    ax2.plot(components, cum_var_ratio, "ro-", linewidth=2)
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance Ratio")
    ax2.set_title("Cumulative Explained Variance")
    ax2.grid(True)

    # Add value labels for cumulative plot
    for i, v in enumerate(cum_var_ratio):
        ax2.text(i + 1, v, f"{v:.2%}", ha="center", va="bottom")

    # Set y-axis to percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Adjust layout
    plt.tight_layout()

    # This line is necessary to display the plot
    plt.show()
