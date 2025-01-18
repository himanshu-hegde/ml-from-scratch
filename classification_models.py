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


np.random.seed(0)


def calculate_binary_classification_accuracy_metrics(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> tuple:
    """
    Calculates various metrics to evaluate the performance of a binary classification model.

    :param actual: The actual values of the target variable.
    :param predicted: The predicted values of the target variable.
    :return: A tuple containing the accuracy, precision, recall, f1 score, and AUC-ROC score.
    """
    # Calculate Accuracy
    accuracy = accuracy_score(actual, predicted)

    # Calculate Precision
    precision = precision_score(actual, predicted)

    # Calculate Recall
    recall = recall_score(actual, predicted)

    # Calculate F1 Score
    f1 = f1_score(actual, predicted)

    # Calculate AUC-ROC
    auc_roc = roc_auc_score(actual, predicted)

    # Print the metrics
    print(
        f"Accuracy: {round(accuracy,3)}  Precision: {round(precision,3)} Recall: {round(recall,3)} \n F1 Score: {round(f1,3)}  AUC-ROC Score: {round(auc_roc,3)}"
    )

    return accuracy, precision, recall, f1, auc_roc


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
class LogiticRegression:
    """
    A class for performing Logistic Regression.
    """

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Calculates the sigmoid function of z.

        :param z: The input array.
        :return: The sigmoid function of z.
        """
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Adds an intercept term to the input array.

        :param X: The input array.
        :return: The input array with an intercept term.
        """
        intercept = np.ones(shape=(X.shape[0], 1))
        return np.hstack(tup=(intercept, X))

    def _feature_and_target_dataset(
        self, dataset: pd.DataFrame, target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:

        return (
            dataset.drop(columns=[target_column]).to_numpy(),
            dataset[target_column].to_numpy(),
        )

    def binary_threshold(self, z: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Applies a threshold function to the input array.

        :param z: The input array.
        :return: The thresholded array.
        """
        return np.where(z >= threshold, 1, 0)

    def _calculate_imbalance_weights(
        self, target_column: np.ndarray | List[int | float]
    ) -> np.ndarray:
        """
        Calculate weights to handle class imbalance.

        :param target_column: The target variable.

        :return: A list of weights to handle class imbalance, or None if handle_imbalance is False or imbalance_weights is provided.
        """

        class_count = Counter(target_column)
        n_samples = len(target_column)
        imbalance_weights_dict = {
            cls: (n_samples / (count * len(class_count)))
            for cls, count in class_count.items()
        }
        imbalance_weight_list = np.array(
            [imbalance_weights_dict[cls] for cls in target_column]
        )
        return imbalance_weight_list

    def _binary_cross_entropy_gradient_calculation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefficients: np.ndarray,
        imbalance_weights_list: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the gradient of the binary cross entropy loss function.

        Gradient = 1/n * Xt . (h - y), where h is the sigmoid function of z, where z = (X . coefficients_matrix). Xt is transpose of X.

        :param X: The input array.
        :param y: The target array.
        :param coefficients: The coefficients of the logistic regression model.
        :param imbalance_weights_list: The list of imbalance weights
        :return: The gradient of the binary cross entropy loss function.
        """

        z = np.dot(X, coefficients)
        h = self.sigmoid(z)

        if len(imbalance_weights_list) > 0:
            gradient = (1 / len(X)) * np.dot(
                X.transpose(), ((h - y) * imbalance_weights_list)
            )
        else:
            gradient = (1 / len(X)) * np.dot(X.transpose(), (h - y))

        return gradient

    def fit_logistic_regression_gradient_descent(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        iterations: int = 10000,
        learning_rate: float = 0.001,
        handle_imbalance: bool = False,
        imbalance_weights: list = [],
    ) -> np.ndarray:
        """
        Performs Logistic Regression gradient descent solution.

        Gradient formula-  1/n * Xt . (h - y), where h is the sigmoid function of z, where z = (X . coefficients_matrix). Xt is transpose of X.
        Formula to calculate weights- if not provided: weight for class 0 = (n_total_samples / (n_classes * n_samples_class_0)), weight for class 1 = (n_total_samples / (n_classes * n_samples_class_1))
        :param dataset: The dataset to perform regression on.
        :param target_column: The column in the dataset to predict.
        :param iterations: The number of iterations to run the gradient descent algorithm.
        :param learning_rate: The learning rate of the gradient descent algorithm.
        :param handle_imbalance: Whether to handle class imbalance in the dataset.
        :param imbalance_weights: The weights to use for class imbalance.
        :return: The coefficients of the logistic regression model.
        :rtype: np.ndarray
        """

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Dataset must be a Pandas DataFrame -> {type(dataset)} has been provided by user"
            )

        X, y = self._feature_and_target_dataset(dataset, target_column)
        self.X_cols_order = dataset.drop(columns=[target_column]).columns

        X_with_intercept = self._add_intercept(X)
        X_transpose = X_with_intercept.transpose()

        n_samples, n_features = X_with_intercept.shape

        self.logisitic_regression_coefficients = np.zeros(n_features)

        # Calculate imbalance weights
        if handle_imbalance and len(imbalance_weights) == 0:
            imbalance_weights = self._calculate_imbalance_weights(target_column=y)
            print(f"Imbalance weights: {type(imbalance_weights)}")

        for _ in range(iterations):
            # Gradient = (1/n)*Xt . (h - y), where h is the sigmoid function of z, where z = (X . coefficients_matrix). Xt is transpose of X.

            gradient = self._binary_cross_entropy_gradient_calculation(
                X=X_with_intercept,
                y=y,
                coefficients=self.logisitic_regression_coefficients,
                imbalance_weights_list=imbalance_weights,
            )

            updated_coefficient = (
                self.logisitic_regression_coefficients - learning_rate * gradient
            )

            if np.any(np.isnan(updated_coefficient)):
                print(" ++> Gradient descent failed to converge.")
                break
            else:
                self.logisitic_regression_coefficients = updated_coefficient

        return self.logisitic_regression_coefficients

    def predict(
        self,
        dataset: np.ndarray,
        coefficients: np.ndarray = None,
        classification_threshold: float = 0.5,
        classification_type: str = "binary logistic",
    ) -> np.ndarray:

        # Quality check
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Dataset must be a Pandas DataFrame -> {type(dataset)} has been provided by user"
            )
        try:
            dataset = dataset[self.X_cols_order]
        except KeyError:
            raise KeyError(
                f"Dataset must have same columns as training dataset -> {dataset.columns} has been provided by user"
            )

        if coefficients is None:
            if (
                hasattr(self, "logisitic_regression_coefficients")
                and classification_type == "binary logistic"
            ):
                coefficients = self.logisitic_regression_coefficients
            else:
                raise AttributeError("Model is not fitted. Please fit the model first.")

        # logic for predicting

        X = dataset.to_numpy()
        X = self._add_intercept(X)
        # Applying sigmoid function and then thresholding
        y_pred = self.binary_threshold(
            self.sigmoid(np.dot(X, coefficients)), threshold=classification_threshold
        )

        return y_pred


@dataclass
class KNNClassifier:
    """
    KNN Classifier
    """

    def _feature_and_target_dataset(
        self, dataset: pd.DataFrame, target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:

        return (
            dataset.drop(columns=[target_column]).to_numpy(),
            dataset[target_column].to_numpy(),
        )

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray, p: int = 3) -> float:
        return (np.sum(np.abs(x1 - x2) ** p)) ** (1 / p)

    def fit_knn_classifier(
        self,
        dataset: pd.DataFrame,
        target_column: str,
    ) -> np.ndarray:
        """
        Fit KNN Classifier
        :param dataset: The dataset to train the model on
        :param target_column: The column in the dataset to predict
        :return: The numpy array of the feature and target dataset
        """

        X, y = self._feature_and_target_dataset(dataset, target_column)
        self.X_cols_order = dataset.drop(columns=[target_column]).columns
        self.X = X
        self.y = y
        return self

    def _predict(
        self, x: np.ndarray, k: int, distance_metric: str, minkowski_p: int = None
    ) -> np.ndarray:
        """
        Helper function to predict.

        :param x: The point to predict (this is a numpy array representing one row of the dataset)
        :param k: The number of neighbors to consider
        :param distance_metric: The distance metric to use
        :return: The predicted class

        """

        if distance_metric == "euclidean":
            distances = [self._euclidean_distance(x, x2) for x2 in self.X]

        elif distance_metric == "manhattan":
            distances = [self._manhattan_distance(x, x2) for x2 in self.X]

        elif distance_metric == "minkowski" and minkowski_p is not None:
            distances = [self._minkowski_distance(x, x2, minkowski_p) for x2 in self.X]
        elif distance_metric == "minkowski" and minkowski_p is None:
            distances = [self._minkowski_distance(x, x2) for x2 in self.X]

        else:
            raise ValueError(
                f"Distance metric must be one of 'euclidean', 'manhattan', 'minkowski' -> {distance_metric} has been provided by user"
            )

        # get the indices of the k smallest distances
        smallest_distances = np.argsort(distances)[:k]

        # get the y values of the k smallest distances
        y_closest_neighbors = self.y[smallest_distances]

        # get the most common y value
        return Counter(y_closest_neighbors).most_common(1)[0][0]

    def predict(
        self,
        dataset: pd.DataFrame | np.ndarray,
        k: int = 3,
        distance_metric: str = "euclidean",
        minkowski_p: int = None,
    ):
        """
        Predict the class of the dataset
        :param dataset: The dataset to predict
        :param k: The number of neighbors to consider
        :param distance_metric: The distance metric to use
        :return: The predicted class

        """

        if not isinstance(dataset, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"Dataset must be a Pandas DataFrame -> {type(dataset)} has been provided by user"
            )

        if isinstance(dataset, pd.DataFrame):
            try:
                dataset = dataset[self.X_cols_order].to_numpy()
            except KeyError:
                raise KeyError(
                    f"Dataset must have same columns as training dataset -> {dataset.columns} has been provided by user"
                )

        return np.array(
            [self._predict(x, k, distance_metric, minkowski_p) for x in dataset]
        )


@dataclass
class SupportVectorClassifier:
    """
    Build a Binary Support Vector Classifier

    Current capability is for Soft-Margin Support Vector Classifier with Gradient Ascent.

    We will be using quadratic loss, not hinge loss.
    Intro to hinge loss for Linear SVM:
    We need to minimize the loss function L = (1/2) * ||w||^2 + C * sum(max(0, 1 - y ( w^T * x + b)), where w is the weight vector(matrix of coefficients), x is the feature vector and y is the target vector.
    Gradient of the cost function is:
    When y(w^T * x + b) < 1, then max(0, 1 - y ( w^T * x + b) will be > 0:
        dL/dw = w - C * sum(y*x)
        dL/db = -C * sum(y)

    When y(w^T * x + b) >= 1, then max(0, 1 - y ( w^T * x + b) will be = 0:
        dL/dw = w
        dL/db = 0

        This is the primal form of the SVM.

    For Quadratic Loss:
    minimize    (1/2) * ||w||^2 + C * sum(ξ_i^2)
    subject to  y_i(w^T * x_i + b) >= 1 - ξ_i,  for all i
                ξ_i >= 0,  for all i
    Where:

    w is the weight vector
    b is the bias term
    ξ_i are the slack variables
    C is the regularization parameter
    (x_i, y_i) are the training samples and their labels

    maximize    sum(a_i) - (1/2) * sum(sum(a_i * a_j * y_i * y_j * K(x_i, x_j))) - (1/4C) * sum(a_i^2)
    subject to  a_i >= 0,  for all i

    Where:

    a_i are the Lagrange multipliers
    K(x_i, x_j) is the kernel function

    The gradient of the dual objective function with respect to a_i is: ∂L/∂a_i = 1 - y_i * sum(a_j * y_j * K(x_i, x_j)) - (1/2C) * a_i ; (this is equivanlent to -1 + y_i * Σ(a_j * y_j * K(x_i, x_j)) + 2C * a_i)

    Gradient Ascent Update Rule:
    a_i := a_i + learning_rate * (1 - y_i * sum(a_j * y_j * K(x_i, x_j)) - (1/2C) * a_i)



    3)Kernel logic- if kernel is 'linear', then we have the following equation:
        kernel tranformed matrix = np.dot(x, x.T)

    if kernel is 'rbf', then we have the following equation:
        kernel tranformed matrix = np.exp(-gamma * (x - x.T) ** 2)

    if kernel is 'polynomial', then we have the following equation:
        kernel tranformed matrix = (np.dot(x, x.T) + 1) ** degree

    Args:
        kernel (str): Type of kernel to use ('linear', 'poly', or 'rbf').
        degree (int): Degree of the polynomial kernel (if used).
        sigma (float): Parameter for the Gaussian kernel (if used).
        epoches (int): Number of training epochs.
        learning_rate (float): Learning rate for gradient ascent.
        C (float): Regularization parameter.
        c (float): Constant term in the polynomial kernel.

    """

    kernel: Literal["linear", "poly", "rbf"] = "rbf"
    degree: int = 2
    sigma: float = 0.1
    epoches: int = 10000
    learning_rate: float = 0.0001
    C: float = 1.0
    c: float = 1.0

    alpha: np.ndarray = field(init=False, default=None)
    b: float = field(init=False, default=0.0)
    X: np.ndarray = field(init=False, default=None)
    y: np.ndarray = field(init=False, default=None)
    kernel_func: Callable = field(init=False)

    def __post_init__(self):
        """Initialize the kernel function based on the chosen kernel type."""
        if self.kernel == "linear":
            self.kernel_func = self.linear_kernel
        elif self.kernel == "poly":
            self.kernel_func = self.polynomial_kernel
        elif self.kernel == "rbf":
            self.kernel_func = self.gaussian_kernel
        else:
            raise ValueError("Kernel must be 'linear', 'poly', or 'rbf'")

    def linear_kernel(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the linear kernel matrix.

        Formula: K(x, z) = x^T * z
        Intuition: Simple dot product of feature vectors, measuring their similarity in the original space.

        Args:
            X (np.ndarray): First input matrix.
            Z (np.ndarray): Second input matrix.

        Returns:
            np.ndarray: Kernel matrix.
        """
        return np.dot(X, Z.T)

    def polynomial_kernel(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the polynomial kernel matrix.

        Formula: K(x, z) = (c + x^T * z)^degree
        Intuition: Allows for higher-order feature interactions, capturing non-linear relationships.

        Args:
            X (np.ndarray): First input matrix.
            Z (np.ndarray): Second input matrix.

        Returns:
            np.ndarray: Kernel matrix.
        """
        return (self.c + np.dot(X, Z.T)) ** self.degree

    def gaussian_kernel(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian (RBF) kernel matrix.

        Formula: K(x, z) = exp(-||x - z||^2 / (2 * sigma^2)),
        where ||x - z||^2 = ||x||^2 + ||z||^2 - 2 * x^T * z
        Intuition: Measures similarity based on distance in feature space, effective for complex non-linear boundaries.

        Args:
            X (np.ndarray): First input matrix.
            Z (np.ndarray): Second input matrix.

        Returns:
            np.ndarray: Kernel matrix.
        """
        # Compute pairwise squared Euclidean distances
        squared_dists = (
            np.sum(X**2, axis=1).reshape(-1, 1)
            + np.sum(Z**2, axis=1)
            - 2 * np.dot(X, Z.T)
        )
        return np.exp(-squared_dists / (2 * self.sigma**2))

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM model using the dual formulation.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        self.X = X
        self.y = y
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        ones = np.ones(n_samples)

        # Precompute kernel matrix
        # Formula: K_ij = K(x_i, x_j)
        # Intuition: Kernel matrix represents similarities between all pairs of training points
        kernel_matrix = self.kernel_func(X, X)

        # Precompute y_i * y_j * K(x_i, x_j)
        # Intuition: This term appears in the dual formulation and gradient calculation
        y_mul_kernel = np.outer(y, y) * kernel_matrix

        for _ in range(self.epoches):
            # Gradient of dual objective: ∇L(α) = 1 - y * (α * y)^T * K
            # Intuition: We're maximizing the margin by adjusting the alphas
            gradient = ones - y_mul_kernel.dot(self.alpha)

            # Update alphas using gradient ascent
            self.alpha += self.learning_rate * gradient

            # Project alphas onto the feasible set: 0 <= α_i <= C
            # Intuition: This enforces the box constraint in soft-margin SVM
            self.alpha = np.clip(self.alpha, 0, self.C)

        # Compute bias term using support vectors (points with 0 < α_i < C)
        # Formula: b = y_s - Σ_i α_i * y_i * K(x_i, x_s) for any support vector x_s
        # Intuition: The bias shifts the decision boundary to its correct position
        support_vector_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C))[
            0
        ]
        if len(support_vector_indices) > 0:
            self.b = np.mean(
                [
                    y[i] - np.sum(self.alpha * y * kernel_matrix[:, i])
                    for i in support_vector_indices
                ]
            )
        else:
            self.b = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the input data.

        Formula: y = sign(Σ_i α_i * y_i * K(x_i, x) + b)
        Intuition: Classification is based on which side of the decision boundary a point falls

        Args:
            X (np.ndarray): Input data features.

        Returns:
            np.ndarray: Predicted labels (-1 or 1).
        """
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the given data.

        Args:
            X (np.ndarray): Input data features.
            y (np.ndarray): True labels.

        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function values for the input data.

        Formula: f(x) = Σ_i α_i * y_i * K(x_i, x) + b
        Intuition: Measures the signed distance from the decision boundary

        Args:
            X (np.ndarray): Input data features.

        Returns:
            np.ndarray: Decision function values.
        """
        return np.dot(self.alpha * self.y, self.kernel_func(self.X, X)) + self.b


@dataclass
class DecisionTreeClassifier:
    """
    Decision Tree Classifier using gini index.

    """

    max_depth: int = 7
    min_samples_split: int = 10
    tree_dict: dict = field(default_factory=dict)

    def _gini(self, x_column: np.ndarray, y_column: np.ndarray) -> float:
        """
        Calculate gini index for one column.

        Args:
            x_column (np.ndarray): Single column of the dataset for which we want to calculate the gini index
            y_column (np.ndarray): Target column of the dataset for which we want to calculate the gini index

        Returns:
            float: gini index
        """

        unique_classes, unique_target = np.unique(x_column), np.unique(y_column)
        total_samples = len(x_column)

        # Calculate gini index for each class
        # Formula: gini = 1 - (p1^2 + p2^2 + ... + pk^2)

        final_gini_index = None
        for class_ in unique_classes:
            filtered_target, length = y_column[x_column == class_], len(
                y_column[x_column == class_]
            )
            probability = np.sum(
                [
                    (
                        len(filtered_target[filtered_target == target])
                        / len(filtered_target)
                    )
                    ** 2
                    for target in unique_target
                ]
            )
            # print(f"{class_} -> {length} -> {filtered_target} -> {[(len(filtered_target[filtered_target == target]) / len(filtered_target))**2 for target in unique_target]}")

            gini_index = 1 - probability
            class_weight = length / total_samples

            final_gini_index = (
                final_gini_index + (class_weight * gini_index)
                if final_gini_index
                else class_weight * gini_index
            )

        # print(f"Final Gini Index: {final_gini_index}")

        return final_gini_index

    def _gini_for_all_columns(self, X: np.ndarray, y_column: np.ndarray) -> List[float]:
        """
        Calculate gini index for all columns in a dataset, using the _gini function

        Args:
            X (np.ndarray): Dataset for which we want to calculate the gini index
            y_column (np.ndarray): Target column of the dataset for which we want to calculate the gini index

        Returns:
            float: gini index
        """

        return [self._gini(X[:, i], y_column) for i in range(X.shape[1])]

    def _threshold_split(self, x_column: np.ndarray, y_column: np.ndarray) -> float:
        """
        Given a column,get the splitting point for the column.
        This is done by sort the values in the column, splitting it into 2 parts (Left and Right based on a threshold),calculate the gini index for the column based on this split.
        Now do this for all thresholds in the column, and return the minimum gini index.

        Args:
            x_column (np.ndarray): Single column of the dataset for which we want to calculate the gini index
            y_column (np.ndarray): Target column of the dataset for which we want to calculate the gini index

        Returns:
            float: Splitting point (Threshold)
        """

        sorted_column = np.sort(x_column)
        unique_values = np.unique(sorted_column)

        current_threshold, current_gini = None, np.inf
        for i, value in enumerate(unique_values):
            if i <= len(unique_values) - 2:
                # print(f"Value {i}: {value}")
                middle_value = (value + unique_values[unique_values > value][0]) / 2
                x_copy = np.copy(x_column)

                x_copy[x_copy > middle_value] = 1
                x_copy[x_copy <= middle_value] = 0

                gini_index = self._gini(x_copy, y_column)

                # print(f"Middle Value: {middle_value} -> Gini Index: {gini_index}")

                if gini_index < current_gini:
                    current_threshold = middle_value
                    current_gini = gini_index
                    # print(
                    #     f"Lowest gini and threshold: {current_gini} -> {current_threshold}"
                    # )

        print(f"Lowest gini and threshold: {current_gini} -> {current_threshold}")

        return current_threshold

    def _split_data_based_on_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_col_with_lowest_gini: int,
        threshold_value: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data based on the threshold.

        Args:
            X (np.ndarray): Dataset for which we want to calculate the gini index
            y_column (np.ndarray): Target column of the dataset for which we want to calculate the gini index
            feature_col_with_lowest_gini (int): Column with the lowest gini index
            threshold_value (float): Threshold value for the column

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Left and right feature and target datasets

        """

        x_copy = np.copy(X)
        x_left, x_right = (
            x_copy[x_copy[:, feature_col_with_lowest_gini] <= threshold_value],
            x_copy[x_copy[:, feature_col_with_lowest_gini] > threshold_value],
        )

        y_left, y_right = (
            y[x_copy[:, feature_col_with_lowest_gini] <= threshold_value],
            y[x_copy[:, feature_col_with_lowest_gini] > threshold_value],
        )

        return x_left, x_right, y_left, y_right

    def train_decision_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """
        Train the decision tree classifier.
        Steps:
        1. Calculate the gini index for each column in the dataset.
        2. Select the column with the lowest gini index
        3. Calculate best threshold for column to be split at.
        4. Split the column based on the threshold- Left and Right- and recursively train the decision tree.

        Rules for leaf nodes:
        - If all the samples in the node are of the same class
        - If max depth is reached
        - if min samples is reached


        Args:
            X (np.ndarray): Input Training data
            y (np.ndarray): Input Training Target column
        """

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        y = y.astype(np.int32)

        print(f"Depth: {depth}")

        n_samples, n_features = X.shape
        n_unique_classes = len(np.unique(y))

        # Return a leaf node class if the following stopping criteria conditions are met
        if (
            (depth >= self.max_depth)
            or (X.shape[0] <= self.min_samples_split)
            or (n_unique_classes == 1)
        ):
            print(f"bincount y: {np.bincount(y)}")
            max_count_y = np.argmax(np.bincount(y))
            return {"class": max_count_y}

        # Find best split
        gini_index_list_for_all_columns = self._gini_for_all_columns(X, y)
        best_feature_index = np.argmin(gini_index_list_for_all_columns)
        best_feature = X[:, best_feature_index]
        best_threshold = self._threshold_split(best_feature, y)

        # Split data based on best threshold
        print(f"Best Threshold: {best_threshold} for feature: {best_feature_index}")
        x_left, x_right, y_left, y_right = self._split_data_based_on_threshold(
            X=X,
            y=y,
            feature_col_with_lowest_gini=best_feature_index,
            threshold_value=best_threshold,
        )

        left = self.train_decision_tree(x_left, y_left, depth + 1)
        right = self.train_decision_tree(x_right, y_right, depth + 1)

        return {
            "feature_index": best_feature_index,
            "threshold": best_threshold,
            "left": left,
            "right": right,
        }

    def fit_decision_tree(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit the decision tree model
        """
        self.tree = self.train_decision_tree(X, y)
        return self.tree

    def _predict_for_one_sample(self, x: np.ndarray, tree: dict) -> int:
        """
        Traverses the decision tree and return prediction for one sample.

        Args:
            x (np.ndarray): One row/sample to predict
            tree (dict): Decision tree

        Returns:
            int: Predicted class
        """
        if "class" in tree:
            return tree["class"]

        if x[tree["feature_index"]] <= tree["threshold"]:
            class_ = self._predict_for_one_sample(x, tree["left"])
        else:
            class_ = self._predict_for_one_sample(x, tree["right"])

        return class_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of the dataset
        :param X: The dataset to predict
        :return: The predicted class
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return np.array([self._predict_for_one_sample(x, self.tree) for x in X])


if __name__ == "__main__":
    model_used = "knn"

    df_train, df_test = Dataset().heart_classication_dataset()
    # onehot__chestpain_asymptomatic, scaler__maxhr

    if model_used == "decision_tree":
        dt = DecisionTreeClassifier(max_depth=5)
        ss = dt.fit_decision_tree(
            X=df_train.drop(columns="remainder__ahd").to_numpy(),
            y=df_train["remainder__ahd"].to_numpy(),
        )

        y_pred = dt.predict(X=df_test.drop(columns="remainder__ahd").to_numpy())
        print(f"y_pred = {y_pred}")
        y_test = df_test["remainder__ahd"].to_numpy()
        y_test = y_test.astype(np.int32)
        print(calculate_binary_classification_accuracy_metrics(y_test, y_pred))

    if model_used == "svc":
        # For svc - convert labels into -1 and 1
        df_train["remainder__ahd"] = df_train["remainder__ahd"].apply(
            lambda x: -1 if x == 0 else 1
        )
        df_test["remainder__ahd"] = df_test["remainder__ahd"].apply(
            lambda x: -1 if x == 0 else 1
        )

        svc = SupportVectorClassifier(kernel="poly")
        svc.train(
            X=df_train.drop(columns="remainder__ahd").to_numpy(),
            y=df_train["remainder__ahd"].to_numpy(),
        )

        y_pred = svc.predict(X=df_test.drop(columns="remainder__ahd").to_numpy())
        y = df_test["remainder__ahd"].to_numpy()
        print(
            f"y_pred {type(y_pred)}  unique = {np.unique(y_pred)} \n y {type(y)}  unique = {np.unique(y)} \n {y_pred}"
        )

        print(calculate_binary_classification_accuracy_metrics(y, y_pred))

    if model_used == "knn":
        model = KNNClassifier()

        model = model.fit_knn_classifier(
            dataset=df_train, target_column="remainder__ahd"
        )

        y_pred = model.predict(
            dataset=df_test, k=10, distance_metric="minkowski", minkowski_p=5
        )

        print(y_pred)

        y = df_test["remainder__ahd"].to_numpy()

        print(calculate_binary_classification_accuracy_metrics(y, y_pred))

    if model_used == "logistic_regression":
        model = LogiticRegression()

        coefficients = model.fit_logistic_regression_gradient_descent(
            dataset=df_train, target_column="remainder__ahd", handle_imbalance=False
        )

        print(f"Coefficients: \n {coefficients}")

        threshold = [0.3, 0.5, 0.7]

        for thresh in threshold:
            print(f"START for threshold = {thresh}")
            y_pred = model.predict(
                dataset=df_test,
                coefficients=coefficients,
                classification_threshold=thresh,
            )

            y = df_test["remainder__ahd"].to_numpy()

            print(calculate_binary_classification_accuracy_metrics(y, y_pred))
