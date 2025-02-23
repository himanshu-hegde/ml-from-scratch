# File Name: regression_models.py

import numpy as np
import pandas as pd
import scipy as sp
from pprint import pprint
from dataclasses import dataclass, field
from typing import Union
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(0)


def calculate_metrics(actual, predicted):
    """
    Calculates various metrics to evaluate the performance of a regression model.

    :param actual: The actual values of the target variable.
    :param predicted: The predicted values of the target variable.
    :return: A tuple containing the Mean Absolute Percentage Error (MAPE),
                the Root Mean Squared Error (RMSE), and the R-squared value (R2).
    """
    # Calculate MAPE
    mape = mean_absolute_percentage_error(actual, predicted)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Calculate R2
    r2 = r2_score(actual, predicted)

    # Print the metrics
    print(
        f"Mean Absolute Percentage Error: {mape} \n Root Mean Squared Error: {rmse} \n R-squared: {r2}"
    )

    return mape, rmse, r2


@dataclass
class Dataset:
    """
    A class with multiple datasets.

    """

    # Create a pandas DataFrame with 5 columns
    mock_df1: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "Name": np.random.choice(
                    ["John", "Mary", "David", "Emily", "Michael"], 10
                ),
                "Age": np.random.randint(20, 60, 10),
                "City": np.random.choice(
                    ["New York", "Los Angeles", "Chicago", "Houston", "Seattle"], 10
                ),
                "Country": np.random.choice(
                    ["USA", "Canada", "Mexico", "UK", "Australia"], 10
                ),
                "Score": np.random.uniform(0, 100, 10),
            }
        )
    )

    mock_df2: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "A": np.random.randint(0, 10, 4),
                "B": np.random.randint(0, 5, 4),
                "C": np.random.randint(10, 20, 4),
                "D": np.random.randint(1, 6, 4),
            }
        )
    )

    def _apply_standard_scaler(
        self, df: pd.DataFrame, exclude_cols: list = []
    ) -> Union[pd.DataFrame, ColumnTransformer]:
        """
        Applies StandardScaler to all columns except those in exclude_cols.

        """
        # Identifying columns to scale
        scale_cols = [col for col in df.columns if col not in exclude_cols]

        # Creating ColumnTransformer with StandardScaler applied to relevant columns
        transformer_object = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), scale_cols)],
            remainder="passthrough",  # Keep other columns unchanged
        )

        # Fit and transform the data
        transformed_df = transformer_object.fit_transform(df)

        # Creating a new DataFrame to hold the transformed data
        # ColumnTransformer returns a numpy array, so we need to ensure the columns are properly aligned
        output_df = pd.DataFrame(transformed_df, columns=scale_cols + exclude_cols)

        # Match the original data types
        for col in output_df.columns:
            output_df[col] = output_df[col].astype(df[col].dtype)

        return output_df, transformer_object

    # Example usage:
    # df_scaled = apply_standard_scaler(df, exclude_cols=['column_to_exclude_1', 'column_to_exclude_2'])

    def bike_demand_dataset(
        self,
        # bike_demand_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Python VS Code\\ML From Scratch\\Datasets\\bike-sharing-demand\\train.csv",
        bike_demand_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Machine Learning Models From Scratch\\ML From Scratch\\Datasets\\bike-sharing-demand\\train.csv",
    ) -> pd.DataFrame:
        """
        Retrieves and preprocesses the bike demand dataset.

        :param bike_demand_path: The path to the bike demand dataset CSV file.
        :return: The preprocessed bike demand dataset with training and testing sets.
        """
        self.bike_demand = pd.read_csv(bike_demand_path)
        self.bike_demand = self.bike_demand.drop(columns=["datetime"])

        # Split the data into training and testing sets
        self.bike_demand_train, self.bike_demand_test = train_test_split(
            self.bike_demand, test_size=0.2, random_state=24, shuffle=True
        )

        return self

    def kc_housing_dataset(
        self,
        # kc_house_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Python VS Code\\ML From Scratch\\Datasets\\kc_house_data_regression.csv",
        kc_house_path: str = "C:\\Users\\himan\\OneDrive\\Documents\\Study Docs\\Machine Learning Models From Scratch\\ML From Scratch\\Datasets\\kc_house_data_regression.csv",
    ) -> pd.DataFrame:
        """
        Retrieves and preprocesses the kc_house dataset.

        :param kc_house_path: The path to the kc_house dataset CSV file.
        :return: The preprocessed kc_house dataset with training and testing sets.
        """
        self.kc_house = pd.read_csv(kc_house_path)
        self.kc_house = self.kc_house[
            [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "condition",
                "grade",
                "floors",
                "price",
            ]
        ]

        # Split the data into training and testing sets
        self.kc_house_train, self.kc_house_test = train_test_split(
            self.kc_house, test_size=0.2, random_state=24, shuffle=True
        )

        self.kc_house_train, scaler_object = self._apply_standard_scaler(
            df=self.kc_house_train, exclude_cols=["price"]
        )

        self.kc_house_test = scaler_object.transform(self.kc_house_test)
        self.kc_house_test = pd.DataFrame(
            self.kc_house_test, columns=self.kc_house_train.columns
        )
        for col in self.kc_house_test.columns:
            self.kc_house_test[col] = self.kc_house_test[col].astype(
                self.kc_house_train[col].dtype
            )

        return self


@dataclass
class LinearRegression:
    """
    A class for performing Linear Regression.
    Different types of Linear Regression can be implemented by using this class.
    They are:
    1. Ordinary Least Squares (OLS)
    2. Ridge Regression

    """

    dataframe: pd.DataFrame | np.ndarray = field(default=None)
    loss: list = field(default_factory=list)

    def __post_init__(self):
        if (
            not isinstance(self.dataframe, (np.ndarray, pd.DataFrame))
            and self.dataframe
        ):
            raise TypeError(
                f"Dataframe must be a Pandas DataFrame or Numpy Array -> {type(self.dataframe)} has been provided by user"
            )
        if isinstance(self.dataframe, pd.DataFrame):
            self.dataframe_array = self.dataframe.to_numpy()

    def convert_pandas_dataframe_to_numpy_array(self):
        return self.dataframe.to_numpy()

    def inverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        return np.hstack(tup=(np.ones(shape=(X.shape[0], 1)), X))

    def _feature_and_target_dataset(
        self, dataset: pd.DataFrame, target_column: str
    ) -> np.ndarray:
        return (
            dataset.drop(columns=[target_column]).to_numpy(),
            dataset[target_column].to_numpy(),
        )

    def fit_least_square_regression(
        self, dataset: pd.DataFrame, target_column: str
    ) -> np.ndarray:
        """
        Performs Least Square caluclation (OLS).

        Formula: x = (Xt . X)^-1 . Xt . y, where X is input matrix, y is output matrix, t (eg.Xt) is transpose, ^-1 (eg. (X)^-1) is the inverse of X.

        :param dataset: The dataset to perform regression on.
        :param target_column: The column in the dataset to predict.\
        :return: The coefficients of the linear regression model.
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
        X_ols_list = [
            np.linalg.inv(np.dot(X_transpose, X_with_intercept)),
            X_transpose,
            y,
        ]
        self.X_ols_coefficients = reduce(np.matmul, X_ols_list).reshape(-1, 1)

        return self.X_ols_coefficients

    def fit_ridge_regression_closed_form(
        self, dataset: pd.DataFrame, target_column: str, alpha: float = 1
    ) -> np.ndarray:
        """
        Performs Ridge Regression closed form solution.

        Formula: x = (Xt . X + alpha*I)^-1 . Xt . y, where X is input matrix, y is output matrix, t (eg.Xt) is transpose, ^-1 (eg. (X)^-1) is the inverse of X.aplha is the regularization parameter.I is the identity matrix


        """

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Dataset must be a Pandas DataFrame -> {type(dataset)} has been provided by user"
            )

        X, y = self._feature_and_target_dataset(dataset, target_column)
        self.X_cols_order = dataset.drop(columns=[target_column]).columns

        X_with_intercept = self._add_intercept(X)
        X_transpose = X_with_intercept.transpose()

        # Add identity matrix, remove 1 from first row,first column as bias(intercept) does not need regularization
        I = np.eye(X_with_intercept.shape[1])
        I[0, 0] = 0

        X_ridge_list = [
            np.linalg.inv(np.dot(X_transpose, X_with_intercept) + alpha * I),
            X_transpose,
            y,
        ]

        self.ridge_coefficients = reduce(np.matmul, X_ridge_list).reshape(-1, 1)

        return self.ridge_coefficients

    def fit_ridge_regression_gradient_descent(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        iterations: int = 20,
        learning_rate: float = 0.00001,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """
        Performs Ridge Regression gradient descent solution.

        Gradient formula-  (-2/n)*Xt.(y - X.coefficient_matrix) + alpha*coefficient_matrix, where X is input matrix, y is output matrix, t (eg.Xt) is transpose, coefficient_matrix is the current coefficients.

        :param dataset: The dataset to perform regression on.
        :param target_column: The column in the dataset to predict.\
        :return: The coefficients of the linear regression model.
        :rtype: np.ndarray
        """
        X, y = self._feature_and_target_dataset(dataset, target_column)
        self.X_cols_order = dataset.drop(columns=[target_column]).columns

        X_with_intercept = self._add_intercept(X)
        X_transpose = X_with_intercept.transpose()

        n_samples, n_features = X_with_intercept.shape
        self.ridge_coefficient = np.zeros(X_with_intercept.shape[1])

        for _ in range(iterations):
            y_pred = np.dot(X_with_intercept, self.ridge_coefficient)

            self.loss.append(np.mean((y - y_pred) ** 2))
            gradient = (-2 / n_samples) * np.dot(
                X_transpose, (y - y_pred)
            ) + alpha * self.ridge_coefficient

            updated_coefficient = self.ridge_coefficient - learning_rate * gradient

            if np.any(np.isnan(updated_coefficient)):
                print(" ++> Gradient descent failed to converge.")
                break
            else:
                self.ridge_coefficient = updated_coefficient

        return self.ridge_coefficient

    def fit_lasso_regression_gradient_descent(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        iterations: int = 20,
        learning_rate: float = 0.00001,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """
        Performs Lasso Regression gradient descent solution.

        Gradient formula-  (-2/n)*Xt.(y - X.coefficient_matrix) + alpha*sign(coefficient_matrix), where X is input matrix, y is output matrix, t (eg.Xt) is transpose, coefficient_matrix is the current coefficients.

        :param dataset: The dataset to perform regression on.
        :param target_column: The column in the dataset to predict.\
        :return: The coefficients of the linear regression model.
        :rtype: np.ndarray
        """
        X, y = self._feature_and_target_dataset(dataset, target_column)
        self.X_cols_order = dataset.drop(columns=[target_column]).columns

        X_with_intercept = self._add_intercept(X)
        X_transpose = X_with_intercept.transpose()

        n_samples, n_features = X_with_intercept.shape
        self.lasso_coefficient = np.zeros(X_with_intercept.shape[1])

        for _ in range(iterations):
            y_pred = np.dot(X_with_intercept, self.lasso_coefficient)
            self.loss.append(np.mean((y - y_pred) ** 2))
            gradient = (-2 / n_samples) * np.dot(
                X_transpose, (y - y_pred)
            ) + alpha * np.sign(self.lasso_coefficient)

            updated_coefficient = self.lasso_coefficient - learning_rate * gradient

            if np.any(np.isnan(updated_coefficient)):
                print(" ++> Gradient descent failed to converge.")
                break
            else:
                self.lasso_coefficient = updated_coefficient

        return self.lasso_coefficient

    def predict(
        self,
        dataset: pd.DataFrame,
        coefficients: np.ndarray = None,
        regression_type: str = "OLS",
    ) -> np.ndarray:

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
            if hasattr(self, "X_ols_coefficients") and regression_type == "OLS":
                coefficients = self.X_ols_coefficients
            elif hasattr(self, "ridge_coefficients") and regression_type == "Ridge":
                coefficients = self.ridge_coefficients
            elif (
                hasattr(self, "ridge_coefficient")
                and regression_type == "Ridge Gradient Descent"
            ):
                coefficients = self.ridge_coefficient
            elif (
                hasattr(self, "lasso_coefficient")
                and regression_type == "Lasso Gradient Descent"
            ):
                coefficients = self.lasso_coefficient
            else:
                raise ValueError("Coefficients not found. Please fit the model first.")

        print(f"Conducting {regression_type} regression...")

        X = dataset.to_numpy()
        X_ones = np.ones(shape=(X.shape[0], 1))
        X_with_intercept = np.hstack(tup=(X_ones, X))

        y_predicted = np.matmul(X_with_intercept, coefficients)

        return y_predicted


if __name__ == "__main__":
    # Load the dataset
    dataset = Dataset().kc_housing_dataset()
    train_data = dataset.kc_house_train

    # Create an instance of the regression class
    regression_model = LinearRegression()

    # Perform least squares regression
    coefficients = regression_model.fit_lasso_regression_gradient_descent(
        dataset=train_data,
        target_column="price",
        iterations=10000,
        learning_rate=0.001,
        alpha=0.01,
    )

    loss = regression_model.loss
    # plt.plot(loss)
    # plt.xlabel('Time')
    # plt.ylabel('Loss')
    # plt.title('Loss over Time')
    # plt.savefig('loss_plot.png')

    # Print the coefficients
    print(f"Coefficients: \n {coefficients} \n Loss: \n {loss}")

    # Get the test data
    test_data = dataset.kc_house_test

    # Extract the target variable
    target_values = test_data["price"].to_numpy()
    test_data = test_data.drop(columns=["price"])

    # Make predictions using the regression model
    predictions = regression_model.predict(
        dataset=test_data, regression_type="Lasso Gradient Descent"
    )

    # Print the target values and predictions
    print(f"Target values: \n {target_values}")
    print(f"Predictions: \n {predictions}")

    print(f"{calculate_metrics(target_values.flatten(), predictions.flatten())}")
