# Machine Learning From Scratch - My Implementation Journey

This is my personal project where I implement various machine learning algorithms from scratch! The main goal is to deepen my understanding of how these algorithms work under the hood, rather than just using them through libraries like scikit-learn.

By building these models from scratch, I've gained insights into:
- The underlying mathematics of each algorithm
- How different parameters affect model performance
- Common challenges in implementing ML algorithms
- The importance of proper data preprocessing
- Why certain algorithms work better for specific problems

## What's Implemented So Far

### Regression Models
I've implemented these regression techniques:
- **Linear Regression**
  - Basic Ordinary Least Squares (OLS)
  - Ridge Regression (L2) - both closed-form and gradient descent
  - Lasso Regression (L1) using gradient descent

### Classification Models
These are my implementations of common classifiers:
- **Logistic Regression** with gradient descent
- **K-Nearest Neighbors (KNN)** with multiple distance metrics
- **Support Vector Machine (SVM)** with different kernels
- **Decision Tree** using Gini index

### Clustering
- **K-Means** with random initialization
  - Includes tools for finding optimal k

### Dimensionality Reduction
- **PCA** using the power method
  - Includes visualization of explained variance

## Key Features
Everything is built using mainly NumPy and pandas, with some help from:
- numpy for mathematical operations
- pandas for data handling
- scipy for specific math functions
- matplotlib for visualizations
- scikit-learn (only for metrics and data preprocessing comparisons)

## How I Organized The Code
I used object-oriented programming with Python's dataclasses to keep things organized:
- `regression_models.py`: All regression algorithms
- `classification_models.py`: Classification implementations
- `clustering_models.py`: Clustering methods
- `dimensionality_reduction_models.py`: PCA implementation

## Example Usage

Here's how you can use my implementations:

### Linear Regression Example
```python
# Train a linear regression model
model = LinearRegression()
coefficients = model.fit_least_square_regression(
    dataset=train_data,
    target_column="price"
)

# Make predictions
predictions = model.predict(
    dataset=test_data,
    regression_type="OLS"
)
```

### K-Means Example
```python
# Cluster your data
kmeans = KMeans(n_clusters=3)
kmeans_fit = kmeans.fit_randomized(dataset)
labels = kmeans_fit.predict(dataset)
```

## Datasets Used
I tested these implementations on:
- KC House Price dataset (for regression)
- Heart Disease Classification dataset
- Bike Sharing Demand dataset
- Credit Card Customer dataset
