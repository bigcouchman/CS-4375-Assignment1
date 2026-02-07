# Part 2: Linear Regression using Scikit-Learn

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import logging
import os
from ucimlrepo import fetch_ucirepo

# Ensure directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set up logging
logging.basicConfig(filename='logs/part2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def preprocess_data(df, target_col):
    # Remove null values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, X.columns

def plot_mse_vs_iterations(model, X_train, y_train, filename):
    # For SGD, we can simulate iterations by partial_fit
    costs = []
    temp_model = SGDRegressor(max_iter=1, learning_rate='constant', eta0=0.01, random_state=42)
    for i in range(100):  # Simulate 100 iterations
        temp_model.partial_fit(X_train, y_train)
        y_pred = temp_model.predict(X_train)
        cost = mean_squared_error(y_train, y_pred)
        costs.append(cost)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Iterations (SGD)')
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_feature_vs_target(X, y, feature_names, target_name, filename):
    n_features = min(5, X.shape[1])
    fig, axes = plt.subplots(1, n_features, figsize=(15, 4))
    if n_features == 1:
        axes = [axes]
    for i in range(n_features):
        axes[i].scatter(X[:, i], y, alpha=0.5)
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel(target_name)
        axes[i].set_title(f'{feature_names[i]} vs {target_name}')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

if __name__ == "__main__":
    # TODO: Load dataset from UCI, pre-process, split, tune SGDRegressor, evaluate, log, plot, answer questions
    pass