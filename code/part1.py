# Part 1: Linear Regression using Gradient Descent from scratch

import numpy as np
import pandas as pd
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
logging.basicConfig(filename='logs/part1.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # For plotting MSE vs iterations

    def fit(self, X, y):
        # Implement gradient descent for multi-attribute linear regression
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            cost = self.mse(y, y_pred)
            self.cost_history.append(cost)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Log every 100 iterations
            if i % 100 == 0:
                logging.info(f"Iteration {i}: MSE = {cost:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

def preprocess_data(df, target_col):
    # Remove null values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical variables if any
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, X.columns

def plot_mse_vs_iterations(cost_history, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Iterations')
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
    # TODO: Load dataset from UCI, pre-process, split, tune LinearRegressionGD, evaluate, log, plot, answer questions
    pass