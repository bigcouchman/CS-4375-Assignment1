# CS 4375 Assignment 1 Part 1 By Nguyen Do (NPD220001) and Casey Nguyen (CXN220034)
# This assignment is an implementation of linear regression using gradient descent
# on UCI wine quality dataset. Question answered in report.md

# Import necessary libraries (run pip install -r requirements.txt for dependencies)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Create directories to store logs and plots
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Custom linear regression model 
class LinearReg:
    # Create model
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_list = [] # Store MSE per iteration
    
    # Gradient descent function
    def grad_desc(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):

            # Predictions, weights and bias
            y_predict = np.dot(X, self.weights) + self.bias
            cost = self.find_mse(y, y_predict)
            self.cost_list.append(cost)

            grad_weights = (1/num_samples) * np.dot(X.T, (y_predict - y))
            grad_bias = (1/num_samples) * np.sum(y_predict - y)
            self.weights = self.weights - (self.learning_rate * grad_weights)
            self.bias = self.bias - (self.learning_rate * grad_bias)
    
    # Predict output and fine MSE
    def find_predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def find_mse(self, y_actual, y_predict):
        return np.mean((y_actual - y_predict) ** 2)

# Data Preprocessing
def data_preproc(df, column_target):

    # Drop nulls and duplicates and separate features and targets
    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop(columns = [column_target])
    y = df[column_target]

    # Label encoding to encode columns and return features and targets items
    for c in X.select_dtypes(include=['object']).columns:
        label = LabelEncoder()
        X[c] = label.fit_transform(X[c])

    return X.values, y.values, X.columns

# Iterations vs MSE plotting
def iterations_mse_plot(cost_list, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_list)), cost_list)
    plt.xlabel('Iteration #')
    plt.ylabel('MSE')
    plt.title('Iteration # vs MSE Plot')
    plt.savefig(f'plots/{filename}')
    plt.close()

# Feature and target plotting
def features_target_plot(X, y, features_names, target_name, filename):
    num_features = min(5, X.shape[1])
    # Graph each feature vs its target, totalling to 11 graphs and formatted
    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    # Build graph layout (3 rows, each with 4 graphs, 11 graphs total)
    if num_features == 1:
        axes = [axes]
    for i in range(min(11, X.shape[1])):
        axes[i].scatter(X[:, i], y, s=10)
        axes[i].set_xlabel(features_names[i])
        axes[i].set_ylabel(target_name)
        axes[i].set_title(f'{features_names[i]} vs {target_name}')
    axes[11].axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

# Main function
if __name__ == "__main__":
    # Get Wine Quality dataset from UCI, geat features and targets
    wines = fetch_ucirepo(id=186)
    column_target = wines.data.targets.columns[0]
    df = pd.concat([wines.data.features, wines.data.targets], axis = 1)

    # Data preprocessing and training/test split (80/20), can be adjusted
    X, y, features_names = data_preproc(df, column_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Every time you run the code, a new log file and plots are created with trial numbers :)
    track_log = [f for f in os.listdir('logs') if f.startswith('part1_trial') and f.endswith('.txt')]
    iteration_num = len(track_log) + 1
    logging.basicConfig(filename=f"logs/part1_trial{iteration_num}.txt", level=logging.INFO, format="%(message)s")
    
    # Scale dataset to avoid leak
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a linear regression model with parameter options
    # More values of parameters can be added to find further improvements
    learning_rate_list = [0.05, 0.01, 0.001]
    iterations_num_list = [500, 1000, 2000]
    mse_optimal = float("inf")
    model_optimal = None
    params_optimal = None

    # Fine tune parameters and find the best combinations of parameters
    for i in learning_rate_list:
        for j in iterations_num_list:
            model = LinearReg(learning_rate=i, num_iterations=j)
            model.grad_desc(X_train, y_train)
            train_y_predict = model.find_predict(X_train)
            mse_train = model.find_mse(y_train, train_y_predict)

            # Continuously train the model and log parameters
            # If optimal parameters are found, make the model to the best model and keep replacing if needed
            logging.info(f"Iterations = {j}: Learning Rate = {i}, Training MSE = {mse_train:.4f}")
            if mse_train < mse_optimal:
                mse_optimal = mse_train
                model_optimal = model
                params_optimal = (i, j)
    
    # Model evaluation on training and test data, and performance metrics (I use test data)
    train_predict = model_optimal.find_predict(X_train)
    mse_train = mean_squared_error(y_train, train_predict)
    test_predict = model_optimal.find_predict(X_test)
    mse_test = mean_squared_error(y_test, test_predict)
    r2 = r2_score(y_test, test_predict)
    exp_var = explained_variance_score(y_test, test_predict)
    
    # Print evaluation for model (Check logs/ for additional metrics!)
    print("Train data MSE: ", mse_train)
    print("Test data MSE: ", mse_test)
    print("R^2: ", r2)

    # Log metrics, parameters, and call plotting functions (at the end)
    logging.info(f"Best parameters (Learning rate, Iterations): {params_optimal}")
    logging.info(f"Train MSE: {mse_optimal:.4f}, Test MSE: {mse_test:.4f}")
    logging.info(f"Explained variance: {exp_var:.4f}, Bias: {model_optimal.bias:.4f}, R^2: {r2:.4f}")
    logging.info(f"Weights: {model_optimal.weights}")
    
    iterations_mse_plot(model_optimal.cost_list, f"p1_mse_vs_iterations_trial{iteration_num}.png")
    features_target_plot(X_train, y_train, features_names, column_target, f"p1_features_vs_target_trial{iteration_num}.png")

