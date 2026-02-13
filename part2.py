# CS 4375 Assignment 1 Part 2 By Nguyen Do (NPD220001) and Casey Nguyen (CXN220034)
# This assignment is an implementation of linear regression using gradient descent
# on an UCI Wine quality dataset.

# Import necessary libraries (run pip install -r requirements.txt for dependencies)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from ucimlrepo import fetch_ucirepo

# Create directories to store logs and plots
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Data Preprocessing
def data_preproc(df, column_target):

    # Drop nulls and duplicates, separate features and targets
    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop(columns = [column_target])
    y = df[column_target]

    # Label encoding to encode columns, return features and targets items
    for column in X.select_dtypes(include=['object']).columns:
        label = LabelEncoder()
        X[column] = label.fit_transform(X[column])

    return X.values, y.values, X.columns

# Iterations vs MSE plotting
def iterations_mse_plot(params, X_train, y_train, filename):
    cost_history = []
    # Create a base regressor model
    base_model = SGDRegressor(
        alpha=params['alpha'],
        learning_rate=params['learning_rate'],
        eta0=params['eta0'],
        max_iter=1,
        random_state=42,
        warm_start=True
    )

    # Keep track of MSE when called to plot the graph
    # This reflects to test data reasonably
    total_iterations = params['max_iter']
    for i in range(total_iterations):
        base_model.partial_fit(X_train, y_train)
        predict_y = base_model.predict(X_train)
        cost = mean_squared_error(y_train, predict_y)
        cost_history.append(cost)

    # Graph configuration
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iteration #')
    plt.ylabel('MSE')
    plt.title('Iteration # vs MSE Plot')
    plt.savefig(f'plots/{filename}')
    plt.close()

# Feature and target plotting
def features_target_plot(X, y, features_names, target_name, filename):
    num_features = min(5, X.shape[1])
    # Graph each feature vs its target, totalling to 11 graphs and formatted
    figure, axes = plt.subplots(3, 4, figsize=(16, 10))
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

if __name__ == "__main__":
    # Get Wine Quality dataset from UCI, get features and targets
    wines = fetch_ucirepo(id=186)

    x_frame = wines.data.features
    y_frame = wines.data.targets

    # Form dataframe with features and target
    column_target = y_frame.columns[0]
    df = pd.concat([x_frame, y_frame], axis = 1)

    # Data preprocessing and training/test split (80/20)
    X, y, features_names = data_preproc(df, column_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.basicConfig(filename="logs/part2.txt", filemode= 'w', level=logging.INFO, format="%(message)s")
    
    # Scale dataset to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a linear regression model with parameter options
    list_learning_rates = ['constant','adaptive','optimal','invscaling']
    list_num_iterations = [500, 1000, 2000]
    list_alphas = [0.0001, 0.001, 0.01]
    list_eta0s = [0.01, 0.001]
    mse_optimal = float("inf")
    model_optimal = None
    params_optimal = None

    # Fine tune paramaters using Regressor, finding the best combinations of parameters resulting in lowest MSE
    for i in list_alphas:
        for j in list_learning_rates:
            for k in list_eta0s:
                for m in list_num_iterations:
                    model = SGDRegressor(alpha = i, learning_rate=j, eta0=k, max_iter=m, random_state=42)
                    model.fit(X_train, y_train)
                    train_y_predict = model.predict(X_train)
                    mse_train = mean_squared_error(y_train, train_y_predict)

                    # Continuously train the model and log parameters
                    # If optimal parameters are found, make the model to the best model and keep replacing if needed
                    logging.info(f"Iteration = {m}: alpha: {i}, Learning Rate = {j}, eta0 = {k}, Training MSE = {mse_train:.4f}")
                    if mse_train < mse_optimal:
                        mse_optimal = mse_train
                        model_optimal = model
                        params_optimal = {'alpha': i, 'learning_rate': j, 'eta0': k, 'max_iter': m}

    # Model evaluation on train, test data, and performance metrics
    train_predict = model_optimal.predict(X_train)
    mse_train = mean_squared_error(y_train, train_predict)
    test_predict = model_optimal.predict(X_test)
    mse_test = mean_squared_error(y_test, test_predict)
    r2 = r2_score(y_test, test_predict)
    exp_var = explained_variance_score(y_test, test_predict)

    # Print out model evaluation and metrics
    print("Best parameters: ", params_optimal)
    print("Train MSE: ", mse_optimal)
    print("Test MSE: ", mse_test)
    print("R^2: ", r2)
    print("Explained Variance: ", exp_var)
    print("Bias: ", model_optimal.intercept_)
    print("Weights: ", model_optimal.coef_)

    # Log parameters, metrics, and call plotting functions (at the end)
    logging.info(f"Best parameters: {params_optimal}")
    logging.info(f"Explained variance: {exp_var:.4f}, Bias: {model_optimal.intercept_}")
    logging.info(f"Weights: {model_optimal.coef_}")

    iterations_mse_plot(params_optimal, X_train, y_train, "mse_vs_iterations_p2.png")
    features_target_plot(X_train, y_train, features_names, column_target, "features_vs_target_p2.png")
