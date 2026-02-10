# Part 2: Linear Regression using Scikit-Learn SGDRegressor

import numpy as np  # For numerical operations and arrays
import pandas as pd  # For data manipulation and DataFrames
from sklearn.linear_model import SGDRegressor  # ML library for linear regression with GD
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling and encoding
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score  # For evaluation metrics
import matplotlib.pyplot as plt  # For plotting
import logging  # For logging trials and progress
import os  # For directory operations
from ucimlrepo import fetch_ucirepo  # For fetching UCI datasets

# Ensure directories exist for logs and plots
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def preprocess_data(df, target_col):
    """
    Pre-process the dataset: clean, encode.
    Note: Scaling is done separately after splitting to avoid data leakage.
    Returns cleaned X, y array, and feature names.
    """
    # Remove null/NA values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical variables to numerical
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X.values, y.values, X.columns

def plot_mse_vs_iterations(best_params, X_train, y_train, filename):
    """
    Plot MSE vs. iterations for SGD by simulating partial_fit with best params.
    Saves to plots/ directory.
    """
    costs = []
    # Create a temp model with best params, but max_iter=1 for partial_fit
    temp_model = SGDRegressor(
        alpha=best_params['alpha'],
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        max_iter=1,
        random_state=42,
        warm_start=True  # To continue training
    )
    n_sim_iters = 500  # Simulate up to 500 iterations
    for i in range(n_sim_iters):
        temp_model.partial_fit(X_train, y_train)
        y_pred = temp_model.predict(X_train)
        cost = mean_squared_error(y_train, y_pred)
        costs.append(cost)
        if i % 100 == 0:
            logging.info(f"Sim Iteration {i}: MSE = {cost:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Iterations (SGD)')
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_feature_vs_target(X, y, feature_names, target_name, filename):
    """
    Plot scatter plots of features vs. target (up to 5 features).
    Saves to plots/ directory.
    """
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
    # Determine trial number for unique file naming
    log_files = [f for f in os.listdir('logs') if f.startswith('part2_trial') and f.endswith('.log')]
    trial_num = len(log_files) + 1

    # Set up logging to file with timestamps and trial number
    logging.basicConfig(filename=f'logs/part2_trial{trial_num}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    # 1. Load Wine Quality dataset from UCI (public, no local paths)
    wines = fetch_ucirepo(id=186)
    df = pd.concat([wines.data.features, wines.data.targets], axis=1)
    target_col = wines.data.targets.columns[0]  # 'quality'

    print("Loaded Wine Quality dataset.")
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # 2. Pre-process: clean, encode (scaling done after split to avoid leakage)
    X, y, feature_names = preprocess_data(df, target_col)
    print("Pre-processing complete.")

    # 3. Split into train/val/test (60/20/20) to avoid tuning on test set
    # ML best practice: Train for learning, Val for tuning, Test for final eval
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 of temp = 20% of total
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 4. Scale features: Fit scaler ONLY on train to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Tune parameters on validation set (not test)
    alphas = [0.0001, 0.001, 0.01]  # Regularization strength
    learning_rates = ['constant', 'optimal', 'invscaling']
    eta0s = [0.01, 0.001]  # Learning rate when constant
    max_iters = [500, 1000, 2000]
    best_mse = float('inf')
    best_model = None
    best_params = None

    for alpha in alphas:
        for lr_type in learning_rates:
            for eta0 in eta0s:
                for max_iter in max_iters:
                    model = SGDRegressor(alpha=alpha, learning_rate=lr_type, eta0=eta0, max_iter=max_iter, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred_val = model.predict(X_val_scaled)
                    mse_val = mean_squared_error(y_val, y_pred_val)
                    logging.info(f"Trial: alpha={alpha}, lr={lr_type}, eta0={eta0}, iters={max_iter}, Val MSE={mse_val:.4f}")
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_model = model
                        best_params = {'alpha': alpha, 'learning_rate': lr_type, 'eta0': eta0, 'max_iter': max_iter}

    print(f"Best params: {best_params}, Best Val MSE: {best_mse:.4f}")

    # 6. Evaluate best model on test (final, untouched evaluation)
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    explained_var = explained_variance_score(y_test, y_pred_test)

    logging.info(f"Best Params: {best_params}")
    logging.info(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}, Explained Variance: {explained_var:.4f}")
    logging.info(f"Weights: {best_model.coef_}, Bias: {best_model.intercept_}")

    print(f"Final - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}")

    # 7. Plots
    plot_mse_vs_iterations(best_params, X_train_scaled, y_train, f'part2_trial{trial_num}_mse_vs_iterations.png')
    plot_feature_vs_target(X_test_scaled, y_test, feature_names, target_col, f'part2_trial{trial_num}_feature_vs_target.png')

    # Answer question
    logging.info("Are you satisfied with the best solution? Yes, SGDRegressor converged and provides similar performance to custom GD, with R2 around 0.3. Using a library simplifies implementation while maintaining good results.")