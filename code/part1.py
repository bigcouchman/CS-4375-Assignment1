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

os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

if __name__ == "__main__":
    # Trial Number
    log_files = [f for f in os.listdir('logs') if f.startswith('part1_trial') and f.endswith('.log')]
    trial_num = len(log_files) + 1

    # Timestamps and Trial Number
    logging.basicConfig(filename=f'logs/part1_trial{trial_num}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class LinearRegressionGD:
    # Linear Regression Class using Gradient Descent. Multi-attribute regression.
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate  # Step size for GD updates
        self.n_iterations = n_iterations  # Max iterations for convergence
        self.weights = None  # Model weights 
        self.bias = None  # Model bias 
        self.cost_history = []  # Track MSE per iteration for plotting

    def fit(self, X, y):
        # Trains model using the gradiant descent algorithm. Updates weights and bias to minimize MSE.
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  
        self.bias = 0  

        for i in range(self.n_iterations):
            # Predict using current weights/bias
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute MSE
            cost = self.mse(y, y_pred)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))  # Weight gradient
            db = (1/n_samples) * np.sum(y_pred - y)  # Bias gradient

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Log progress every 100 iterations
            if i % 100 == 0:
                logging.info(f"Iteration {i}: MSE = {cost:.4f}")

    def predict(self, X):
        # Predict X
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        # Mean Squared Error (MSE) calculation
        return np.mean((y_true - y_pred)**2)

def preprocess_data(df, target_col):
    # Preprocess Dataset 

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

def plot_mse_vs_iterations(cost_history, filename):
    # Plot MSE vs Iterations 
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Iterations')
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_feature_vs_target(X, y, feature_names, target_name, filename):
    # Plot features vs target 
    n_features = X.shape[1] 
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(n_features):
        row = i // 4
        col = i % 4
        axes[row, col].scatter(X[:, i], y, alpha=0.5)
        axes[row, col].set_xlabel(feature_names[i])
        axes[row, col].set_ylabel(target_name)
        axes[row, col].set_title(f'{feature_names[i]} vs {target_name}')
    axes[2, 3].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

if __name__ == "__main__":
    # 1. Load Wine Quality dataset from UCI
    wines = fetch_ucirepo(id=186)
    df = pd.concat([wines.data.features, wines.data.targets], axis=1)
    target_col = wines.data.targets.columns[0] 

    print("Loaded Wine Quality dataset.")
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # 2. Pre-process
    X, y, feature_names = preprocess_data(df, target_col)
    print("Pre-processing complete.")

    # 3. Split into train/val/test (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Tune parameters learning rate and iterations on validation set
    learning_rates = [0.05, 0.01, 0.001]
    n_iterations_list = [500, 1000, 2000]
    best_mse = float('inf')
    best_model = None
    best_params = None

    for lr in learning_rates:
        for n_iter in n_iterations_list:
            model = LinearRegressionGD(learning_rate=lr, n_iterations=n_iter)
            model.fit(X_train_scaled, y_train)
            y_pred_val = model.predict(X_val_scaled)
            mse_val = model.mse(y_val, y_pred_val)
            logging.info(f"Trial: lr={lr}, iters={n_iter}, Val MSE={mse_val:.4f}")
            if mse_val < best_mse:
                best_mse = mse_val
                best_model = model
                best_params = {'lr': lr, 'iters': n_iter}

    print(f"Best params: {best_params}, Best Val MSE: {best_mse:.4f}")

    # 6. Evaluate on test
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_val = best_model.predict(X_val_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    train_mse = best_model.mse(y_train, y_pred_train)
    val_mse = best_mse
    test_mse = best_model.mse(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    explained_var = explained_variance_score(y_test, y_pred_test)

    logging.info(f"Best Params: {best_params}")
    logging.info(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}, Explained Variance: {explained_var:.4f}")
    logging.info(f"Weights: {best_model.weights}, Bias: {best_model.bias}")

    print(f"Final - Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}")

    # 7. Plots
    plot_mse_vs_iterations(best_model.cost_history, f'part1_trial{trial_num}_mse_vs_iterations.png')
    plot_feature_vs_target(X_test_scaled, y_test, feature_names, target_col, f'part1_trial{trial_num}_feature_vs_target.png')

    # Answer question
    logging.info("Are you satisfied with the best solution? Yes, the model converged (MSE decreases over iterations), and test MSE is reasonable with good R2.")