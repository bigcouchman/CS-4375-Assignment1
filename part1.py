import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Keep track of logs and plots
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
logging.basicConfig(filename="part1.txt", level=logging.INFO, format="%(message)s")

# Linear regression model
class LinearReg:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = [] # Store MSE per iteration
    
    # Gradient descent function
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):

            # Find predictions
            predict_y = np.dot(X, self.weights) + self.bias
            cost = self.mse(y, predict_y)
            self.cost_history.append(cost)

            # Find gradients
            d_weights = (1/num_samples) * np.dot(X.T, (predict_y - y))
            d_bias = (1/num_samples) * np.sum(predict_y - y)

            # Update parameters
            self.weights = self.weights - (self.learning_rate * d_weights)
            self.bias = self.bias - (self.learning_rate * d_bias)

            # For every 100 iterations, log MSE
            if i % 100 == 0:
                logging.info(f"Iteration {i}: MSE = {cost:.4f}")
    
    # Find predicted output values using weights and MSE
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def mse(self, actual_y, predict_y):
        return np.mean((actual_y - predict_y) ** 2)

# Data Preprocessing
def data_preproc(df, column_target):

    # Drop nulls and duplicates, separate features and targets
    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop(columns = [column_target])
    y = df[column_target]

    # Label encoding to encode columns
    for column in X.select_dtypes(include=['object']).columns:
        label = LabelEncoder()
        X[column] = label.fit_transform(X[column])

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(X)
    return scaled_x, y.values, X.columns

# Iterations vs MSE plotting
def iterations_mse_plot(cost_history, filename):
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
    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    if num_features == 1:
        axes = [axes]
    for i in range(11):
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
    # Get Wine Quality dataset from UCI
    wines = fetch_ucirepo(id=186)

    x_frame = wines.data.features
    y_frame = wines.data.targets

    # Form dataframe with features and target
    column_target = y_frame.columns[0]
    df = pd.concat([x_frame, y_frame], axis = 1)

    print("Loaded dataset.")
    print(df.head())

    # Data preprocessing and training/test split
    X, y, features_names = data_preproc(df, column_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Completed data preprocessing.")
    print(f"Training data: {X_train.shape[0]}")
    print(f"Test data: {X_test.shape[0]}")

    # Create a linear regression model with parameter options
    list_learning_rates = [0.05, 0.01, 0.001]
    list_num_iterations = [500, 1000, 2000]

    mse_optimal = float("inf")
    model_optimal = None
    params_optimal = None

    # Tune parameters to find the best model (I am not sure if i do this right pls reivew)
    for i in list_learning_rates:
        for j in list_num_iterations:
            model = LinearReg(learning_rate=i, num_iterations=j)
            model.fit(X_train, y_train)

            predict_train = model.predict(X_train)
            mse_train = model.mse(y_train, predict_train)
            logging.info(f"Learning Rate = {i}, Iterations = {j}, Train MSE = {mse_train:.4f}")
            if mse_train < mse_optimal:
                mse_optimal = mse_train
                model_optimal = model
                params_optimal = (i, j)
    
    print("Best MSE for training: ", mse_optimal)
    print("Best parameters: ", params_optimal)

    # Model evaluation on test data
    predict_test = model_optimal.predict(X_test)
    mse_test = mean_squared_error(y_test, predict_test)
    r2 = r2_score(y_test, predict_test)

    print("Test MSE: ", mse_test)
    print("Test R^2: ", r2)

    # Plotting 
    iterations_mse_plot(model_optimal.cost_history, "mse_vs_iterations_p1.png")
    features_target_plot(X_train, y_train, features_names, column_target, "features_vs_target_p1.png")

