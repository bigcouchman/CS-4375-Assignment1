import numpy as np
import pandas as pd
import matplotlib.pyplot
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

logging.basicConfig(
    filename="train_data_log.txt",
    level=logging.INFO,
    format="%(message)s"
)

class LinearReg:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):
            predict_y = np.dot(X, self.weights) + self.bias
            cost = self.mse(y, predict_y)
            d_weights = (1/num_samples) * np.dot(X.T, (predict_y - y))
            d_bias = (1/num_samples) * np.sum(predict_y - y)

            self.weights = self.weights - (self.learning_rate * d_weights)
            self.bias = self.bias - (self.learning_rate * d_bias)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def mse(self, actual_y, predict_y):
        return np.mean((actual_y - predict_y) ** 2)
    
def data_preproc(df, column_target):
    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop(columns = [column_target])
    y = df[column_target]

    for column in X.select_dtypes(include=['object']).columns:
        label = LabelEncoder()
        X[column] = label.fit_transform(X[column])

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(X)
    return scaled_x, y.values, X.columns

if __name__ == "__main__":
    wines = fetch_ucirepo(id=186)

    x_frame = wines.data.features
    y_frame = wines.data.targets

    column_target = y_frame.columns[0]
    df = pd.concat([x_frame, y_frame], axis = 1)

    print("Loaded dataset.")
    print(df.head())

    # Data preprocessing and training/test split
    X, y, features_names = data_preproc(df, column_target)
    X_train, X_test, y_train, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Completed data preprocessing.")
    print(f"Training data: {X_train.shape[0]}")
    print(f"Test data: {X_test.shape[0]}")

    # Create a linear regression model
    list_learning_rates = [0.05, 0.01, 0.001]
    list_num_iterations = [500, 1000, 2000]

    mse_optimal = float("inf")
    model_optimal = None
    params_optimal = None

    for i in list_learning_rates:
        for j in list_num_iterations:
            model = LinearReg(learning_rate=i, num_iterations=j)
            model.fit(X_train, y_train)

            predict_train = model.predict(X_train)
            mse_train = model.mse(y_train, predict_train)
            logging.info(f"Learning Rate = {i}, Iterations = {j}, Train MSE = {mse_train}")
            if mse_train < mse_optimal:
                mse_optimal = mse_train
                model_optimal = model
                params_optimal = (i, j)
    
    print("Best MSE for training: ", mse_optimal)
    print("Best parameters: ", params_optimal)

