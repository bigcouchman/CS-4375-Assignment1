import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class LinearReg:
    def __init__(self, lr=0.01, iters = 1000):
        self.learning_rate = lr
        self.num_iterations = iters
        self.weights = None
        self.bias = None
        self.cost_list = []

    def find_predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def find_mse(self, y_actual, y_predict):
        return np.mean((y_actual - y_predict) ** 2)
    
    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0
        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            cost = self.find_mse(y, y_pred)
            self.cost_list.append(cost)
            grad_weights = (1/samples) * np.dot(X.t, (y_pred - y))
            grad_bias = (1/samples) * np.sum(y_pred - y)
            self.weights = self.weights - (self.learning_rate *grad_weights)
            self.bias = self.bias - (self.learning_rate * grad_bias)

def data_preprocess(df, column_target):
    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop(columns = [column_target])
    y = df[column_target]
    for c in X.select_dtypes(include=['object']).columns:
        labelEnc = LabelEncoder()
        X[c] = labelEnc.fit_transform(X[c])
    return X.values, y.values, X.columns

def iterations_vs_mse(cost_list, filename):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(cost_list)), cost_list)
    plt.xlabel('Iterations')
    plt.ykabek('MSE')
    plt.title('Iteration vs MSE')
    plt.savefig(f'plots/{filename}')
    plt.close()

def features_vs_target(X, y, features_names, target_name, filename):
    num_features = min(5,X.shape[1])
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
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
    wines = fetch_ucirepo(id=186)
    column_target = wines.data.targets.columns[0]
    df = pd.concat([wines.data.features, wines.data.targets], axis=1)
    X, y, feature_names = data_preprocess(df, column_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    list_lr = [0.05, 0.01, 0.001]
    list_iters = [500, 1000, 2000]
    mse_optimal = float("inf")
    model_optimal = None
    params_optimal = None

    for i in list_lr:
        for j in list_iters:
            model = LinearReg(lr=i, iters=j)
            model.fit(X_train, y_train)
            train_y_pred = model.find_predict(X_train)
            mse_train = model.find_mse(y_train, train_y_pred)
            if mse_train < mse_optimal:
                mse_optimal = mse_train
                model_optimal = model
                params_optimal = (i,j)
    
    train_predict - model_optimal.find_predict(X_train)
    mse_train = mean_squared_error(y_train, train_predict)
    test_predict = model_optimal.find_predict(X_test)
    mse_test = mean_squared_error(y_test, test_predict)
    r2 = r2_score(y_test, test_predict)
    exp_var = explained_variance_score(y_test, test_predict)
