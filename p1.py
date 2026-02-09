import numpy as np
import pandas as pd
import matplotlib.pyplot
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

class LinearReg:
    def _init_(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):
            predict_y = np.dot(x, self.weights) + self.bias
            cost = self.mse(y, predict_y)
            d_weights = (1/num_samples) * np.dot(x.T, (predict_y - y))
            d_bias = (1/num_samples) * np.sum(predict_y - y)

            self.weights = self.weights - (self.learning_rate * d_weights)
            self.bias = self.bias - (self.learning_rate * d_bias)
    
    def predicting(self, X):
        return np.dot(X, self.weights) + self.bias
    def mse(self, actual_y, predict_y):
        return np.mean((actual_y - predict_y) ** 2)
    
def data_preproc(df, column_target):
    df = df.dropna()
    df = df.drop_duplicates()
    x = df.drop(columns = [column_target])
    y = df[column_target]

    for column in x.select_dtypes(include=['object']).columns:
        label = LabelEncoder()
        x[column] = label.fit_transform(x[column])

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    return scaled_x, y.values, x.columns

if __name__ == "__main__":
    pass
            
