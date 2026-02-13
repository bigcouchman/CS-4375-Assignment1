import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# How is THIS AI????

os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
logging.basicConfig(filename="logs/test1.txt", filemode="w", level=logging.INFO, format="%(message)s")

class CustomLinearRegressor:
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):
            predict_y = np.dot(X, self.weights) + self.bias
            cost = self.mse(y, predict_y)
            self.cost_history.append(cost)

            d_weights = (1/num_samples) * np.dot(X.T, (predict_y - y))
            d_bias = (1/num_samples) * np.sum(predict_y)
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
     
     return X.values, y.values, X.columns

def iterations_mse_plot(cost_history, filename):
     plt.figure(figsize=(10, 6))
     plt.plot(range(len(cost_history)), cost_history)
     plt.xlabel('Iteration #')
     plt.ylabel('MSE')
     plt.title('Iteration # vs MSE Plot')
     plt.savefig(f'plots/{filename}')
     plt.close()

def features_target_plot(X,y,features_name, target_name, filename):
     num_features = min(5, X.shape[1])
     axes = plt.subplots(3, 4, figSize = (15, 10))
     axes = axes.flatten
     if num_features == 1:
          axes = [axes]
     for i in range(11):
          axes[i].scatter(X[:, i], y, s=10)
          axes[i].set_xlabel(features_name[i])
          axes[i].set_ylabel(target_name)
          axes[i].set_title(f'{features_name[i]} vs {target_name}')
     axes[11].axis('off')
     plt.tight_layout()
     plt.savefig(f'plots/{filename}')
     plt.close()

if __name__ == "__main__":
     wines = fetch_ucirepo(id = 186)
     x_frame = wines.data.features
     y_frame = wines.data.targets

     column_target = y_frame.columns[0]
     df = pd.concat([x_frame, y_frame], axis = 1)
     
     X, y, features_names = data_preproc(df, column_target)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)

     list_lr = [0.05, 0.01, 0.001]
     list_iter = [500, 1000, 2000]
     mse_opt = float("inf")
     model_opt = None
     params_opt = None

     for i in list_lr:
          for j in list_iter:
               model = CustomLinearRegressor(learning_rate=i, num_iterations=j)
               model.fit(X_train, y_train)
               predict_train = model.predict(X_train)
               mse_train = model.mse(y_train, predict_train)
               logging.info(f"Iterations {j}: Learning Rate={i}, Training MSE={mse_train}")
               if mse_train < mse_opt:
                    mse_opt = mse_train
                    model_opt = model
                    params_opt = (i, j)
            
     predict_train = model_opt.predict(X_train)
     mse_train = mean_squared_error(y_train, predict_train)
     predict_test = model_opt.predict(X_test)
     mse_test = mean_squared_error(y_test, predict_test)
     exp_var = explained_variance_score(y_test, predict_test)
     r2 = r2_score(y_test, predict_test)

     logging.info(f"Best params={params_opt}")
     logging.info(f"Explained variance={exp_var:.4f}, Bias={model_opt.bias}")
     logging.info(f"Weights={model_opt.weights}")

     iterations_mse_plot(model_opt.cost_history, "p1_mse_iter.png")
     features_target_plot(X_test, y_test, features_name, column_target, "p1_feature_target.png")       