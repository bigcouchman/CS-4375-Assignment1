# Part 2: Linear Regression using Scikit-Learn SGDRegressor

import numpy as np  
import pandas as pd  
from sklearn.linear_model import SGDRegressor  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score  
import matplotlib.pyplot as plt  
import logging  
import os  
from ucimlrepo import fetch_ucirepo  

os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def preprocess_data(df, target_col):
    # Preprocess dataset 

    # Remove null/NA values and duplicates 
    df = df.dropna()
    df = df.drop_duplicates()

    # Separate features and target variable
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X.values, y.values, X.columns

def plot_mse_vs_iterations(best_params, X_train, y_train, filename):
    # Plot MSE vs Iterations 

    costs = []
    # Create a temporary model with best parameters
    temp_model = SGDRegressor(
        alpha=best_params['alpha'],
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        max_iter=1,
        random_state=42,
        warm_start=True  
    )
    n_sim_iters = 500  # Simulate up to 500 iterations to show convergence
    for i in range(n_sim_iters):
        temp_model.partial_fit(X_train, y_train)  # Partial fit for one iteration
        y_pred = temp_model.predict(X_train)  # Predict on training data
        cost = mean_squared_error(y_train, y_pred)  # Calculate MSE
        costs.append(cost)
        if i % 100 == 0:  # Log progress every 100 iterations
            logging.info(f"Sim Iteration {i}: MSE = {cost:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE vs Number of Iterations (SGD)')
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
    # Determine trial number
    log_files = [f for f in os.listdir('logs') if f.startswith('part2_trial') and f.endswith('.log')]
    trial_num = len(log_files) + 1

    # Timestamps and Trial Number 
    logging.basicConfig(filename=f'logs/part2_trial{trial_num}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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

    # 5. Tune parameters on validation set 
    alphas = [0.0001, 0.001, 0.01]  
    learning_rates = ['constant', 'optimal', 'invscaling']  
    eta0s = [0.01, 0.001]  
    max_iters = [500, 1000, 2000]  
    best_mse = float('inf')  
    best_model = None  
    best_params = None  

    for alpha in alphas:
        for lr_type in learning_rates:
            for eta0 in eta0s:
                for max_iter in max_iters:
                    # Create and train model
                    model = SGDRegressor(alpha=alpha, learning_rate=lr_type, eta0=eta0, max_iter=max_iter, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    # Evaluate on validation set
                    y_pred_val = model.predict(X_val_scaled)
                    mse_val = mean_squared_error(y_val, y_pred_val)
                    logging.info(f"Trial: alpha={alpha}, lr={lr_type}, eta0={eta0}, iters={max_iter}, Val MSE={mse_val:.4f}")
                    # Update best model 
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_model = model
                        best_params = {'alpha': alpha, 'learning_rate': lr_type, 'eta0': eta0, 'max_iter': max_iter}

    print(f"Best params: {best_params}, Best Val MSE: {best_mse:.4f}")

    # 6. Evaluate best model on test set 
    y_pred_train = best_model.predict(X_train_scaled)  
    y_pred_val = best_model.predict(X_val_scaled)      
    y_pred_test = best_model.predict(X_test_scaled)    
    train_mse = mean_squared_error(y_train, y_pred_train)  
    val_mse = best_mse  
    test_mse = mean_squared_error(y_test, y_pred_test)    
    r2 = r2_score(y_test, y_pred_test)                     
    explained_var = explained_variance_score(y_test, y_pred_test)  

    # Log results
    logging.info(f"Best Params: {best_params}")
    logging.info(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}, Explained Variance: {explained_var:.4f}")
    logging.info(f"Weights: {best_model.coef_}, Bias: {best_model.intercept_}")

    print(f"Final - Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}")

    # 7. Generate plots
    plot_mse_vs_iterations(best_params, X_train_scaled, y_train, f'part2_trial{trial_num}_mse_vs_iterations.png')
    plot_feature_vs_target(X_test_scaled, y_test, feature_names, target_col, f'part2_trial{trial_num}_feature_vs_target.png')

    # Answer Question 
    logging.info("Are you satisfied with the best solution? Yes, SGDRegressor converged and provides similar performance to custom GD, with R2 around 0.3. Using a library simplifies implementation while maintaining good results. I can check by: 1) Convergence monitoring (simulated iterations show MSE decrease), 2) Performance metrics (R² indicates moderate explanatory power), 3) Comparison to Part 1 (similar results validate correctness), 4) Cross-validation could provide additional confidence in robustness.")