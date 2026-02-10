# Assignment 1 Report

## Part 1: Custom Gradient Descent

### Dataset
- Chosen dataset: Wine Quality (UCI ID 186) - https://archive.ics.uci.edu/dataset/186/wine+quality
- Features: 11 physicochemical properties (e.g., fixed acidity, volatile acidity, citric acid, etc.)
- Target: Quality (score between 0-10)

### Pre-processing Steps
- Cleaned: Removed null values and duplicates
- Encoded: Categorical variables (none in this dataset)
- Split: Train/Val/Test (60/20/20) to follow ML best practices
- Scaled: StandardScaler fitted only on train set to avoid data leakage

### Training
- Algorithm: Custom linear regression with batch gradient descent
- Parameters tuned: Learning rate [0.05, 0.01, 0.001], Iterations [500, 1000, 2000]
- Tuning done on validation set (not test) to prevent data leakage
- Logs: See logs/part1_trial*.log (each run creates a new trial file)
- Plots: See plots/part1_trial*_mse_vs_iterations.png and plots/part1_trial*_feature_vs_target.png

### Results
- Best params: lr=0.01, iters=500 (from validation tuning)
- Train MSE: 0.5449
- Test MSE: 0.5139
- R2: 0.3149
- Explained Variance: 0.3199
- Weights: [ 0.03651942 -0.18265162  0.03313027  0.11196006 -0.04536825  0.11152364 -0.12506205 -0.10088688  0.05963639  0.11302823  0.33614918]
- Bias: 5.7522371811046975
- Plots: See plots/part1_trial*_mse_vs_iterations.png and plots/part1_trial*_feature_vs_target.png

### Answer to Question
- Are you satisfied with the best solution? Yes, the model converged (MSE decreases over iterations), and test MSE is reasonable with moderate R2. The model captures some variance in wine quality based on physicochemical properties.

## Part 2: Scikit-Learn SGDRegressor

### Dataset
- Same as Part 1: Wine Quality (UCI ID 186)
- Features: 11 physicochemical properties
- Target: Quality (score between 0-10)

### Pre-processing Steps
- Same as Part 1: Cleaned, encoded, split train/val/test (60/20/20), scaled on train only

### Training
- Algorithm: SGDRegressor from Scikit-Learn (stochastic gradient descent for linear regression)
- Parameters tuned: alpha [0.0001, 0.001, 0.01], learning_rate ['constant', 'optimal', 'invscaling'], eta0 [0.01, 0.001], max_iter [500, 1000, 2000]
- Tuning done on validation set
- Logs: See logs/part2_trial*.log
- Plots: See plots/part2_trial*_mse_vs_iterations.png and plots/part2_trial*_feature_vs_target.png

### Results
- Best params: alpha=0.01, learning_rate='invscaling', eta0=0.001, max_iter=500 (from validation tuning)
- Train MSE: 0.5438
- Test MSE: 0.5118
- R2: 0.3178
- Explained Variance: 0.3205
- Weights: [ 0.03869663 -0.18464064  0.03052353  0.11382545 -0.04348814  0.1124588 -0.1258278  -0.10235081  0.06131342  0.11061398  0.33489839]
- Bias: [5.76791672]
- Plots: See plots/part2_trial*_mse_vs_iterations.png and plots/part2_trial*_feature_vs_target.png

### Answer to Question
- Are you satisfied with the best solution? Yes, SGDRegressor converged and provides similar performance to custom GD, with R2 around 0.32. Using a library simplifies implementation while maintaining good results. I can check by: 1) Convergence monitoring (simulated iterations show MSE decrease), 2) Performance metrics (R² indicates moderate explanatory power), 3) Comparison to Part 1 (similar results validate correctness), 4) Cross-validation could provide additional confidence in robustness.

### Comparison to Part 1
- Both achieve similar R2 (~0.32), showing SGD and batch GD perform comparably on this dataset
- SGD is faster and uses stochastic updates, while custom GD uses batch updates
- Library implementation is more robust with built-in optimizations