# Assignment 1 Report

## Part 1: Custom Gradient Descent

### Dataset
- Dataset: Wine Quality (UCI ID 186) - https://archive.ics.uci.edu/dataset/186/wine+quality
- Features: 11 features (e.g., fixed acidity, volatile acidity, citric acid, etc.)
- Target: Quality (score between 0-10)

### Pre-processing Steps
- Cleaned: Removed null values and duplicates
- Encoded: Categorical variables 
- Split: Train/Val/Test (80/10/10) 
- Scaled: StandardScaler on train set only

### Training
- Algorithm: Custom linear regression with gradient descent
- Parameters tuned: Learning rate [0.05, 0.01, 0.001], Iterations [500, 1000, 2000]
- Tuning done on validation set
- Logs: See logs/part1_trial2.log
- Plots: See plots/part1_trial2_mse_vs_iterations.png and plots/part1_trial2_feature_vs_target.png

### Results
- Best params: lr=0.05, iters=2000
- Train MSE: 0.5427
- Val MSE: 0.5097
- Test MSE: 0.5095
- R2: 0.3057
- Explained Variance: 0.3066
- Weights: [ 0.07358287 -0.19358273  0.02227252  0.1666487  -0.02510952  0.12427732 -0.15069123 -0.16917101  0.0850293   0.12480173  0.31321147]
- Bias: 5.787729196051344
- Plots: See plots/part1_trial2_mse_vs_iterations.png and plots/part1_trial2_feature_vs_target.png 

### Answer to Question
- Are you satisfied with the best solution? Yes, the model converged (MSE decreases over iterations), and test MSE is reasonable with moderate R2. Our program had some variance due to the features in this dataset. 

## Part 2: Scikit-Learn SGDRegressor

### Dataset
- Same as Part 1: Wine Quality (UCI ID 186)
- Features: 11 properties
- Target: Quality (score between 0-10)

### Pre-processing Steps
- Same as Part 1: Cleaned, encoded, split train/val/test (80/10/10), scaled on train only

### Training
- Algorithm: SGDRegressor from Scikit-Learn
- Parameters tuned: alpha [0.0001, 0.001, 0.01], learning_rate ['constant', 'optimal', 'invscaling'], eta0 [0.01, 0.001], max_iter [500, 1000, 2000]
- Tuning done on validation set
- Logs: See logs/part2_trial2.log
- Plots: See plots/part2_trial2_mse_vs_iterations.png and plots/part2_trial2_feature_vs_target.png

### Results
- Best params: alpha=0.0001, learning_rate='constant', eta0=0.001, max_iter=500 (from validation tuning)
- Train MSE: 0.5439
- Val MSE: 0.5085
- Test MSE: 0.5080
- R2: 0.3079
- Explained Variance: 0.3082
- Weights: [ 0.04664766 -0.19321093  0.01334619  0.16110443 -0.0289664   0.12388341 -0.15218621 -0.11965024  0.08563906  0.11028577  0.33820245]
- Bias: [5.79835407]
- Plots: See plots/part2_trial2_mse_vs_iterations.png and plots/part2_trial2_feature_vs_target.png (feature plot shows all 11 features in 3x4 grid)

### Answer to Question
- Are you satisfied with the best solution? Yes, SGDRegressor converged and provides similar results to custom GD, with R2 around 0.31. Using a library simplifies implementation while maintaining good results. I can check by: 1) Convergence monitoring (Shows MSE decrease), 2) Performance (R² indicates moderate explanatory power), 3) Comparison to Part 1 (similar results).

### Comparison to Part 1
- Both have similar R2 (~0.31), showing SGD and batch GD perform comparably on this dataset
- SGD is faster and uses stochastic updates, while custom GD uses batch updates
- Library implementation is more robust with built-in optimizations