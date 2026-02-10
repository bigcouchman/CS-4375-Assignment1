# CS-4375-Assignment1

## Linear Regression using Gradient Descent

### Overview
This assignment implements linear regression using gradient descent in two parts:
- **Part 1**: Custom implementation from scratch
- **Part 2**: Using Scikit-Learn's SGDRegressor

### Setup Instructions
1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries:
   - `numpy`: Numerical computations
   - `pandas`: Data manipulation
   - `scikit-learn`: ML algorithms and preprocessing
   - `matplotlib`: Plotting
   - `ucimlrepo`: Fetching UCI datasets

2. **Dataset**: Wine Quality dataset from UCI ML Repository (ID 186)
   - Hosted publicly at: https://archive.ics.uci.edu/dataset/186/wine+quality
   - No local dataset files needed - fetched programmatically

### Running the Code
- **Part 1 (Custom GD)**: `python code/part1.py`
- **Part 2 (Scikit-Learn SGD)**: `python code/part2.py`

Both scripts will:
- Load and preprocess the dataset
- Tune hyperparameters on validation set
- Evaluate on test set
- Generate logs and plots
- Print final results

### File Structure
- `code/part1.py`: Custom LinearRegressionGD class
- `code/part2.py`: SGDRegressor implementation
- `logs/`: Trial logs (part1_trial*.log, part2_trial*.log)
- `plots/`: Generated plots (MSE vs iterations, feature vs target)
- `report/report.md`: Comprehensive report with results
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes venv, logs, plots

### Important Notes
- **No Hardcoded Paths**: All paths are relative
- **Public Data**: Dataset fetched from UCI - no local files to submit
- **Trial Versioning**: Each run creates new log/plot files (preserves history)
- **ML Best Practices**: Proper train/val/test splits, no data leakage

### Libraries Used (Part 2)
- `sklearn.linear_model.SGDRegressor`: Stochastic Gradient Descent for linear regression
- `sklearn.preprocessing.StandardScaler`: Feature scaling
- `sklearn.preprocessing.LabelEncoder`: Categorical encoding
- `sklearn.model_selection.train_test_split`: Data splitting
- `sklearn.metrics`: Evaluation metrics (MSE, R², explained variance)