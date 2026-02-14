<h1>CS 4375 Assignment 1 Report</h1>
<div>
  <h2>Part 1. Linear Regression using Gradient Descent</h2>
  <h3>Chosen Dataset</h3>
  <ul>
    <li>Dataset: UCI Wine Quality (Id: 186) https://archive.ics.uci.edu/dataset/186/wine+quality</li>
    <li>Features: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol</li>
    <li>Target: Quality (0 - 10)</li>
    <li>We use a seed to randomize part of the data to test on</li>
  </ul>
  <h3>Data pre-processing</h3>
  <ul>
    <li>Remove null and duplicate values</li>
    <li>Encode Categorical variables to Numerical variables</li>
    <li>Scale appropriate features to standardize train dataset</li>
  </ul>
  <h3>Training</h3>
  <ul>
    <li>Split dataset to (80/20) train/test ratio</li>
    <li>Tune Parameters:
      <ul>
        <li>Learning Rates: [0.01, 0.05, 0.001]</li>
        <li>Number of Iterations: [500, 1000, 2000]</li>
      </ul>
    </li>
    <li>Logs: Review logs/part1_trial#.txt to keep track of learning rates, number of iterations, and MSE</li>
    <li>Plots: Review plots/p1_mse_vs_iteration_trial#.png and plots/p1_features_vs_target_trial#.png for MSE and feature trends</li>
  </ul>
  <h3>Results</h3>
  <ul>
    <li>Optimal parameters: Learning Rate = 0.05, Number of Iterations = 2000</li>
    <li>R^2 = 0.320655</li>
    <li>Training MSE: 0.542747</li>
    <li>Test MSE: 0.509624</li>
    <li>Explained Variance: 0.3214</li>
    <li>Bias: 5.7877</li>
    <li>Weights: [0.07358287, -0.19358273, 0.02227252, 0.1666487, -0.02510952, 0.12427732, -0.15069123, -0.16917101, 0.0850293, 0.12480173, 0.31321147]</li>
  </ul>
  <h3>Are you satisfied that you have found the best solution? Explain.</h3>
  <ul>
    <li>Yes, because as the number of iterations increase, MSE converges to a steady value, and the test MSE is consistent with the result R^2.</li>
  </ul>
</div>
<div>
  <h2>Part 2. Linear Regression using ML library</h2>
  <h3>Chosen Dataset</h3>
  <ul>
    <li>Dataset: UCI Wine Quality (Id: 186)</li>
    <li>Features: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol</li>
    <li>Target: Quality (0 - 10)</li>
    <li>We use a seed to randomize part of the data to test on</li>
  </ul>
  <h3>Data pre-processing</h3>
  <ul>
    <li>Remove null and duplicate values</li>
    <li>Encode Categorical variables to Numerical variables</li>
    <li>Scale appropriate features to standardize train dataset</li>
  </ul>
  <h3>Training</h3>
  <ul>
    <li>Split dataset to (80/20) train/test ratio</li>
    <li>Use SGDRegressor from Scikit-Learn (linear regression gradient descent)</li>
    <li>Tune Parameters:
      <ul>
        <li>Learning Rates: [0.01, 0.05, 0.001]</li>
        <li>Number of Iterations: [500, 1000, 2000]</li>
      </ul>
    </li>
    <li>Logs: Review logs/part2_trial#.txt to keep track of learning rates, number of iterations, and MSE</li>
    <li>Plots: Review plots/p2_mse_vs_iteration_trial#.png and plots/p2_features_vs_target_trial#.png for MSE and feature trends</li>
  </ul>
  <h3>Results</h3>
  <ul>
    <li>Optimal parameters: Alpha: 0.0001, Learning Rate = 'adaptive', eta0 = 0.01, Number of Iterations = 500</li>
    <li>R^2 = 0.320518</li>
    <li>Training MSE: 0.542749</li>
    <li>Test MSE: 0.509726</li>
    <li>Explained Variance: 0.3213</li>
    <li>Bias: 5.7872</li>
    <li>Weights: [0.0743224, -0.1925495, 0.02211341, 0.16721579, -0.02562908, 0.12492398, -0.15111438, -0.17186991, 0.08507857, 0.12545151, 0.31213117]</li>
  </ul>
  <h3>Are you satisfied that the package has found the best solution? How can you check? Explain.</h3>
  <ul>
    <li>Yes, because similar to Part 1, the more iterations there is, the closer the MSE converges to a consistent value. To check, I monitor MSE convergence and use R^2 as a performance metric. I compare my results to Part 1 and while the optimal number of iterations and learning rate are different, the performance metrics and model evaluation are consistent thanks to the additional parameters.</li>
  </ul>
</div>

