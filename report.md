<h1>CS 4375 Assignment 1 Report</h1>
<div>
  <h2>Part 1. Linear Regression using Gradient Descent</h2>
  <h3>Chosen Dataset</h3>
  <ul>
    <li>Dataset: UCI Wine Quality (Id: 186) https://archive.ics.uci.edu/dataset/186/wine+quality</li>
    <li>Features: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol</li>
    <li>Target: Quality (0 - 10)</li>
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
    <li>Logs: Review logs/part1.txt to keep track of learning rates, number of iterations, and MSE</li>
    <li>Plots: Review plots/mse_vs_iteration_p1.png and plots/features_vs_target_p1.png for MSE and feature trends</li>
  </ul>
  <h3>Results</h3>
  <ul>
    <li>Optimal parameters: Learning Rate = 0.05, Number of Iterations = 2000</li>
    <li>R^2 = 0.320655</li>
    <li>Training MSE: 0.542747</li>
    <li>Test MSE: 0.509624</li>
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
    <li>Logs: Review logs/part2.txt to keep track of learning rates, number of iterations, and MSE</li>
    <li>Plots: Review plots/mse_vs_iteration_p2.png and plots/features_vs_target_p2.png for MSE and feature trends</li>
  </ul>
  <h3>Results</h3>
  <ul>
    <li>Optimal parameters: Alpha: 0.0001, Learning Rate = 'adaptive', eta0 = 0.01, Number of Iterations = 500</li>
    <li>R^2 = 0.320518</li>
    <li>Training MSE: 0.542749</li>
    <li>Test MSE: 0.509726</li>
  </ul>
  <h3>Are you satisfied that the package has found the best solution. How can you check. Explain.</h3>
  <ul>
    <li>Yes, because similar to Part 1, the more iterations there is, the closer the MSE converges to a consistent value. To check, I monitor MSE convergence and use R^2 as a performance metric. I compare my results to Part 1 and see that they are all consistent.</li>
  </ul>
</div>

