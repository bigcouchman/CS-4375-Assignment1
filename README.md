# CS-4375-Assignment1
<h1>Linear Regresion with Gradient Descent</h1>
<h2>Overview</h2>
<h3>Linear regression using gradient descent with custom implementation (Part 1) and Sciki-Learn's SGDRegressor (Part 2)</h3>
<h2>Setup</h2>
  <ul>
    <li>Dataset: Wine Quality from UCI ML (ID 186)
      <ul>
        <li>Publicly hosted at: https://archive.ics.uci.edu/dataset/186/wine+quality</li>
      </ul>
    </li>
    <li>Dependencies: numpy, pandas, scikit-learn, matplotlib, ucimlrepo
      <ul>
        <li>Install dependencies with: pip install -r requirements.txt</li>
        <li>Specific libraries for Part 2:
          <ul>
            <li>sklearn.preprocessing.StandardScaler: Scaling features</li>
            <li>sklearn.preprocessing.LabelEncoder: Encoding categorical variables</li>
            <li>sklearn.metrics: Tracking MSE, R^2, Variance</li>
            <li>sklearn.model_selection: Splitting data</li>
            <li>sklearn.linear_model.SGDRegressor: Stochastic Gradient Descent for Linear Regression</li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
<h2>Programs</h2>
<ul>
  <li>To run: python part1.py for Part 1, or python part2.py for Part 2</li>
  <li>Both parts: 
    <ul>
      <li>Load and preprocess dataset</li>
      <li>Fine tune various parameters on training data</li>
      <li>Evaluate model on test data</li>
      <li>Print results and generate log files and plot graphs</li>
    </ul>
  </li>
</ul>
<h2>File Contents and Structures</h2>
<ul>
  <li>logs/: Track training data and metrics for Part 1 and Part 2</li>
  <li>plots/: Track MSE vs iterations and features vs target graphs for Part 1 and Part 2</li>
  <li>part1.py: Code for Part 1</li>
  <li>part2.py: Code for Part 2</li>
  <li>README.md: Set up instructions for program</li>
  <li>report.md: Detailed report on trained model from dataset</li>
  <li>requirements.txt: To install needed dependencies</li>
</ul>
