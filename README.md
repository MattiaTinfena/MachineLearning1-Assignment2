# MachineLearning1-Assignment2
## Linear Regression from Scratch

Python experiments implementing linear regression with NumPy and comparing model formulations on stock-index and vehicle datasets.

### Scope

- Univariate regression without an intercept.
- Univariate regression with an intercept.
- Multivariate regression.
- Comparison of fitted coefficients, predictions and training/test errors over repeated random splits.
- Plotting of regression results and error summaries.

### Data

The repository includes `turkish-se-SP500vsMSCI.csv` for stock-index experiments and `mtcarsdata-4features.csv` for vehicle-data regression. Input-column choices and experiment settings are defined in `lab2Main.py`.

### Setup and execution

```bash
git clone https://github.com/MattiaTinfena/MachineLearning1-Assignment2.git
cd MachineLearning1-Assignment2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib
python lab2Main.py
```

Run from the repository root to preserve the relative CSV paths. Results are printed and displayed in Matplotlib figures.

### Main files

- `lab2Main.py`: experiment orchestration and visualisation.
- `linearRegressionFunc.py`: regression functions.
- `loadData.py`: data preparation and splitting.

### Reproducibility

The experiments use random splits. Fix the random seeds before comparing runs if repeatable numerical results are required. The stock-index experiment is an academic regression exercise and does not evaluate an investment strategy.
