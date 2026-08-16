# Linear Regression

This experiment explores linear regression from its mathematical formulation to implementations with NumPy,
scikit-learn, and PyTorch.

The main goal is to understand what happens behind a high-level `fit()` call and connect classical machine learning
optimization with the training workflow used in neural networks.

## Goal

Understand linear regression at several levels of abstraction:

* Derive the linear regression objective and normal equation solution.
* Implement the normal equation solution with NumPy.
* Implement linear regression using gradient descent with NumPy.
* Compare the implementations with `sklearn.linear_model.LinearRegression`.
* Implement the same model with PyTorch using automatic differentiation and an optimizer.
* Compare model parameters and evaluation metrics across implementations.

## Implementations

The experiment contains four approaches to the same regression problem:

* **Normal equation** - Analytical solution derived from the mean squared error objective and implemented with NumPy.
* **NumPy gradient descent** - Iterative optimization with manually calculated gradients and parameter updates.
* **Scikit-learn** - Linear regression using `sklearn.linear_model.LinearRegression`.
* **PyTorch** - Linear regression using `nn.Linear`, automatic differentiation, and an optimizer.

Reusable implementations are stored in [`implementations/`](./implementations), while the full experiment is documented
and executed in [`linear_regression.ipynb`](logistic_regression.ipynb).

## Experiment checklist

* [x] Define the regression problem and linear model.
* [x] Derive the mean squared error objective.
* [x] Derive the normal equation solution.
* [x] Implement the normal equation solution with NumPy.
* [x] Derive gradients for model parameters.
* [x] Standardize input features for stable gradient-based optimization.
* [x] Implement gradient descent with NumPy.
* [x] Visualize true vs predicted values on the test set.
* [x] Train the same model with scikit-learn.
* [x] Train the same model with PyTorch and autograd.
* [x] Compare learned parameters and evaluation metrics across implementations.
* [x] Write conclusions about the differences between the approaches.

## Full Experiment

For the complete step-by-step walkthrough, see [`linear_regression.ipynb`](logistic_regression.ipynb).

The notebook includes exploratory data analysis, the mathematical derivation of linear regression and the normal
equation, feature standardization, four model implementations, coefficient interpretation, evaluation metrics,
prediction visualization, and final conclusions.
