# Logistic Regression

This experiment explores logistic regression from its mathematical formulation to implementations with NumPy,
scikit-learn, and PyTorch.

The main goal is to extend the linear regression workflow to binary classification and understand how sigmoid, Binary
Cross-Entropy, and classification metrics change the model while preserving the underlying linear transformation.

The experiment uses the Oxford-IIIT Pet dataset for binary Cat vs Dog image classification.

## Goal

Understand logistic regression at several levels of abstraction:

* Formulate binary classification as an extension of a linear model.
* Derive Binary Cross-Entropy and its gradient for logistic regression.
* Implement logistic regression using full-batch gradient descent with NumPy.
* Compare the implementation with `sklearn.linear_model.LogisticRegression`.
* Implement the same linear classifier with PyTorch using automatic differentiation and an optimizer.
* Compare classification performance across implementations.

## Implementations

The experiment contains three approaches to the binary classification problem:

* **NumPy gradient descent** - Logistic regression with manually implemented sigmoid, analytically derived gradients,
  and parameter updates.
* **Scikit-learn** - Logistic regression using `sklearn.linear_model.LogisticRegression` with its default L2
  regularization.
* **PyTorch** - Logistic regression using `nn.Linear`, `BCEWithLogitsLoss`, automatic differentiation, and an SGD
  optimizer.

Reusable implementations are stored in [`implementations/`](./implementations), while the full experiment is
documented and executed in [`logistic_regression.ipynb`](./logistic_regression.ipynb).

## Experiment checklist

* [x] Define the binary classification problem and logistic regression model.
* [x] Load and inspect the Oxford-IIIT Pet dataset.
* [x] Inspect the Cat and Dog class distribution.
* [x] Preprocess images while preserving their aspect ratio.
* [x] Convert images into flattened feature vectors.
* [x] Define Binary Cross-Entropy.
* [x] Derive the Binary Cross-Entropy gradient using the chain rule.
* [x] Implement logistic regression with NumPy and gradient descent.
* [x] Train the same type of classifier with scikit-learn.
* [x] Train the model with PyTorch and autograd.
* [x] Compare Binary Cross-Entropy, Accuracy, Precision, Recall, and F1-score.
* [x] Compare confusion matrices across implementations.
* [x] Write conclusions about the differences between the approaches and the limitations of a linear image classifier.

## Full Experiment

For the complete step-by-step walkthrough, see [`logistic_regression.ipynb`](./logistic_regression.ipynb).

The notebook includes the mathematical transition from linear regression to binary logistic regression, Oxford-IIIT
Pet dataset exploration, image preprocessing, Binary Cross-Entropy and gradient derivation, three model
implementations, classification metrics, confusion matrices, implementation comparison, and final conclusions.
