# Decision Tree Revision Notes

## What is a Decision Tree?

A Decision Tree is a supervised Machine Learning algorithm used for:

- Classification
- Regression

It makes decisions by asking a sequence of questions about the input features.

---

## Types

1. Decision Tree Classifier
2. Decision Tree Regressor

---

## Libraries

```python
from sklearn.tree import DecisionTreeClassifier
```

---

## Steps

1. Import libraries
2. Load/Create dataset
3. Split X and y
4. Train model
5. Predict
6. Evaluate
7. Visualize

---

## Model Training

```python
model = DecisionTreeClassifier()
model.fit(X, y)
```

---

## Prediction

```python
prediction = model.predict(X)
```

---

## Accuracy

```python
accuracy_score(y, prediction)
```

Higher accuracy means better prediction.

---

## Confusion Matrix

Shows:

- True Positive
- True Negative
- False Positive
- False Negative

```python
confusion_matrix(y, prediction)
```

---

## Classification Report

Contains:

- Precision
- Recall
- F1 Score
- Support

```python
classification_report(y, prediction)
```

---

## Tree Visualization

```python
plot_tree(model)
```

Shows how the model makes decisions.

---

## Feature Importance

```python
model.feature_importances_
```

Indicates which features influenced the model the most.

Higher value = More important feature.

---

## Gini Index

Measures impurity.

Formula:

```
Gini = 1 − Σ(p²)
```

Lower Gini means a better split.

- Gini = 0 → Pure node
- Higher Gini → Mixed classes

---

## Advantages

✔ Easy to understand

✔ No feature scaling required

✔ Fast

✔ Handles nonlinear relationships

---

## Disadvantages

✘ Can overfit

✘ Sensitive to noisy data

✘ Small data changes may create different trees

---

## Interview Questions

### What is a Decision Tree?

A supervised learning algorithm used for classification and regression.

---

### What is Gini Index?

A metric used to find the best split by measuring impurity.

---

### Difference between Decision Tree Classifier and Regressor?

Classifier predicts categories.

Regressor predicts continuous values.

---

### Why Feature Importance?

To identify which features contribute most to predictions.

---

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Commands to Remember

```python
DecisionTreeClassifier()

fit()

predict()

accuracy_score()

confusion_matrix()

classification_report()

plot_tree()

feature_importances_
```

---

## One-Line Revision

Decision Tree learns decision rules from data by repeatedly splitting features to classify or predict outcomes.