# 🌲 Random Forest Notes

## What is Random Forest?

Random Forest is an Ensemble Machine Learning algorithm used for Classification and Regression.

Instead of building one Decision Tree, it builds multiple Decision Trees and combines their predictions.

---

## Why Random Forest?

Decision Trees can overfit the training data.

Random Forest reduces overfitting by combining predictions from many trees.

---

## How Random Forest Works

Step 1

Randomly selects samples from the dataset (Bootstrap Sampling).

↓

Step 2

Each Decision Tree is trained using different random samples.

↓

Step 3

Each tree uses a random subset of features while splitting.

↓

Step 4

Every tree predicts a class.

↓

Step 5

Majority Voting determines the final prediction.

---

## Ensemble Learning

Ensemble Learning means combining multiple models to produce a stronger and more accurate prediction.

Random Forest is an example of Ensemble Learning.

---

## Bootstrap Sampling

Each Decision Tree is trained on a randomly selected dataset created from the original dataset.

Different trees receive different samples.

---

## Random Feature Selection

Every tree considers only a random subset of features while finding the best split.

This makes trees different from each other and improves generalization.

---

## Majority Voting

Example

Tree 1 → Positive

Tree 2 → Positive

Tree 3 → Negative

Tree 4 → Positive

Tree 5 → Positive

Final Prediction → Positive

---

## Feature Importance

Feature Importance measures how much each feature contributes to the prediction.

Higher Importance

→ More influence on prediction.

Lower Importance

→ Less influence but still may contribute.

The total importance of all features equals 1.

---

## Evaluation Metrics

Accuracy

Measures overall prediction performance.

Confusion Matrix

Shows:

- True Positive
- True Negative
- False Positive
- False Negative

Classification Report

Includes:

- Precision
- Recall
- F1-score

---

## Advantages

✔ High Accuracy

✔ Reduces Overfitting

✔ Handles Large Datasets

✔ Works well with Non-linear Data

✔ Provides Feature Importance

✔ Robust and Reliable

---

## Disadvantages

✘ Slower than Decision Trees

✘ Uses More Memory

✘ Less Interpretable than a Single Decision Tree

---

## Decision Tree vs Random Forest

| Decision Tree | Random Forest |
|---------------|---------------|
| Single Tree | Multiple Trees |
| Can Overfit | Less Overfitting |
| Faster | Slightly Slower |
| Less Accurate | More Accurate |
| Easy to Interpret | Harder to Interpret |
| No Ensemble Learning | Uses Ensemble Learning |

---

## Interview Questions

Q. Why Random Forest performs better than Decision Tree?

Answer:

Random Forest combines predictions from multiple Decision Trees using Majority Voting. Since each tree is trained on different samples and different subsets of features, the model reduces overfitting and generally provides better accuracy and generalization than a single Decision Tree.

---

Q. What is Feature Importance?

Answer:

Feature Importance indicates how much each feature contributes to the model's predictions. Features with higher importance have a greater influence on the final decision.

---

Q. Why is the sum of Feature Importance equal to 1?

Answer:

Feature importance values are normalized. Each value represents the proportion of the model's total importance contributed by that feature, so all importance values together sum to 1.

---

## Project Result

Dataset:
Heart Disease Prediction

Algorithm:
Random Forest Classifier

Accuracy:
98.48%

Top Features:

1. Troponin
2. KCM
3. Age

Result:

Random Forest achieved excellent performance and slightly outperformed the Decision Tree model by reducing prediction errors.