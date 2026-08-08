# K-Nearest Neighbors (KNN) Notes

## What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

It predicts the output based on the majority class of the K nearest data points.

---

## Why is it called Lazy Learning?

KNN does not build a mathematical model during training.

During `fit()`, it simply stores the training dataset.

The actual computation happens during prediction.

---

## How KNN Works

Step 1:
Choose a value for K.

Step 2:
Calculate the distance between the new data point and every training sample.

Step 3:
Select the K nearest neighbors.

Step 4:
Count the class labels.

Step 5:
The class with the majority vote becomes the prediction.

---

## Distance Calculation

The default distance metric is Euclidean Distance.

Formula:

Distance = √((x₂-x₁)² + (y₂-y₁)²)

The smaller the distance, the more similar the data points.

---

## Choosing K

Small K

Advantages

- Captures local patterns
- More flexible

Disadvantages

- Can overfit
- Sensitive to noise

Large K

Advantages

- More stable
- Less affected by noise

Disadvantages

- Can underfit
- May ignore local patterns

A common practice is to test multiple K values and choose the one with the highest validation accuracy.

---

## Why Feature Scaling is Important

KNN is a distance-based algorithm.

If one feature has much larger values than another, it dominates the distance calculation.

Example

Age: 20–80

Salary: 20,000–1,000,000

Without scaling, Salary contributes much more to the distance than Age.

Using StandardScaler or MinMaxScaler ensures that all features contribute fairly.

---

## Advantages

- Simple to understand
- Easy to implement
- No training phase
- Works well on small datasets
- Supports multi-class classification

---

## Disadvantages

- Slow prediction on large datasets
- Sensitive to feature scaling
- Sensitive to irrelevant features
- Requires choosing an appropriate K value

---

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

## Accuracy vs K Graph

Purpose

- Helps determine the optimal K value.
- Shows how model performance changes with different K values.
- Identifies possible overfitting or underfitting.

---

## Real-World Applications

- Medical Diagnosis
- Recommendation Systems
- Image Classification
- Handwriting Recognition
- Pattern Recognition

---

## Interview Questions

### Why is KNN called a Lazy Learning Algorithm?

Because it stores the training data during training and performs most computations during prediction.

---

### Why do we use odd values of K?

Odd K values help reduce ties during majority voting in binary classification.

---

### Why is feature scaling necessary in KNN?

Because KNN uses distance calculations, and features with larger numerical ranges can dominate the distance if the data is not scaled.

---

### Which distance metric is commonly used in KNN?

Euclidean Distance.

---

### When should KNN be avoided?

- Very large datasets
- High-dimensional data
- When prediction speed is critical

---

## Summary

KNN predicts the class of a new sample by finding the K nearest neighbors and selecting the majority class. Since it relies on distance calculations, choosing an appropriate K value and applying feature scaling are essential for achieving good performance.