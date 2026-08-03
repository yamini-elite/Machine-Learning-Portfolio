# Decision Tree Classifier using Scikit-Learn

## Project Overview
This project demonstrates how to build a Decision Tree Classifier using Python and Scikit-learn.

The model predicts whether a student will **Pass** or **Fail** based on study-related features.

---

## Dataset

Features used:

- Study_Hours
- Attendance
- Previous_Score
- Assignments

Target:

- Pass
- Fail

---

## Libraries Used

- pandas
- numpy
- matplotlib
- scikit-learn

---

## Workflow

1. Import libraries
2. Create the dataset
3. Split features and target
4. Train the Decision Tree model
5. Predict results
6. Evaluate model performance
7. Display Confusion Matrix
8. Display Classification Report
9. Visualize Decision Tree
10. Plot Feature Importance

---

## Results

Accuracy:
```
100%
```

Confusion Matrix:

```
[[2 0]
 [0 4]]
```

Classification Report:

- Precision: 1.00
- Recall: 1.00
- F1-score: 1.00

---

## Feature Importance

| Feature | Importance |
|----------|-----------|
| Study_Hours | 1.00 |
| Attendance | 0.00 |
| Previous_Score | 0.00 |
| Assignments | 0.00 |

The model found that **Study Hours alone was sufficient** to classify the training data.

---

## Decision Tree

The learned rule is:

```
If Study_Hours <= 4
    → Fail
Else
    → Pass
```

---

## Skills Learned

- Decision Tree Classification
- Gini Index
- Model Training
- Prediction
- Accuracy Evaluation
- Confusion Matrix
- Classification Report
- Decision Tree Visualization
- Feature Importance

---

## Author

Built as part of my Machine Learning learning journey.
