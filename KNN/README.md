# K-Nearest Neighbors (KNN) Classification

## 📌 Project Overview

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using the Iris dataset. KNN is a supervised machine learning algorithm used for classification by predicting the class based on the majority vote of the nearest neighbors.

---

## 📂 Dataset

**Dataset:** Iris Dataset

The dataset contains 150 flower samples with four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Variable:

- Setosa
- Versicolor
- Virginica

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🔄 Workflow

1. Import Libraries
2. Load the Dataset
3. Split Features (X) and Target (Y)
4. Train-Test Split
5. Train the KNN Classifier
6. Predict Test Data
7. Evaluate the Model
8. Plot Accuracy vs K

---

## 📊 Model Evaluation

Metrics Used:

- Accuracy Score
- Confusion Matrix
- Classification Report

### Results

Accuracy: **100%**

Confusion Matrix

```
[[10 0 0]
 [0 9 0]
 [0 0 11]]
```

The model correctly classified all test samples.

---

## 📈 Visualization

Accuracy vs K Value

This graph illustrates how the accuracy changes as the number of neighbors (K) increases.

It helps identify the optimal K value while understanding overfitting and underfitting behavior.

---

## 📚 Key Concepts Learned

- Supervised Learning
- Classification
- Lazy Learning Algorithm
- Distance-Based Classification
- Majority Voting
- Euclidean Distance
- Choosing the K Value
- Model Evaluation
- Importance of Feature Scaling

---

## 🚀 Conclusion

The KNN classifier achieved excellent performance on the Iris dataset with 100% accuracy. Since the Iris dataset is clean and well-separated, KNN performs consistently across multiple K values.