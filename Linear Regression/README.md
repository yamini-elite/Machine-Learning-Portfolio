# 📈 Linear Regression — Machine Learning

A practical implementation and study of **Linear Regression and its related regression techniques** using Python and Scikit-learn.

This project covers multiple regression approaches through practical examples involving **house prices, income, salary, car prices, and experience-based salary prediction**.

The notebook also demonstrates regression evaluation metrics and regularization techniques such as **Ridge and LASSO Regression**.

---

## 🎯 What is Linear Regression?

Linear Regression is a supervised Machine Learning algorithm used to predict a **continuous numerical value**.

The basic equation is:

```text
y = mx + b
```

Where:

```text
y → Predicted value
m → Coefficient / Slope
x → Input feature
b → Intercept
```

---

## 🧠 Concepts Covered

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
- LASSO Regression
- Coefficients
- Intercept
- Train-Test Split
- Regression Line
- Prediction
- MAE
- MSE
- RMSE
- R² Score
- L1 Regularization
- L2 Regularization
- Feature Selection with LASSO

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab

---

## 📂 Project Structure

```text
Linear_Regression/
│
├── Linear Regression.ipynb
├── note.md
└── README.md
```

---

# 🏠 Example 1 — House Price Prediction

The first example uses a house pricing dataset.

### Features

```text
area
```

### Target

```text
price
```

The model learns the relationship between:

```text
House Area → House Price
```

---

## 📊 Data Visualization

A scatter plot is used to visualize the relationship between:

```text
X-axis → Area (square feet)

Y-axis → Price (US$)
```

A regression line is then plotted over the data points.

---

## ⚙️ Model

The model is created using:

```python
from sklearn import linear_model

reg = linear_model.LinearRegression()
```

Training:

```python
reg.fit(
    df[["area"]],
    df[["price"]]
)
```

---

## 📐 Learned Equation

The model learns:

```text
y = mx + b
```

The notebook obtains:

```python
reg.coef_
```

for the coefficient and:

```python
reg.intercept_
```

for the intercept.

For the house price example:

```text
Coefficient ≈ 130.87686567

Intercept ≈ 188871.26865672
```

The equation can therefore be represented as:

```text
Price =
130.87686567 × Area
+
188871.26865672
```

---

# 🌎 Example 2 — Canada Per Capita Income

The notebook also uses:

```text
canada_per_capita_income.csv
```

The model uses:

```text
Input → year

Target → per capita income (US$)
```

Model:

```python
reg = linear_model.LinearRegression()

reg.fit(
    df[["year"]],
    df[["per capita income (US$)"]]
)
```

This demonstrates using Linear Regression to model the relationship between year and per capita income.

---

# 💼 Example 3 — Salary Prediction

A small dataset is created with:

```text
id
salary
```

Example:

```text
id → salary
```

The model is trained using:

```python
X = df[["id"]]
Y = df[["salary"]]

model = LinearRegression()

model.fit(X, Y)
```

A salary prediction is then generated for:

```text
id = 8
```

---

# 📈 Example 4 — Polynomial Regression

Polynomial Regression is demonstrated using a **Car Age vs Price** dataset.

### Dataset

```text
Age
Price
```

Example values:

```text
Age  → 1, 2, 3, 4, 5, 6, 7

Price → 950000, 880000, 810000,
        720000, 620000, 510000, 430000
```

The relationship is modeled using polynomial features.

---

## Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

For degree 2, the transformation introduces polynomial terms such as:

```text
1
x
x²
```

The transformed features are then used with Linear Regression.

---

## Model

```python
model = LinearRegression()

model.fit(X_poly, y)
```

A new car's price is then predicted after applying the same transformation:

```python
new_car_poly = poly.transform(new_car)

prediction = model.predict(new_car_poly)
```

---

# 💰 Example 5 — Experience vs Salary

Another Polynomial Regression example uses:

```text
Experience → Salary
```

The dataset contains experience values from 1 to 7 years and corresponding salaries.

Polynomial features are created using:

```python
PolynomialFeatures(degree=2)
```

The model is trained using Linear Regression and then used to predict the salary for:

```text
Experience = 8 years
```

---

# 🛡️ Example 6 — Ridge Regression

Ridge Regression is a regularized version of Linear Regression.

The notebook uses:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=100)
```

Dataset:

```text
Experience → Salary
```

The model predicts salary for:

```text
Experience = 9 years
```

---

## Ridge Regularization

Ridge uses:

```text
L2 Regularization
```

Its purpose is to control the size of model coefficients and reduce model complexity.

The `alpha` parameter controls the strength of regularization.

```text
Higher alpha
     ↓
Stronger regularization
```

---

# 🔎 Example 7 — LASSO Regression

LASSO stands for:

> Least Absolute Shrinkage and Selection Operator

The notebook uses LASSO for house price prediction.

Dataset:

```text
house_price.csv
```

### Features

```text
Area
Bedrooms
House_Age
Distance_City
```

### Target

```text
Price
```

---

## Train-Test Split

The dataset is split using:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

Therefore:

```text
80% → Training
20% → Testing
```

---

## LASSO Model

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=100)

model.fit(X_train, Y_train)
```

Predictions:

```python
y_pred = model.predict(X_test)
```

---

# 📊 Regression Evaluation

The LASSO model is evaluated using:

```text
MAE
MSE
RMSE
R²
```

### MAE

Mean Absolute Error.

```text
MAE =
Average(|Actual - Predicted|)
```

Lower is generally better.

### MSE

Mean Squared Error.

```text
MSE =
Average((Actual - Predicted)²)
```

Lower is generally better.

### RMSE

Root Mean Squared Error.

```text
RMSE = √MSE
```

Lower is generally better.

### R²

R-squared measures how well the model explains variation in the target.

Generally:

```text
Higher R² → Better fit
```

---

# 🔍 LASSO Coefficients

The notebook examines:

```python
model.coef_
```

LASSO uses:

```text
L1 Regularization
```

L1 regularization can shrink some coefficients exactly to zero.

Therefore, LASSO can also perform a form of **feature selection**.

---

# 🏠 New House Prediction

A new house is created:

```python
new_house = pd.DataFrame({
    "Area": [1800],
    "Bedrooms": [4],
    "House_Age": [5],
    "Distance_City": [3]
})
```

The trained LASSO model predicts its price:

```python
price = model.predict(new_house)
```

---

# 📚 Regression Techniques Covered

| Technique | Main Use |
|---|---|
| Linear Regression | Continuous prediction |
| Multiple Linear Regression | Multiple input features |
| Polynomial Regression | Curved relationships |
| Ridge Regression | L2 regularization |
| LASSO Regression | L1 regularization + feature selection |

---

# 🔄 Overall Workflow

```text
Dataset
   ↓
Understand Features
   ↓
Separate X and Y
   ↓
Train Model
   ↓
Predict
   ↓
Evaluate
   ↓
Interpret Coefficients
   ↓
Try Appropriate Regression Technique
```

---

# 🧠 Key Learning

The core Linear Regression concept is:

```text
Input Features
      ↓
Linear Equation
      ↓
Prediction
```

For simple Linear Regression:

```text
y = mx + b
```

For Multiple Linear Regression:

```text
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Polynomial Regression adds polynomial terms.

Ridge adds L2 regularization.

LASSO adds L1 regularization.

---

# 🚀 Future Improvements

Possible improvements to this project:

- Feature Scaling
- Cross-Validation
- Hyperparameter Tuning
- Better train-test strategies
- Residual Analysis
- R² comparison between models
- Regularization parameter tuning
- Model comparison

---

# 📌 Key Takeaway

This project demonstrates the progression from basic Linear Regression to more advanced regression techniques:

```text
Linear Regression
       ↓
Multiple Linear Regression
       ↓
Polynomial Regression
       ↓
Ridge Regression
       ↓
LASSO Regression
```

> **Linear Regression predicts continuous values by learning relationships between features and a target. Polynomial Regression models curved relationships, while Ridge and LASSO introduce regularization to control model complexity.**