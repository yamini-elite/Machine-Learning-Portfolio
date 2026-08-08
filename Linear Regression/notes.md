# 📈 Linear Regression — Revision Notes

## 1. What is Linear Regression?

Linear Regression is a supervised Machine Learning algorithm used to predict a **continuous numerical value**.

Examples:

- House Price Prediction
- Salary Prediction
- Per Capita Income Prediction
- Car Price Prediction

The basic idea is to find a relationship between input features and a continuous target.

---

# 2. Basic Linear Regression Equation

The simple linear regression equation is:

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

In Machine Learning:

```text
y = coefficient × feature + intercept
```

---

# 3. Simple Linear Regression

Simple Linear Regression uses:

```text
1 Input Feature
       ↓
1 Target Variable
```

Example:

```text
Area → House Price
```

The model learns the relationship between house area and price.

---

# 4. House Price Prediction

Dataset:

```python
df = pd.read_csv("/content/housepricing.csv")
```

The dataset contains:

```text
area
price
```

The relationship is visualized using:

```python
plt.scatter(
    df.area,
    df.price,
    color="red",
    marker="+"
)
```

The axes represent:

```text
X-axis → Area (square feet)

Y-axis → Price (US$)
```

---

# 5. Creating the Linear Regression Model

```python
from sklearn import linear_model

reg = linear_model.LinearRegression()
```

At this stage, the model is created but has not learned anything yet.

---

# 6. Training the Model

```python
reg.fit(
    df[["area"]],
    df[["price"]]
)
```

Here:

```text
X → area

Y → price
```

The model learns the coefficient and intercept.

---

# 7. Making a Prediction

To predict the price for a house with an area of `3300`:

```python
reg.predict(
    pd.DataFrame([3300])
)
```

The model uses the learned equation:

```text
y = mx + b
```

---

# 8. Coefficient

The coefficient represents the slope `m`.

It can be obtained using:

```python
reg.coef_
```

For the house price model:

```text
Coefficient ≈ 130.87686567
```

Therefore:

```text
m = 130.87686567
```

---

# 9. Intercept

The intercept represents `b`.

It can be obtained using:

```python
reg.intercept_
```

For the house price model:

```text
Intercept ≈ 188871.26865672
```

Therefore:

```text
b = 188871.26865672
```

---

# 10. Manual Prediction

The equation becomes:

```text
y = mx + b
```

For an area of `3300`:

```text
y =
130.87686567 × 3300
+
188871.26865672
```

Python:

```python
130.87686567 * 3300 + 188871.26865672
```

This demonstrates how Linear Regression makes a prediction mathematically.

---

# 11. Regression Line

The relationship between area and price can be visualized with a regression line.

```python
plt.scatter(
    df.area,
    df.price,
    color="red",
    marker="+"
)

plt.plot(
    df.area,
    reg.predict(df[["area"]]),
    color="blue"
)
```

The scatter plot shows the actual data points.

The line represents the values predicted by the Linear Regression model.

---

# 12. Predicting Multiple Values

Another dataset is loaded:

```python
d = pd.read_csv("area.csv")
```

Predictions are generated using:

```python
p = reg.predict(d)
```

The predictions are added to the DataFrame:

```python
d["price"] = p
```

Then the updated data is saved:

```python
d.to_csv("area.csv")
```

This demonstrates how a trained model can be used to generate predictions for multiple input values.

---

# 13. Canada Per Capita Income Example

Another dataset is loaded:

```python
df = pd.read_csv(
    "/content/canada_per_capita_income.csv"
)
```

The model is trained using:

```python
reg = linear_model.LinearRegression()

reg.fit(
    df[["year"]],
    df[["per capita income (US$)"]]
)
```

Here:

```text
Input Feature → year

Target → per capita income (US$)
```

The model learns the relationship between year and per capita income.

---

# 14. Multiple Linear Regression

Multiple Linear Regression is used when there are **multiple input features**.

Instead of:

```text
y = mx + b
```

we have:

```text
y =
m₁x₁
+
m₂x₂
+
m₃x₃
+
...
+
b
```

Where:

```text
x₁, x₂, x₃ → Multiple input features

m₁, m₂, m₃ → Corresponding coefficients

b → Intercept
```

---

# 15. Simple Salary Prediction Example

A small DataFrame is created:

```python
data = {
    "id": [1, 2, 3, 4, 5],
    "salary": [25000, 30000, 38000, 45000, 52000]
}
```

Features:

```python
X = df[["id"]]
```

Target:

```python
Y = df[["salary"]]
```

The model:

```python
model = LinearRegression()
```

Training:

```python
model.fit(X, Y)
```

Prediction:

```python
prediction = model.predict([[8]])
```

This predicts the salary corresponding to `id = 8` based on the learned linear relationship.

---

# 16. Polynomial Regression

Polynomial Regression is used when the relationship between the input and target is not adequately represented by a straight line.

Example:

```text
Age → Car Price
```

The relationship can be curved rather than linear.

---

# 17. Polynomial Regression Equation

For degree 2:

```text
y = b₀ + b₁x + b₂x²
```

For higher degrees:

```text
y =
b₀
+
b₁x
+
b₂x²
+
b₃x³
+
...
```

---

# 18. PolynomialFeatures

Polynomial features can be created using:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
```

Then:

```python
X_poly = poly.fit_transform(X)
```

The original feature is transformed into polynomial features.

For example:

```text
x
```

becomes approximately:

```text
1, x, x²
```

for degree 2.

---

# 19. Polynomial Regression — Car Price

The notebook uses:

```python
data = {
    "Age": [1,2,3,4,5,6,7],
    "Price": [
        950000,
        880000,
        810000,
        720000,
        620000,
        510000,
        430000
    ]
}
```

Here:

```text
X → Age

Y → Price
```

Polynomial features:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

Then Linear Regression is applied:

```python
model = LinearRegression()

model.fit(X_poly, Y)
```

---

# 20. Predicting Car Price

A new car is created:

```python
new_car = pd.DataFrame({
    "Age": [10]
})
```

The new data must also be transformed:

```python
new_car_poly = poly.transform(new_car)
```

Then:

```python
prediction = model.predict(new_car_poly)
```

Important:

> The same polynomial transformation used during training must also be applied to new data before prediction.

---

# 21. Polynomial Regression — Experience vs Salary

Another example uses:

```python
data = {
    "exp": [1,2,3,4,5,6,7],
    "salary": [
        25000,
        32000,
        42000,
        55000,
        70000,
        85000,
        98000
    ]
}
```

Here:

```text
X → Experience

Y → Salary
```

Polynomial features are created:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

The model is trained:

```python
model = LinearRegression()

model.fit(X_poly, Y)
```

Then salary for `8` years of experience is predicted.

---

# 22. Ridge Regression

Ridge Regression is a regularized version of Linear Regression.

It adds a penalty to the model to control the size of coefficients.

The notebook uses:

```python
from sklearn.linear_model import Ridge
```

Model:

```python
model = Ridge(alpha=100)
```

---

# 23. Ridge Regression — Why?

Regularization helps control model complexity.

Ridge uses **L2 regularization**.

Conceptually:

```text
Linear Regression
       +
Coefficient Penalty
       ↓
Ridge Regression
```

The `alpha` parameter controls the strength of regularization.

```text
Higher alpha
    ↓
Stronger regularization
```

---

# 24. Ridge Regression Example

Dataset:

```python
data = {
    "Experience": [1,2,3,4,5,6,7,8],
    "Salary": [
        25000,
        32000,
        42000,
        55000,
        70000,
        85000,
        98000,
        110000
    ]
}
```

Features:

```python
X = df[["Experience"]]
```

Target:

```python
Y = df["Salary"]
```

Model:

```python
model = Ridge(alpha=100)
```

Training:

```python
model.fit(X, Y)
```

Prediction for 9 years of experience:

```python
new_data = pd.DataFrame({
    "Experience": [9]
})

prediction = model.predict(new_data)
```

---

# 25. LASSO Regression

LASSO stands for:

> Least Absolute Shrinkage and Selection Operator

It is another regularized version of Linear Regression.

The notebook uses:

```python
from sklearn.linear_model import Lasso
```

---

# 26. LASSO vs Ridge

Both are regularization techniques.

```text
Ridge
→ L2 Regularization

LASSO
→ L1 Regularization
```

Important difference:

```text
Ridge
→ Shrinks coefficients toward zero

LASSO
→ Can shrink some coefficients exactly to zero
```

Therefore, LASSO can perform a form of feature selection.

---

# 27. LASSO House Price Prediction

Dataset:

```python
df = pd.read_csv(
    "/content/house_price.csv"
)
```

Missing values are checked:

```python
df.isnull().sum()
```

---

# 28. LASSO Features

The model uses:

```text
Area
Bedrooms
House_Age
Distance_City
```

Input:

```python
X = df[
    [
        "Area",
        "Bedrooms",
        "House_Age",
        "Distance_City"
    ]
]
```

Target:

```python
Y = df["Price"]
```

---

# 29. LASSO Train-Test Split

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

# 30. Training the LASSO Model

```python
model = Lasso(alpha=100)

model.fit(
    X_train,
    Y_train
)
```

Predictions:

```python
y_pred = model.predict(X_test)
```

---

# 31. Regression Evaluation Metrics

The notebook evaluates the LASSO model using:

```text
MAE
MSE
RMSE
R²
```

---

# 32. MAE

MAE stands for:

> Mean Absolute Error

Formula:

```text
MAE =
Average(
|Actual - Predicted|
)
```

It represents the average absolute difference between actual and predicted values.

Lower MAE is generally better.

---

# 33. MSE

MSE stands for:

> Mean Squared Error

Formula:

```text
MSE =
Average(
(Actual - Predicted)²
)
```

Because errors are squared, larger errors receive more penalty.

Lower MSE is generally better.

---

# 34. RMSE

RMSE stands for:

> Root Mean Squared Error

Formula:

```text
RMSE = √MSE
```

RMSE is in the same unit as the target variable.

Lower RMSE is generally better.

---

# 35. R² Score

R² stands for:

> R-squared / Coefficient of Determination

It measures how well the model explains the variation in the target.

The notebook calculates it using:

```python
r2_score(Y_test, y_pred)
```

Generally:

```text
Higher R² → Better fit
```

But R² should be interpreted together with other metrics and the context of the problem.

---

# 36. LASSO Model Coefficients

The notebook checks:

```python
print(model.coef_)
```

This displays the learned coefficient for each feature.

Because LASSO uses L1 regularization, some coefficients can become zero.

A coefficient of zero means the corresponding feature has been removed from the model's linear contribution.

---

# 37. Predicting a New House

The notebook creates:

```python
new_house = pd.DataFrame({
    "Area": [1800],
    "Bedrooms": [4],
    "House_Age": [5],
    "Distance_City": [3]
})
```

Prediction:

```python
price = model.predict(new_house)
```

Then:

```python
print(price)
```

The model predicts the estimated price for the new house.

---

# 38. Linear Regression Family

The notebook covers several regression techniques:

```text
Linear Regression
      ↓
Polynomial Regression
      ↓
Ridge Regression
      ↓
LASSO Regression
```

---

# 39. Simple Linear Regression

Use when there is one main input feature.

Example:

```text
Area → House Price
```

Equation:

```text
y = mx + b
```

---

# 40. Multiple Linear Regression

Use when there are multiple input features.

Example:

```text
Area
Bedrooms
Age
Distance
     ↓
House Price
```

Equation:

```text
y =
b₀
+
b₁x₁
+
b₂x₂
+
b₃x₃
+
...
```

---

# 41. Polynomial Regression

Use when the relationship is curved/non-linear and polynomial terms can model that relationship.

Example:

```text
Car Age → Car Price
```

Degree 2:

```text
y = b₀ + b₁x + b₂x²
```

---

# 42. Ridge Regression

Use Linear Regression with L2 regularization.

```text
Ridge
→ L2
→ Shrinks coefficients
→ Helps control model complexity
```

---

# 43. LASSO Regression

Use Linear Regression with L1 regularization.

```text
LASSO
→ L1
→ Shrinks coefficients
→ Can make coefficients exactly zero
→ Can perform feature selection
```

---

# 44. Important Python Functions

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

### Train

```python
model.fit(X_train, Y_train)
```

### Predict

```python
model.predict(X_test)
```

### Coefficients

```python
model.coef_
```

### Intercept

```python
model.intercept_
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

### Ridge

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=100)
```

### LASSO

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=100)
```

---

# 45. Regression Evaluation Metrics

```text
MAE
MSE
RMSE
R²
```

Remember:

```text
MAE
→ Average absolute error

MSE
→ Average squared error

RMSE
→ Square root of MSE

R²
→ How well the model explains target variation
```

For error metrics:

```text
Lower MAE → Better
Lower MSE → Better
Lower RMSE → Better
```

For R²:

```text
Higher R² → Generally better
```

---

# 46. Important Difference: Classification vs Regression

### Classification

Predicts a class.

Examples:

```text
Loan Approved / Not Approved
Pass / Fail
Spam / Not Spam
```

Common metrics:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

### Regression

Predicts a continuous numerical value.

Examples:

```text
House Price
Salary
Income
```

Common metrics:

```text
MAE
MSE
RMSE
R²
```

---

# 47. One-Minute Revision

```text
Linear Regression
       ↓
Continuous Prediction
       ↓
y = mx + b
```

### Simple Linear Regression

```text
1 Feature
→ 1 Target
```

### Multiple Linear Regression

```text
Multiple Features
→ 1 Target
```

### Polynomial Regression

```text
Curved Relationship
→ Polynomial Features
→ Linear Regression
```

### Ridge

```text
L2 Regularization
→ Coefficients Shrink
```

### LASSO

```text
L1 Regularization
→ Coefficients Can Become Zero
→ Feature Selection
```

---

# 48. Final Takeaway

The main workflow learned from this notebook is:

```text
Dataset
   ↓
Understand Features and Target
   ↓
Create X and Y
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

The key concepts to remember are:

```text
Linear Regression
Simple Linear Regression
Multiple Linear Regression
Polynomial Regression
Ridge Regression
LASSO Regression
Coefficient
Intercept
MAE
MSE
RMSE
R²
Regularization
L1
L2
```

> **Linear Regression predicts continuous numerical values by learning the relationship between input features and a target. Polynomial Regression can model curved relationships, while Ridge and LASSO add regularization to control model complexity.**