# XGBoost - Bank Marketing Revision Notes

## 1. What is XGBoost?

XGBoost stands for Extreme Gradient Boosting.

It is a supervised machine learning algorithm based on Boosting.

It can be used for:

- Classification
- Regression

XGBoost builds decision trees sequentially.

Each new tree tries to correct the errors made by previous trees.

---

## 2. Ensemble Learning

XGBoost belongs to:

> Boosting

### Bagging

Example: Random Forest

Trees are built independently.

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤ → Final Prediction
Tree 4 ─┘


### XGBoost Classification

Our project is:

Problem → Binary Classification

Target:
y

No  → 0
Yes → 1


Dataset

We are using the Bank Marketing Dataset.

Objective

Predict whether a customer will subscribe to a term deposit.

Target Distribution
No  = 39,922
Yes = 5,289

Approximately:

No  → 88%
Yes → 12%

Therefore, the dataset is imbalanced.

Features
Numerical Features
age
balance
day
duration
campaign
pdays
previous
Categorical Features
job
marital
education
default
housing
loan
contact
month
poutcome


Target Encoding

The target is binary.

No  → 0
Yes → 1

Label Encoding can be used for the target.


One-Hot Encoding

Categorical input features need to be converted into numerical values.

X = pd.get_dummies(X, drop_first=True)

This converts categorical columns into numerical dummy variables.

drop_first=True

Removes one category from each categorical feature to avoid redundant information.

Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

Meaning:

80% → Training
20% → Testing

random_state=42 makes the split reproducible.



Creating XGBoost Model
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

Creating the model does not train it.

Training happens using:

model.fit(X_train, y_train)

Prediction
y_pred = model.predict(X_test)

Predictions are:

0 → No
1 → Yes

Model Evaluation

Important classification metrics:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

For an imbalanced classification problem, do not rely only on Accuracy.


Precision
Simple Meaning

Naan YES sonna, adhu evlo correct?

In English:

Of all customers predicted as YES, how many were actually YES?

Bank Example

Model says 100 customers are likely to subscribe.

Actually 60 of them subscribe.

Precision = 60%



Recall
Simple Meaning

Actually YES ah irundha people-la, evlo YES-ah kandupidichom?

In English:

Of all customers who were actually YES, how many did the model successfully identify?

Bank Example

There are actually 100 customers who will subscribe.

Model identifies 80 of them.

Recall = 80%


Precision vs Recall
Precision
"Naan YES sonna, adhu correct-ah?"
Recall
"Actually YES people-la, evlo perai kandupidichom?"
Business Decision

If missing a potential customer is expensive:

Recall → Important

If contacting an uninterested customer is expensive:

Precision → Important


Baseline XGBoost Result

Initial model:

Accuracy  ≈ 90.36%
Precision (Yes) ≈ 63%
Recall (Yes) ≈ 49%
F1-score (Yes) ≈ 55%

Although Accuracy was high, Recall was only around 49%.

That means the model was missing many actual YES customers.


Confusion Matrix

Baseline:

[[7638  314]
 [ 557  534]]

Meaning:

TN = 7638
FP = 314
FN = 557
TP = 534
Important

False Negative:

Actual YES, but predicted NO.

For our business problem, reducing False Negatives is important.

Imbalanced Dataset

Our target distribution:

No  → 88%
Yes → 12%

This can cause the model to favor the majority class.

Therefore:

High Accuracy
      ≠
Good Model

We need to look at Recall, Precision and F1-score.



Ways to Handle Imbalanced Data

Common approaches:

Class Weighting
SMOTE
Undersampling
Threshold Tuning
Collecting more minority-class data

For XGBoost, we tested:

scale_pos_weight



scale_pos_weight

Formula:

scale_pos_weight = negative_samples / positive_samples

For our dataset:

39922 / 5289 ≈ 7.55

Therefore:

model_weighted = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=7.55
)
What does it do?

It gives greater importance to the minority/positive class.

In our case:

Yes → minority class

The model is encouraged to pay more attention to identifying YES customers.



Result After Class Weighting
Accuracy  ≈ 88%

Precision (Yes) ≈ 49%

Recall (Yes) ≈ 81%

F1-score (Yes) ≈ 61%

Most important change:

Recall:

49% → 81%

So the model found many more potential YES customers.

Trade-off:

Precision:

63% → 49%

Precision-Recall Trade-off

When we make the model more focused on the positive class:

Recall ↑

but:

Precision may ↓

This is called the:

Precision-Recall Trade-off

There is no universally best balance.

The correct balance depends on the business requirement.



Threshold Tuning

Normally, the model uses approximately:

Probability >= 0.50 → YES
Probability < 0.50  → NO

We can change the threshold.

For example:

Threshold = 0.40

Then:

Probability = 0.45

0.45 >= 0.40

→ YES

With threshold 0.50:

0.45 < 0.50

→ NO

Getting Prediction Probabilities

Instead of:

y_pred = model.predict(X_test)

we can get probabilities:

y_prob = model_weighted.predict_proba(X_test)[:, 1]

Then apply our own threshold:

threshold = 0.40

y_pred_threshold = (
    y_prob >= threshold
).astype(int)



Threshold 0.40 Result

Our result:

Accuracy ≈ 85.90%

Precision (Yes) ≈ 45%

Recall (Yes) ≈ 84%

F1-score (Yes) ≈ 59%

Recall improved:

81% → 84%

So lowering the threshold made the model more willing to predict YES.

Threshold Rules

Generally:

Threshold ↓
     ↓
More YES predictions
     ↓
Recall tends to ↑
     ↓
Precision may ↓

And:

Threshold ↑
     ↓
Fewer YES predictions
     ↓
Precision may ↑
     ↓
Recall may ↓
Important

These are general trends, not absolute guarantees.



Business Decision

Our bank business requirement:

Don't miss potential customers.

Therefore:

Recall is important.

So the 0.40 threshold model is preferable to the baseline for this particular business objective.

We accept some additional False Positives in exchange for finding more actual YES customers.


Feature Engineering

Feature Engineering means:

Creating meaningful new features from existing data.

Example:

df["total_contacts"] = (
    df["campaign"] + df["previous"]
)

If:

campaign = 3
previous = 2

then:

total_contacts = 5

This represents the customer's overall contact history.



Contact Intensity

Another possible feature:

df["contact_intensity"] = (
    df["duration"] / df["campaign"]
)

Example:

duration = 600 seconds
campaign = 3

Therefore:

600 / 3 = 200

So:

contact_intensity = 200 seconds/contact
Important

Division by zero must be handled safely.



Feature Engineering Rule

Do not create features randomly.

Follow:

Create Feature
      ↓
Train Model
      ↓
Evaluate
      ↓
Compare
      ↓
Keep if Useful

A feature should have a meaningful business interpretation.



Data Leakage
Definition

Data Leakage occurs when the model receives information that would NOT be available at the time of prediction.

This causes unrealistic model performance.

31. Bank Example of Data Leakage

Suppose we want to predict whether a customer will subscribe:

BEFORE the marketing call

But we use:

duration

duration represents the length of the marketing call.

Before the call:

duration → Unknown

Therefore using it would be problematic because the model is receiving future information.

Important Rule

A feature should only be used if its information is available at prediction time.

32. Cross-Validation

Cross-validation gives us multiple evaluations instead of relying on one train/test split.

For 5-Fold Cross-Validation:

Fold 1 → Validation
Fold 2 + 3 + 4 + 5 → Training

Fold 2 → Validation
Fold 1 + 3 + 4 + 5 → Training

Fold 3 → Validation
Remaining folds → Training

Fold 4 → Validation
Remaining folds → Training

Fold 5 → Validation
Remaining folds → Training

Each fold becomes the validation set once.

The final CV score is generally the average of the fold scores.

33. Example of Cross-Validation

Suppose:

Fold 1 = 0.88
Fold 2 = 0.91
Fold 3 = 0.89
Fold 4 = 0.90
Fold 5 = 0.92

Average:

(0.88 + 0.91 + 0.89 + 0.90 + 0.92) / 5
= 0.90

Overall CV score:

90%
34. Hyperparameter Tuning

Hyperparameters are values we choose before training the model.

Important XGBoost hyperparameters:

n_estimators

Controls the number of trees.

n_estimators = 200

means approximately:

200 boosting trees

Increasing it can make the model more powerful, but can also:

Increase training time
Increase the risk of overfitting
35. max_depth

Controls the depth/complexity of each tree.

Lower value:

Simpler trees
↓
Less complexity

Higher value:

More complex trees
↓
Higher risk of overfitting
36. learning_rate

Controls how much each new tree contributes to the final model.

Lower learning rate:

Slower learning

Usually requires:

More trees

Higher learning rate:

Faster learning

but can increase the risk of overfitting.

37. Important Hyperparameter Relationship

Remember:

learning_rate ↓
       ↓
Usually need more trees
       ↓
n_estimators ↑
38. GridSearchCV

GridSearchCV tests different combinations of hyperparameters.

Example:

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

Number of combinations:

2 × 2 × 2 = 8

With:

cv = 5

Total model training operations:

8 × 5 = 40
39. GridSearchCV Configuration
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)
Important Parameters

scoring="recall"

Because Recall is our business priority.

cv=5

Uses 5-Fold Cross-Validation.

n_jobs=-1

Uses available CPU cores to speed up the search.

40. Best Parameters We Found

Our GridSearchCV result:

learning_rate = 0.1
max_depth = 5
n_estimators = 200

Best CV Recall:

≈ 48.47%

This taught us:

Hyperparameter tuning does NOT automatically guarantee a better model.

The main issue in our dataset was class imbalance.

41. Why Accuracy Alone Is Not Enough

Our baseline:

Accuracy ≈ 90%
Recall ≈ 49%

This looks good if we only look at Accuracy.

But the model was missing many actual YES customers.

After class weighting:

Accuracy ≈ 88%
Recall ≈ 81%

Accuracy decreased.

But from the business perspective, the model became more useful because it found more potential customers.


Machine Learning Approach

The project follows a complete Machine Learning workflow:

Dataset
   ↓
Data Understanding
   ↓
EDA
   ↓
Data Preprocessing
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Baseline XGBoost
   ↓
Model Evaluation
   ↓
Class Imbalance Handling
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Feature Engineering
   ↓
Final Evaluation