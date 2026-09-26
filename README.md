Credit Worthiness Analysis
===========================
Explores the German Credit Risk dataset, prepares it for modeling,
trains and compares four classification algorithms, and saves the
final model plus its encoders for deployment in a Streamlit app.

Dataset source:
- [Igor Trevelin, German Credit Risk], (https://www.kaggle.com/code/igortrevelin/german-credit-risk)
"""

# =====================================================================
# STEP 1: ANALYSIS
# =====================================================================

# Step 1.1: Import the libraries needed for data handling, plotting,
# and later modeling.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   # matplotlib.pyplot, not matplotlib itself,
                                   # since plotting functions (hist, show, etc.)
                                   # live inside the pyplot submodule
import seaborn as sns

# Step 1.2: Set display and style options so wide DataFrames print in
# full and charts share one consistent look.
pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# Step 1.3: Load the dataset.
# In the original Colab notebook this came from an interactive upload
# (google.colab.files.upload()). Locally, just point read_csv at the
# file's path.
german = pd.read_csv("german_credit_data.csv")

# Step 1.4: First look at the data, shape, and structure.
print(german.head())
print(german.tail())
print(german.info())
print(german.shape)  # 1000 rows, 11 columns before cleaning

# Step 1.5: Check for duplicate rows. None were found in this dataset.
print("Duplicate rows:", german.duplicated().sum())

# Step 1.6: Check for missing values, column by column.
print(german.isna().sum())

# Step 1.7: Quantify missingness as a percentage of the dataset, since
# raw counts alone don't say whether a column is safe to impute or
# risky to drop.
missing_pct = round((german.isna().sum() / german.shape[0]) * 100, 2)
print(missing_pct)
# Result: Saving accounts = 18.3% missing, Checking account = 39.4%
# missing. Every other column is complete.

# Step 1.8: Inspect the actual rows that contain missing values, to
# confirm the pattern before deciding how to treat them.
print(german[german.isnull().any(axis=1)])

# Step 1.9: Fill the missing values with an explicit "No_Account"
# category rather than dropping the rows or imputing a statistical
# average. This choice mirrors the original Statlog dataset, where
# "no checking account" / "no savings account" is a real category,
# not a genuine data gap, and preserves what may be a predictive
# signal (account status is a known risk indicator in the literature).
german["Saving accounts"] = german["Saving accounts"].fillna("No_Account")
german["Checking account"] = german["Checking account"].fillna("No_Account")
print(german.isna().sum())  # confirms zero missing values remain

# Step 1.10: Convert columns that represent a fixed set of categories
# from generic object type to pandas' category dtype. This is more
# memory-efficient and makes downstream filtering (e.g. select_dtypes)
# explicit about which columns are categorical.
german["Sex"] = german["Sex"].astype("category")
german["Saving accounts"] = german["Saving accounts"].astype("category")
german["Checking account"] = german["Checking account"].astype("category")
german["Risk"] = german["Risk"].astype("category")
german["Housing"] = german["Housing"].astype("category")
german["Job"] = german["Job"].astype("category")
german["Purpose"] = german["Purpose"].astype("category")
print(german.dtypes)

# Step 1.11: Review summary statistics for the numeric columns to
# understand scale, spread, and skew before visualizing them.
print(german.describe())

# Step 1.12: Drop the unnamed index column carried over from the CSV
# export; it duplicates the DataFrame's own index and adds nothing.
german.drop(columns="Unnamed: 0", inplace=True)
print(german.columns)

# Step 1.13: Visualize the distribution of the three numeric features
# (Age, Credit amount, Duration) to see their shape at a glance.
german[["Age", "Credit amount", "Duration"]].hist(bins=10, edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

# Step 1.14: Use boxplots to check each numeric feature for outliers.
plt.figure(figsize=(10, 5))
for i, col in enumerate(["Age", "Credit amount", "Duration"]):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=german[col], color="skyblue")
    plt.title(col)
plt.tight_layout()
plt.show()

# Step 1.15: Drill into one specific outlier pattern: loans with a
# duration of 60 months or more, to see who they belong to and
# whether they skew toward one risk class.
print(german.query("Duration >= 60"))

# Step 1.16: Define the categorical columns once, so later plotting
# loops don't repeat the same list.
categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

# Step 1.17: Plot the distribution of each categorical column on its
# own subplot, ordered from most to least common category.
plt.figure(figsize=(10, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=german, x=col, hue=col, palette="Set2",
                  order=german[col].value_counts().index, legend=False)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 1.18: Compute the correlation matrix among the numeric features
# to spot any strong linear relationships.
corr = german[["Age", "Job", "Credit amount", "Duration"]].corr()
print(corr)

# Step 1.19: Visualize that correlation matrix as a heatmap for a
# quicker read than the raw numbers.
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

# Step 1.20: Compare average credit amount across job skill levels and
# across sex, to see how those groups differ in typical loan size.
print(german.groupby("Job", observed=True)["Credit amount"].mean())
print(german.groupby("Sex", observed=True)["Credit amount"].mean())

# Step 1.21: Build a pivot table of average credit amount by Housing
# and Purpose, an SQL-style cross-tabulation of two categorical fields
# against one numeric measure.
print(pd.pivot_table(german, values="Credit amount", index="Housing", columns="Purpose"))

# Step 1.22: Explore how Age, Credit amount, Sex, and Duration relate
# together in a single scatterplot (color = Sex, point size = Duration).
sns.scatterplot(data=german, x="Age", y="Credit amount", hue="Sex",
                 size="Duration", alpha=0.7, palette="Set1")
plt.title("Credit amount vs Age coloured by Sex and Sized by Duration")
plt.show()

# Step 1.23: Compare the spread of credit amounts across Saving
# accounts categories using a violin plot, which shows both the
# distribution shape and its spread per group.
sns.violinplot(data=german, x="Saving accounts", y="Credit amount",
                hue="Saving accounts", palette="pastel", legend=False)
plt.title("Credit Amount Distribution by Saving Account")
plt.show()

# Step 1.24: Finally, compare every categorical feature's distribution
# split by Risk, to see which ones visibly separate good and bad risk
# applicants. This is the step that most directly informs feature
# selection for modeling.
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=german, x=col, hue="Risk", palette="Set2",
                  order=german[col].value_counts().index)
    plt.title(f"Distribution of {col} by Risk")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================================================
# STEP 2: FEATURE ENGINEERING
# =====================================================================

# Step 2.1: Choose the features to carry into modeling, based on the
# EDA above. Purpose was explored but excluded from this feature set.
features = ["Age", "Sex", "Job", "Housing", "Saving accounts",
            "Checking account", "Credit amount", "Duration"]

# Step 2.2: Name the target column being predicted.
target = "Risk"

# Step 2.3: Build a modeling-ready copy containing only the chosen
# features plus the target, leaving the original `german` DataFrame
# untouched for reference.
german_model = german[features + [target]].copy()
print(german_model.head())

# =====================================================================
# STEP 3: MODEL
# =====================================================================

# Step 3.1: Import the tools needed for encoding and for saving
# fitted objects to disk.
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 3.2: Identify which feature columns are categorical, so they
# can be label-encoded. Risk is excluded here since the target is
# encoded separately in Step 3.4.
cat_cols = german_model.select_dtypes(include="category").columns.drop("Risk")
print(cat_cols)

# Step 3.3: Label-encode each categorical feature in place, and save
# every fitted encoder so the exact same encoding can be reapplied
# later on new applicant data (e.g. inside the Streamlit app).
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    german_model[col] = le.fit_transform(german_model[col])
    le_dict[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")

# Step 3.4: Label-encode the target column separately, and save that
# encoder too, since it maps "good"/"bad" to 1/0 and is needed to
# interpret predictions later.
le_target = LabelEncoder()
german_model[target] = le_target.fit_transform(german_model[target])
print(german_model[target].value_counts())  # 700 good (1), 300 bad (0)
joblib.dump(le_target, "target_encoder.pkl")
print(german_model.head())

# Step 3.5: Split the data into predictors (X) and target (y).
from sklearn.model_selection import train_test_split

X = german_model.drop(target, axis=1)
y = german_model[target]

# Step 3.6: Create an 80/20 train-test split, stratified on the
# target so both sets preserve the original 70/30 class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=45
)
print(X_train.shape)  # (800, 8)
print(X_test.shape)   # (200, 8)

# Step 3.7: Import the four classification algorithms being compared,
# plus the tools used to tune and score them.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Step 3.8: Define a reusable training function: given a model and a
# hyperparameter grid, run 5-fold cross-validated grid search, refit
# on the best parameters, and report test accuracy.
def train_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return best_model, acc, grid.best_params_


# Step 3.9: Train and tune a Decision Tree. class_weight="balanced"
# compensates for the 70/30 class imbalance in the target.
dt = DecisionTreeClassifier(random_state=45, class_weight="balanced")
dt_param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
best_dt, acc_dt, params_dt = train_model(dt, dt_param_grid, X_train, y_train, X_test, y_test)
print("Decision Tree Accuracy", acc_dt)      # 0.655
print("Best Parameters", params_dt)

# Step 3.10: Train and tune a Random Forest with the same balanced
# class weighting.
rf = RandomForestClassifier(random_state=45, class_weight="balanced", n_jobs=-1)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
best_rf, acc_rf, params_rf = train_model(rf, rf_param_grid, X_train, y_train, X_test, y_test)
print("Random Forest Accuracy", acc_rf)      # 0.725 (best of the four)
print("Best Parameters", params_rf)

# Step 3.11: Train and tune an Extra Trees classifier as a further
# tree-ensemble comparison point.
et = ExtraTreesClassifier(random_state=45, class_weight="balanced", n_jobs=-1)
et_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
best_et, acc_et, params_et = train_model(et, et_param_grid, X_train, y_train, X_test, y_test)
print("Extra Tree Accuracy", acc_et)         # 0.690
print("Best Parameters", params_et)

# Step 3.12: Train and tune an XGBoost classifier. scale_pos_weight
# is XGBoost's own mechanism for handling class imbalance, computed
# here as the ratio of negative to positive training examples.
xgb = XGBClassifier(
    random_state=45,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
)
xgb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 1],
    "colsample_bytree": [0.7, 1],
}
best_xgb, acc_xgb, params_xgb = train_model(xgb, xgb_param_grid, X_train, y_train, X_test, y_test)
print("XGB Accuracy", acc_xgb)               # 0.675
print("Best Parameters", params_xgb)

# Step 3.13: Compare all four accuracy scores. Random Forest gives the
# best result (72.5%), so it is selected as the final model.
best_rf.predict(X_test)

# Step 3.14: Save the winning model to disk with joblib, so it can be
# loaded later without retraining, e.g. inside the Streamlit app.
joblib.dump(best_rf, "random_forest_credit_model.pkl")

# Step 3.15: Confirm every expected file (model + encoders) has been
# written to the working directory.
import os
print(os.listdir())

# Step 3.16 (Colab only): download the saved files to the local
# machine. Skip this step when running outside Colab; the files are
# already sitting in the working directory printed above.
# from google.colab import files
# files.download('random_forest_credit_model.pkl')
# files.download('Saving accounts_encoder.pkl')
# files.download('Checking account_encoder.pkl')
# files.download('Sex_encoder.pkl')
# files.download('Job_encoder.pkl')
# files.download('Housing_encoder.pkl')
# files.download('target_encoder.pkl')

# =====================================================================
# STEP 4: WEB APPLICATION
# =====================================================================
# The saved model (random_forest_credit_model.pkl) and the six saved
# encoders (Sex, Housing, Saving accounts, Checking account, Job, and
# target) are consumed by a separate Streamlit script, app.py, which
# collects applicant details through input widgets, encodes them with
# these same fitted encoders, and returns a Good/Bad Risk prediction.
# http://192.168.0.199:8501/ to run the app.py
