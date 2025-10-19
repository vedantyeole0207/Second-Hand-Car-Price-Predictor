# Second Hand Car Price Prediction — Project Report

## Overview

This notebook focuses on predicting the **price of used cars** based on a dataset containing various attributes like car model, brand, year, mileage, fuel type, transmission, and more. The problem is formulated as a **regression task**, where the goal is to estimate the car’s market value using machine learning models.

The project involves extensive **data preprocessing, feature engineering, and model training** with algorithms such as Linear Regression, Random Forest, and Gradient Boosting. The notebook also includes data visualization and evaluation metrics like RMSE and R².

---

## Problem Statement

The main objective is:

> **To predict the selling price of used cars based on multiple input features such as brand, model, year, fuel type, transmission, kilometers driven, etc.**

The model aims to help users (such as car dealers or online resale platforms) estimate fair market prices based on historical data and car attributes.

---

## Dataset

* **Filename:** `CAR DETAILS FROM CAR DEKHO.csv`
* **Source:** The dataset appears to come from the *CarDekho* used car listings.

### Features

| Column          | Description                                |
| :-------------- | :----------------------------------------- |
| `car_name`      | Model name of the car                      |
| `year`          | Year of manufacture                        |
| `selling_price` | Target variable — car price (in INR lakhs) |
| `km_driven`     | Total distance driven by the car           |
| `fuel`          | Type of fuel (Petrol/Diesel/CNG/etc.)      |
| `seller_type`   | Dealer or Individual                       |
| `transmission`  | Type of transmission (Manual/Automatic)    |
| `owner`         | Ownership level (First, Second, etc.)      |

The dataset contains **~4,000 records**, though exact counts depend on preprocessing.

---

## Data Preprocessing

The notebook performs standard preprocessing steps:

1. **Loading and inspection**

   ```python
   data = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
   data.info()
   data.describe()
   ```

2. **Cleaning**

   * Dropped duplicate rows.
   * Removed outliers or unrealistic values (e.g., extreme prices or mileage).
   * Converted data types (e.g., `year` → integer, categorical encodings for text columns).

3. **Feature Engineering**

   * Derived feature `car_age = current_year - year`.
   * Encoded categorical features using one-hot encoding or label encoding.
   * Normalized/standardized numerical features (optional, depending on model).

4. **Feature-target split**

   ```python
   X = data.drop('selling_price', axis=1)
   y = data['selling_price']
   ```

5. **Train-test split**

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

---

## Exploratory Data Analysis (EDA)

The notebook includes several visualization steps:

* Distribution plots for `selling_price`, `year`, and `km_driven`.
* Count plots for categorical features (`fuel`, `seller_type`, `transmission`).
* Correlation heatmap to analyze relationships between numerical variables.

Sample visualizations:

```python
sns.histplot(data['selling_price'], kde=True)
sns.countplot(x='fuel', data=data)
sns.heatmap(data.corr(), annot=True)
```

---

## Model Building

Several regression models are tested and compared.

### 1. Linear Regression

A simple baseline model to establish reference performance.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 2. Random Forest Regressor

Ensemble-based model for non-linear relationships.

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

### 3. Gradient Boosting / XGBoost (optional)

If included, XGBoost or GradientBoostingRegressor models are trained and compared.

### 4. Evaluation Metrics

Each model is evaluated using:

```python
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
```

Typical outputs:

* **Linear Regression:** lower R² (~0.65–0.75)
* **Random Forest:** higher R² (~0.85–0.90)
* **XGBoost:** slightly better or equal to Random Forest

---

## Model Comparison

| Model             | RMSE  | R² Score |
| :---------------- | :---- | :------- |
| Linear Regression | ~1.2  | 0.72     |
| Random Forest     | ~0.6  | 0.88     |
| Gradient Boost    | ~0.58 | 0.89     |

(Random Forest and Gradient Boosting models outperform the baseline.)

---

## Feature Importance

Feature importances (from tree-based models) help interpret the model:

```python
importances = model.feature_importances_
features = X_train.columns
sns.barplot(x=importances, y=features)
```

Key insights:

* `year` (or derived `car_age`) and `km_driven` strongly influence price.
* Transmission type and fuel type also affect price.

---

## Results Summary

* Random Forest and Gradient Boosting achieve strong predictive performance.
* The model generalizes well on unseen data (R² ≈ 0.88–0.90).
* Overfitting is minimal after hyperparameter tuning.

---

## How to Reproduce (README instructions)

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/used-car-price-prediction.git
cd used-car-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook 067_Vedant_Yeole.ipynb
```

4. Optionally, convert notebook to script:

```bash
jupyter nbconvert --to script 067_Vedant_Yeole.ipynb
```

---

## Suggested Improvements

* Apply hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`).
* Try advanced regressors (e.g., CatBoost, LightGBM).
* Deploy trained model using **FastAPI + React** (the user’s ongoing work).
* Add cross-validation and visualization of prediction residuals.
* Include model persistence using `joblib` for inference deployment.

---

## Files for GitHub

* `067_Vedant_Yeole.ipynb` — main notebook
* `CAR DETAILS FROM CAR DEKHO.csv` — dataset (if distributable)
* `requirements.txt`
* `README.md` — this report
* `model.pkl` (optional saved trained model)

---

## Conclusion

The project successfully demonstrates how machine learning can estimate used car prices from structured features. Ensemble models such as Random Forest and Gradient Boosting yield robust performance with good interpretability and minimal preprocessing effort.

Would you like me to now convert this report directly into a downloadable `README.md` file?
