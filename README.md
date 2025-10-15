# House Price Prediction Using Machine Learning

## Project Overview
This project predicts house prices using a range of machine learning models to determine which algorithm performs best.  
The goal is to analyze housing data, identify the key factors influencing prices, and build reliable predictive models for accurate estimation.

---

## Problem Statement
House prices depend on several factors such as location, size, build quality, and neighborhood characteristics.  
The challenge is to use data-driven modeling to predict the sale price and understand which features have the most significant impact.

---

## Dataset
The dataset includes variables such as:
- LotArea, OverallQual, YearBuilt, GrLivArea, GarageCars, and more  
- Target variable: SalePrice

### Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Normalized numerical features  
- Split data into training and test sets  

---

## Models Implemented
The following models were trained and compared:

1. Linear Regression – baseline model  
2. Ridge Regression – regularized linear model  
3. Decision Tree Regressor – captures non-linear relationships  
4. Random Forest Regressor – ensemble learning model  
5. Gradient Boosting Regressor (GBR) – boosting-based algorithm  
6. Artificial Neural Network (ANN) – deep learning approach  

---

## Tools and Libraries
- Python 3.10+  
- Pandas, NumPy – for data manipulation  
- Matplotlib, Seaborn – for visualization  
- Scikit-learn – for training and evaluating ML models  
- TensorFlow / Keras – for neural network implementation  

---

## Model Evaluation
All models were evaluated using the following metrics:
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)

These metrics helped compare performance and determine the best model.

---

## Results and Conclusions
- Gradient Boosting achieved the highest accuracy and lowest error rate.  
- Random Forest performed well, with consistent generalization.  
- Neural Networks performed competitively but required more fine-tuning.  
- Linear Regression provided strong interpretability but struggled with complex non-linear data.

### Final Takeaway
Ensemble learning models like Gradient Boosting and Random Forest are the most effective for predicting house prices in this dataset.

---

## Key Insights
- The most influential features were OverallQual, GrLivArea, and GarageCars.  
- Removing outliers and scaling features improved results.  
- Comparing traditional ML and neural networks showed trade-offs between accuracy and interpretability.  

---

## Example Workflow
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("housing.csv")

# Split data
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Evaluate
preds = gbr.predict(X_test)
print("R² Score:", r2_score(y_test, preds))
