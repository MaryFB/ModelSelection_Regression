#Selecting the best regression model 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor_mul = LinearRegression()
regressor_mul.fit(X_train, y_train)

# Polynomial Regression model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, y_train)

# Decision Tree Regression model 
from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state = 0)
regressor_tree.fit(X_train, y_train)

# Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_forest.fit(X_train, y_train)

# Predicting the Test set results for all models
y_pred_mul = regressor_mul.predict(X_test)
y_pred_tree = regressor_tree.predict(X_test)
y_pred_forest = regressor_forest.predict(X_test)
y_pred_poly = regressor_poly.predict(poly_reg.transform(X_test))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("R-squared for Multiple Linear Regression: {:.3f} ".format(r2_score(y_test, y_pred_mul)))
print("R-squared for Polynomial Regression: {:.3f} ".format(r2_score(y_test, y_pred_poly)))
print("R-squared for Decission Tree: {:.3f} ".format(r2_score(y_test, y_pred_tree)))
print("R-squared for Random Forest: {:.3f} ".format(r2_score(y_test, y_pred_forest)))
print()
model_list = [('Linear Regression' , r2_score(y_test, y_pred_mul)),('Decision Tree' , r2_score(y_test, y_pred_tree)),('Random Forest' , r2_score(y_test, y_pred_forest)) ]
print ("The best regression model is {} with the  R-squared of {:.3f}".format(max(model_list)[0] , max(model_list)[1]))