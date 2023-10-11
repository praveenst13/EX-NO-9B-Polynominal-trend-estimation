# EX-NO-9B-Polynominal-trend-estimation
## AIM:
  To implement Polynominal and linear trend estimation  using python program 
## PROCEDURE:
   1 Import necessary libraries, including NumPy for numerical operations, Matplotlib for plotting, and Pandas for data handling.
   
   2 Fit a linear regression model to the data using LinearRegression() from the sklearn.linear_model module.
   
   3 Use PolynomialFeatures from the sklearn.preprocessing module to create polynomial features of degree 4.
   
   4 Add a title, xlabel, and ylabel for the plot and Display the plot using plt.show()
   
   5 Use lin.predict() to predict with the linear regression model
   
## PROGRAM:
```

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
datas = pd.read_csv('data.csv')
datas

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# Features and the target variables
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
 
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
 
lin.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
 
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
 
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
# Visualising the Linear Regression results
plt.scatter(X, y, color='blue')
 
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
 
plt.show()


plt.scatter(X, y, color='blue')
 
plt.plot(X, lin2.predict(poly.fit_transform(X)),
         color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
 
plt.show()


# Predicting a new result with Linear Regression
# after converting predict variable to 2D array
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)


# Predicting a new result with Polynomial Regression
# after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))




```
## OUTPUT:




### Linear Regression:

![image](https://github.com/praveenst13/EX-NO-9A-linear-trend-estimation/assets/118787793/52095cee-2d03-457e-84a9-e517a4ae1234)
### Polynominal Regression:
![image](https://github.com/praveenst13/EX-NO-9A-linear-trend-estimation/assets/118787793/a817f843-cad3-41e5-8c2e-a0ade87c1b37)


## RESULT :
  Thus the program to implement Polynominal and linear trend estimation is written and verified using python programming.


