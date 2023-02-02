import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('./datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder() # Encoding text to numbers
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

X = X.astype(float)
y = y.astype(float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
SL = 0.05

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary Least Squares
print(regressor_OLS.summary())
print()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
print()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

# Automatic Backward Elimination with p-values only:
# import statsmodels.api as sm
# def backwardElimination(x, sl):    
#     numVars = len(x[0])    
#     for i in range(0, numVars):        
#         regressor_OLS = sm.OLS(y, x.tolist()).fit()        
#         maxVar = max(regressor_OLS.pvalues).astype(float)        
#         if maxVar > sl:            
#             for j in range(0, numVars - i):                
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
#                     x = np.delete(x, j, 1)    
#     regressor_OLS.summary()    
#     return x 
 
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)

# Automatic Backward Elimination with p-values and Adjusted R Squared:
# import statsmodels.api as sm
# def backwardElimination(x, SL):    
#     numVars = len(x[0])    
#     temp = np.zeros((50,6)).astype(int)    
#     for i in range(0, numVars):        
#         regressor_OLS = sm.OLS(y, x.tolist()).fit()        
#         maxVar = max(regressor_OLS.pvalues).astype(float)        
#         adjR_before = regressor_OLS.rsquared_adj.astype(float)        
#         if maxVar > SL:            
#             for j in range(0, numVars - i):                
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
#                     temp[:,j] = x[:, j]                    
#                     x = np.delete(x, j, 1)                    
#                     tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
#                     adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
#                     if (adjR_before >= adjR_after):                        
#                         x_rollback = np.hstack((x, temp[:,[0,j]]))                        
#                         x_rollback = np.delete(x_rollback, j, 1)     
#                         print (regressor_OLS.summary())                        
#                         return x_rollback                    
#                     else:                        
#                         continue    
#     regressor_OLS.summary()    
#     return x 
 
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)