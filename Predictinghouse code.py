
file = open("Pridictinghouse.text","w")
import pandas as pd 
import numpy as np
import seaborn as sns    # To visualize data


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#============================================================================
#Importing data
#============================================================================

data = pd.read_excel('DS - Assignment Part 1 data set.xlsx')
data2 = data.copy()


#==============================================================================
# Structure of the dataset
#==============================================================================
data2.info()

#==============================================================================
# Summarizing data
#==============================================================================
data2.describe()

# Data cleaning
#==============================================================================
data2.isnull().sum()
data2.duplicated(keep = 'first').sum()
data2.drop_duplicates(keep='first',inplace=True)

# Transaction data
data2['Transaction date'].value_counts().sort_index()
sns.distplot(data2['Transaction date'])
sns.boxplot(y=data2['Transaction date'])

# House age
data2['House Age'].value_counts().sort_index()
sns.distplot(data2['House Age'])
sns.boxplot(y=data2['House Age'])

# Distance from nearest Metro station 
data2['Distance from nearest Metro station (km)'].value_counts().sort_index()
sns.distplot(data2['Distance from nearest Metro station (km)'])
sns.boxplot(y=data2['Distance from nearest Metro station (km)'])

# Number of convenience stores
data2['Number of convenience stores'].value_counts().sort_index()
sns.distplot(data2['Number of convenience stores'])
sns.boxplot(y=data2['Number of convenience stores'])

# latitude
data2['latitude'].value_counts().sort_index()
sns.distplot(data2['latitude'])
sns.boxplot(y=data2['latitude'])

# longitude
sns.distplot(data2['longitude'])
sns.boxplot(y=data2['longitude'])

# Number of bed
sns.distplot(data2['Number of bedrooms'])
sns.boxplot(y=data2['Number of bedrooms '])

# House size
sns.distplot(data2['House size (sqft)'])
sns.boxplot(y=data2['House size (sqft)'])

# Visualizing parameters 
sns.regplot(x='Transaction date', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2)

sns.regplot(x='House Age', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2)

sns.regplot(x='House Age', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 

sns.regplot(x='Distance from nearest Metro station (km)', y='House price of unit area', 
            scatter=True,fit_reg=False, data=data2) 

sns.regplot(x='Number of convenience stores', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 

sns.regplot(x='latitude', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 

sns.regplot(x='longitude', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 

sns.regplot(x='Number of bedrooms', y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 

sns.regplot(x='House size (sqft)',  y='House price of unit area', scatter=True,
            fit_reg=False, data=data2) 


# Check the correlation between variables
correlation = data2.corr()

# dropping the insignificant variables
col=['Transaction date','House size (sqft)','Number of bedrooms']
data3 = data2.drop(columns=col, axis=1)

# Converting categorical variables to dummy variables
data_predict = pd.get_dummies(data3,drop_first=True) 

# Separating input and output features
x1 = data3.drop(['House price of unit area'], axis='columns', inplace=False)
y1 = data3['House price of unit area']

# Take a logarithmic
y1 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#================================================================================
# BASELINE MODEL 
#=================================================================================

"""
We are making a base model by using test data mean value 
This is to set a benchmark and to compare with our regression model
"""

# finding the mean for test data value
base_pred = np.mean(y_test) 
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test))

# finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)

#==============================================================================
# LINEAR REGRESSION 
#==============================================================================

# setting intercept as true
lgr=LinearRegression(fit_intercept=True)

# Model
model_lin1=lgr.fit(X_train,y_train)

# Predicting model on test set
house_predictions_lin1 = lgr.predict(X_test)

# Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, house_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

# R squared value
r2_lin_test1 = model_lin1.score(X_test,y_test)
r2_lin_train1 = model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

# Regression diagnostics - Residual plot analysis 
residuals1=y_test-house_predictions_lin1
sns.regplot(x=house_predictions_lin1, y=residuals1, scatter=True,
            fit_reg=False, data=data3)
residuals1.describe()

#=============================================================================
# RANDOM FOREST 
#=============================================================================

# Model parameters
rf = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)


# Model
model_rf1=rf.fit(X_train,y_train)

# Pridicting model on test set
house_predictions_rf1 = rf.predict(X_test)

# Computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test,house_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)
file.close()
