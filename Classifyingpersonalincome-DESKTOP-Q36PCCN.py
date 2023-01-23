# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:34:51 2022

@author: my
"""
#==============================================================================
#CLASSIFYING PERSONAL INCOME
#=============================================================================
f = open("Classify personal income.text","w")
print("#===================== Classifying prsonal income =====================",file = f)

import pandas as pd 
import numpy as np
import seaborn as sns    # To visualize data

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix


#============================================================================
#Importing data
#============================================================================

data_income = pd.read_csv('income.csv')

# Creating a copy of original data
data = data_income.copy()

""" Exploratory data analysis:

#1. Getting to know the data
#2. Data preprocessing (missing values)
#3. Cross tables and data visualisation
"""
print("\n", "Exploratory data analysis:", file = f)
#=============================================================================
# Getting to know the data
#=============================================================================
#****** To check variables' data type
print(data.info())

#************** check for missing values

print("\n\n\n", "Check the missing values :", file = f )
data.isnull().sum()
print("\n", data.isnull().sum(), file = f)
print("\n", " There are  no missing Values.",file = f)
#****** No missing values!

#****** Summary of numerical variables

print("\n\n\n", "Summary of numerical variables:", file = f)
summary_num = data.describe()
print("\n", data.describe(),file = f)
print(summary_num)

#****** Summary of categorical variables

print("\n\n\n","Summary of categorical variables :",file = f)
pd.set_option('display.max_columns', 500)
summary_cate = data.describe(include = 'O')
print("\n",data.describe(include = 'O'), file = f)
print(summary_cate)

#****** Frequency of each categories

print("\n\n\n","#==========Frequency of each categories:", file = f)
data['JobType'].value_counts()
print("\n","Frequency of JobType:","\n", data['JobType'].value_counts(), file = f)
data['occupation'].value_counts()
print("\n","Frequency of occupation:","\n", data['occupation'].value_counts(), file = f)

#**** Checking for unique classes

print("\n", "#================Checking the unique class================",file = f)
print(np.unique(data['JobType']))
print("\n","unique class of JobType:",np.unique(data['JobType']), file = f)
print(np.unique(data['occupation']))
print("\n","unique class of occupation:", np.unique(data['occupation']), file = f)

""" Go back and read the data by including "na_values[' ?']" 
"""
print("\n\n", "Go back and read the data by including na_values[' ?']." , file = f)
data = pd.read_csv('income.csv',na_values=[" ?"])

#=============================================================================
# Data pre-processing
#=============================================================================

print("\n\n\n","#===================== Data pre-processing=====================", file = f)
data.isnull().sum()
print("\n",data.isnull().sum(), file = f)

missing = data[data.isnull().any(axis=1)]
# axis=1, to consider at least one column value is missing 

""" Points to note:
1. Missing values in jobtype  = 1809
2. Missing values in occupation = 1816
3. There are 1809 rows where two specific columns i.e.occupation & JobType
 have missing values
4. (1816-1809) = 7, we  still have occupation unfilled for these 7 rows.
    Because, jobtype is never worked """

print("\n\n", "1. Missing values in jobtype  = 1809.",
            "\n"," 2. Missing values in occupation = 1816.",
            "\n", "3. There are 1809 rows where two specific columns i.e.occupation & JobType",
            "\n",  "have missing values. ",
            "\n"," 4. (1816-1809) = 7, we  still have occupation unfilled for these 7 rows.",
            "\n",   "Because, jobtype is never worked.", file = f)

data2 = data.dropna(axis=0)      

# Relationship between independent variables
print("\n\n\n", "Relationship between independent variables:", file = f)
correlation = data2.corr()
print("\n","correlation:","\n", data2.corr(), file = f)
print(correlation)

#=============================================================================
# Cross tables & Data visualization
#=============================================================================
# Extracting the column names
print("\n\n\n", "#======== Cross tables & Data visualization============ ",file = f)
data.columns

#=============================================================================
# Gender proportion table:
#=============================================================================

print("\n\n\n", "Gender proportion table:", file = f)
gender = pd.crosstab(index    = data2["gender"],
                     columns  = 'count',
                     normalize = True)

print("\n\n", pd.crosstab(index  = data2["gender"], columns  = 'count', normalize = True), file = f)
print(gender)

#=============================================================================
# Gender vs Salary status:
#=============================================================================
print("\n\n\n", " Gender vs Salary status: ", file = f)

gender_salstat = pd.crosstab(index      = data2["gender"],
                             columns    = data2['SalStat'],
                             margins    = True,
                             normalize  = 'index') # "normalize=index" to get the row proportion =1

print("\n\n",  pd.crosstab(index      = data2["gender"],
             columns    = data2['SalStat'],
             margins    = True,
             normalize  = 'index'), file = f) 
print(gender_salstat)

#=============================================================================
# Frequency distribution of 'Salary status'
#=============================================================================
print("\n\n\n", "Frequency distribution of Salary status:", file = f)
SalStat = sns.countplot(data2['SalStat'])

""" 75% of people's salary status is <=50,000
    & 25% of people's salary status is >50,000
"""
print("\n\n", "75% of people's salary status is <=50,000",
             "\n","& 25% of people's salary status is >50,000", file = f)

######################### Histogram of Age ###################################
sns.distplot(data['age'], bins=10, kde=False)
# people with age 20-45 age are high in frequency
print("\n", "people with age 20-45 age are high in frequency.", file = f)
########################  Box plot - Age vs Salary status#####################
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

###########################jobType vs Salary Status###########################
sns.countplot(x ="EdType", Data= "data2", hue = "SalStat" )


#=============================================================================
# Logistic regression
#=============================================================================
print("\n\n\n", "#==================Logistic Regression :===============", file = f)
# Reindexing the salary status names to 0,1
print("\n", "Reindexing the salary status names to 0,1.", file = f)
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first=True)

# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
print("\n"," Storing the output values in y.", file = f)
y=new_data['SalStat'].values  # '.values' is used to extract the values from Salary status and store it into y
print(y) 

# Storing the values from input features
print("\n", "Storing the values from input features in x.", file = f)
x = new_data[features].values
print(x)

# Splitting the data into train and test
print("\n", "Splitting the data into train and test.",file = f)
train_x,test_x,train_y,test_y =   train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the model
print("\n", "Make an instance of the model.", file = f)
logistic = LogisticRegression()

# Fitting the values for x and y
print("\n", "Fitting the values for x and y.", file = f)
logistic.fit(train_x,train_y) 
logistic.coef_
logistic.intercept_

# Prediction from test data
print("\n", "Prediction from test data.", file = f)
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
print("\n", "Confusion matrix:", file = f)
confusion_matrix = confusion_matrix(test_y,prediction)
print("\n", confusion_matrix(test_y,prediction), file = f)
print(confusion_matrix)

# Calculating the accuracy
print("\n\n", "Calculating the accuracy of this model:", file = f)
accuracy_score=accuracy_score(test_y,prediction)
print("\n",accuracy_score(test_y,prediction), file = f)
print(accuracy_score)

# Printing the misclassified values from prediction

print('Misclassified samples: %d' %(test_y != prediction).sum())
print("\n", "Misclassified samples: %d" %(test_y != prediction).sum(), file = f)


#=============================================================================
# Logistic Regression - Removing insignificant variables
#=============================================================================

print("\n\n\n", "#============Logistic Regression with removing insignificant variable======", file = f)
print("\n", "Again Reindexing the salary status.", file =f)

# Reindexing the salary status names to 0,1

data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])

print("\n", "Removing the gender, race, Jobtype variables",
       "\n","and repeat the process to build a model  as we did previously. ", file = f)
cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols, axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train & test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)

# Prediction from test data
prediction = logistic.predict(test_x)

# Confusion matrix
print("\n\n\n", "Confusion matrix:",file = f) 
confusion_matrix = confusion_matrix(test_y,prediction)
print("\n", confusion_matrix(test_y,prediction), file = f)
print(confusion_matrix)

# Calculating the accuracy
print("\n\n\n", "Calculating the accuracy of this model: ", file = f)
accuracy_score = accuracy_score(test_y,prediction)
print("\n", accuracy_score(test_y,prediction),file = f)
print(accuracy_score)

#==============================================================================
# KNN
#==============================================================================
print("\n\n\n", "#============Using KNN algorithm to build model===============", file = f)

# importing the library of KNN
print("\n", "Importing the library of KNN.", file = f)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Storing the K nearest neighbors classifier
print("\n", "Storing the K nearest neighbors classifier.", file = f)
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
 
# Fitting the values for X and Y
print("\n", "Fitting the values for X and Y. ", file = f)
KNN_classifier.fit(train_x,train_y)

print("\n", "Predict the test_x: ", file = f)
prediction = KNN_classifier.predict(test_x)

print(prediction)

# Confusion matrix
print("\n\n", "Confusion matrix:", file = f)
confusion_matrix = confusion_matrix(test_y,prediction)
print("\n",confusion_matrix(test_y,prediction), file = f)
print(confusion_matrix)

# Calculating the accuracy
print("\n","Calculate the accuracy score:", file = f)
accuracy_score=accuracy_score(test_y,prediction)
print("\n",accuracy_score(test_y,prediction), file = f)
print(accuracy_score)

print('Misclassified samples: %d' %(test_y != prediction).sum())

print("\n\n","accuracy_score of logistic classifier = 0.8365565255829374",file = f)
print("\n","Misclassified samples by logistic classifier = 1479",file = f)
print("\n\n","accuracy score of logistic classifier without insignficant variables = 0.8355619405459167",file = f)
print("\n\n", "Misclassified samples of KNN classifier = 1490",file = f)
print("\n", "accuracy score of KNN classifier = 0.835340921648801",file = f)
print("\n\n", "Logistic classifier is comparatively  good performing than KNN. ",
        "\n", "But logistic classifier without insignificant variables doesn't produce better result ", file = f )

f.close()
