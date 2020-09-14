# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:44:35 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv("D:\\chetan\\assignment\\11decision tree\\Company_Data.csv")
data.head()
data.describe()
data.info()
from sklearn.preprocessing import LabelEncoder

categorical_feature_mask = data.dtypes==object
print(categorical_feature_mask)
categorical_cols = data.columns[categorical_feature_mask].tolist()
print(categorical_cols)
le=LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
data[categorical_cols].head(10)
data.info()
data.describe()
data.isnull().sum()

bins = [0, 8, np.inf]
names = [0, 1]# 0=low sale,1=high sale
data['Sales_n'] = pd.cut(data['Sales'], bins, labels=names)
data.info()
data.head(5)

colnames = list(data.columns)
predictors = colnames[1:11]
target = colnames[11]
print(colnames)
data.fillna(method='ffill',inplace=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
type(preds)

pd.crosstab(test[target],preds)
np.mean(preds==test.Sales_n)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.model_selection import cross_val_score
parameters = {'max_depth':[1,2,3,4,5], 'min_sample_leaf':[1,2,3,4,5],
              'min_sample_split':[2,3,4,5],'criterion':['gini','entropy']}
scorer = make_scorer(f1_score)
grid_obj = GridSearchCV(model,parameters,scoring=scorer) 
grid_fit = grid_obj.fit(train[predictors],train[target])
best_clf = grid_obj.best_estimater_
best_clf
scores = cross_val_score(best_clf,train[predictors],train[target],cv=5,scoring='f1_marco')
scores.mean