# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:09:59 2021

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os 
from sklearn.linear_model import LogisticRegression


print(os.getcwd())

os.chdir("C:\\Users\\HP\\Desktop\\Demograpic")

print(os.getcwd())

data = pd.read_csv("hotel_bookings.csv")
data.head()

df_base = data 

arr = df_base.values


X = df_base.iloc[:,[2,7,8,9]] # ivs for train 
X

y = df_base.iloc[:,28] 
y


regr = LogisticRegression()

regr.fit(X, y)

pred = regr.predict(X)
 
pred

regr.score(X,y)

model_data = data.sample(frac=0.8) 

print(model_data.head())





model_data = data.sample(frac=0.8,random_state=1234)# Random state is used for reproducibility
print(model_data.head())

data.shape

model_data = data.sample(frac=0.8,random_state=1234)
print(model_data.shape)



test_data = data.loc[~df_base.index.isin(model_data.index), :]
print(test_data.shape)


df = model_data









