# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:15:56 2020

@author: Nikitha
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Nikitha\.spyder-py3\combine.csv')
df.head

#handling missing values
df.isnull()

#plot a heatmap and check for missing values
plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#handle nan values
df['T'].replace(to_replace = np.nan, value = 23.7, inplace=True) 
df['T']

df['TM'].replace(to_replace = np.nan, value = 30, inplace=True) 
df['TM']

df['tm'].replace(to_replace = np.nan, value = 19.4, inplace=True) 
df['tm']

df['H'].replace(to_replace = np.nan, value = 64.27, inplace=True) 
df['H']

df['P'].replace(to_replace = np.nan, value = 3.43, inplace=True) 
df['P']

df['VV'].replace(to_replace = np.nan, value = 6.58, inplace=True) 
df['VV']

df['V'].replace(to_replace = np.nan, value = 4.14, inplace=True) 
df['V']

df['VM'].replace(to_replace = np.nan, value = 7.45, inplace=True) 
df['VM']

df['VM'].replace(to_replace = np.nan, value = 7.45, inplace=True) 
df['VM']


plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df.drop(df.columns[0],axis=1,inplace=True)
df.columns
df.corr()
#convert object dtypes to numeric
for col in df.columns:
    df[col] = np.where(df[col] == '-', 0 ,df[col])
    df[col] = pd.to_numeric(df[col],errors ='coerce').fillna(0).astype(np.float64)
    
df.corr()
##Linear Regression for AQI

##Scaling
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),yticklabels=False,cbar=False,cmap='viridis',annot=True)


X = df.iloc[:,:-1]
y = df.iloc[:,-1]



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X= scaler.fit_transform(X)

new_X = pd.DataFrame(scaled_X,columns=X.columns)
new_X.head



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.33, random_state=42)


#check r2 score accuracy for Train data
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))

#check r2 score accuracy for Test data
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor()
model.fit(X_test,y_test)
print(model.score(X_test,y_test))

print(model.feature_importances_)
imp_feat = pd.Series(model.feature_importances_,index=X.columns)
imp_feat.nlargest(5).plot(kind='barh')
plt.show()


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

prediction = lm.predict(X_test)

plt.figure(figsize=(10,7))
sns.distplot(y_test-prediction)

from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,prediction)
print(MAE)
MSE = metrics.mean_squared_error(y_test,prediction)
print(MSE)
RMSE = np.sqrt(MSE)
print(RMSE)

lm.coef_
print(lm.coef_)

lm.intercept_
print(lm.intercept_)

df_coef = pd.DataFrame(lm.coef_,index=X.columns,columns=['Coefficient'])
df_coef

plt.figure(figsize=(10,7))
plt.scatter(y_test,prediction)


import pickle
#open a file you eant to store data
file = open('linear_model.pkl','wb')

#dump input to that file
pickle.dump(lm,file)




























