from IPython.display import display # DataFrame Display
import matplotlib.pyplot as plt # Data/Statistics Plot
import seaborn as sns
import numpy as np # Scientific Calculation
import pandas as pd # Data Analysis
from pandas import Series, DataFrame # Data Structure
from sklearn.ensemble import RandomForestRegressor # Training
import sklearn.preprocessing as preprocessing # Scaling
from sklearn import linear_model # Logistic Regression
from sklearn.model_selection import learning_curve # Track Learning Curve

data_train = pd.read_csv("D:\\Kaggle\\HousePrice\\train.csv")

print(data_train.SalePrice[data_train.MSZoning == 'FV'].sum() / data_train.SalePrice[data_train.MSZoning == 'FV'].size)
print(data_train.SalePrice[data_train.MSZoning == 'RH'].sum() / data_train.SalePrice[data_train.MSZoning == 'RH'].size)
print(data_train.SalePrice[data_train.MSZoning == 'RL'].sum() / data_train.SalePrice[data_train.MSZoning == 'RL'].size)
print(data_train.SalePrice[data_train.MSZoning == 'RM'].sum() / data_train.SalePrice[data_train.MSZoning == 'RM'].size)


k = 10 # k most correlated label
plt.figure(figsize=(12,9))
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# scatterplot (cannot contain empty data)
# can be used to detect abnormal values
sns.pairplot(data_train[cols], height = 2.5)
plt.show()

# corr = data_train.corr()
# corr[corr['SalePrice']>0.5]

