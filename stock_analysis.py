#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:28:47 2019

@author: Wesley Kirui
@goal: Use Git & perform quick analysis
"""

#import libraries
import datetime
import math
import numpy as np
import pandas as pd
#from pandas import Series, DataFrame
import pandas_datareader as pdr
#from pandas_datareader import data, wb
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


#exec('%matplotlib inline')

#set the timeframe to get data for
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2018, 1, 12)

#read the available stock data online for Google
stock_data = pdr.DataReader("GOOG", "yahoo", start, end)

stock_data.tail()

#generate the rolling mean
price_at_close = stock_data['Adj Close']
mov_avg = price_at_close.rolling(window=100).mean()


#plot the series chart
mpl.rc('figure', figsize=(8, 7))
#mpl.__version__
style.use('ggplot')
price_at_close.plot(label='GOOG')
mov_avg.plot(label='mavg')
plt.legend()

#calculate returns on investment & plot the returns over time
return_val = price_at_close/price_at_close.shift(1)-1
return_val.plot(label='return')

#compare competition among the top listed companies in the stock exchange
#'''ie: Google, Microsoft, Apple, GE, IBM, Amazon,Facebook'''
stock_comp = pdr.DataReader(['GOOG', 'GE', 'IBM', 'FB', 'AMZN', 'AAPL',
                             'MSFT'], "yahoo", start, end)['Adj Close']


#check the percentage change among the competitor stocks
comp_rates = stock_comp.pct_change()

#return the correlation between the competitor stocks
corr = comp_rates.corr()

#plot correlation between Amazon & Microsoft
plt.scatter(stock_comp.AMZN, stock_comp.MSFT)
plt.xlabel("Amazon Returns")
plt.ylabel("Microsoft Returns")

#plot scatter matrix to visualize correlation between competitor
#stocks --> Kernel Distribution Estimate (KDE)
pd.plotting.scatter_matrix(stock_comp, diagonal='kde', figsize=(10, 10))

#plot correlation matrix
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

#stock analysis risk and returns
plt.scatter(comp_rates.mean(), comp_rates.std())
plt.xlabel('Expected Returns')
plt.ylabel('Risk')
for label, x, y in zip(comp_rates.columns, comp_rates.mean(),
                       comp_rates.std()):
    plt.annotate(label, xy=(x, y), xytext=(20, -20),
                 textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round, pad=0.5', fc='yellow',
                           alpha=0.3),
                 arrowprops=dict(arrowstyle='->',connectionstyle=
                                 'arc3,rad=0'))             

#price prediction analysis
#feature engineering
stock_datareg = stock_data.loc[:, ['Adj Close', 'Volume']]
stock_datareg['HL_PCT'] = (stock_data['High'] -
                          stock_data['Low'])/stock_data['Close'] * 100.0

stock_datareg['PCT_Change'] = (stock_data['Close']-
                              stock_data['Open'])/stock_data['Open'] * 100.0

#pre-process the data
#fill na values
stock_datareg.fillna(value=-99999, inplace=True)

#separate 1% of the data for forecasting
forecast_data = int(math.ceil(0.01*len(stock_datareg)))

#define column to forecast
forecast_col = 'Adj Close'

stock_datareg['label'] = stock_datareg[forecast_col].shift(-forecast_data)
X = np.array(stock_datareg.drop(['label'], 1))

#scale X for uniform distribution
X = preprocessing.scale(X)

#find data series for late & early X
X_lately = X[-forecast_data:]
X = X[:-forecast_data]

#separate label and identify as y
y = np.array(stock_datareg['label'])
y = y[:-forecast_data]

# Separation of training and testing of model by
# cross validation train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#generate model
#linear regression
stock_reg = LinearRegression(n_jobs= -1)
stock_reg.fit(X_train, y_train)

#quadratic regression 2
stock_poly2 = make_pipeline(PolynomialFeatures(2), Ridge())
stock_poly2.fit(X_train, y_train)
#quadratic regression 3
stock_poly3 = make_pipeline(PolynomialFeatures(3), Ridge())
stock_poly3.fit(X_train, y_train)

#KNN Regression
stock_knn=KNeighborsRegressor(n_neighbors=2)
stock_knn.fit(X_train, y_train)

#evaluate the models
#find the score of the model predictions
confidence_reg = stock_reg.score(X_test, y_test)
confidence_poly2 = stock_poly2.score(X_test, y_test)
confidence_poly3 = stock_poly3.score(X_test, y_test)
confidence_knn = stock_knn.score(X_test, y_test)


#print some of the stocks forecasts to test the model
forecast_set = stock_reg.predict(X_lately)
stock_datareg['Forecast'] = np.nan
print(forecast_set, confidence_reg, forecast_data)

#plot the prediction
last_date= stock_datareg.iloc[-1].name
last_unix= last_date
next_unix= last_unix + datetime.timedelta(days=1)
for i in forecast_set:
    next_date= next_unix
    next_unix+=datetime.timedelta(days = 1)
    stock_datareg.loc[next_date] = [np.nan for _ in
                     range(len(stock_datareg.columns)-1)] +[i]
stock_datareg['Adj Close'].tail(500).plot()
stock_datareg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()