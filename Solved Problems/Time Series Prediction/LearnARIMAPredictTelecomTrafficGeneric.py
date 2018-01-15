# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:23:26 2017

@author: echtpar
"""

#%%
import pandas as pd
import numpy as np
import itertools
import warnings
import scipy
from datetime import timedelta

# Forceasting with decompasable model
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot


# For marchine Learning Approach
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Visualisation
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

from datetime import datetime
import datetime as dt

plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')

#%%

"""

# Define data path and define generic parameters of the time series
# This is the only input / configuration needed from the user

"""


path_train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllPPEData\\combined_101.csv"
ColName = 'smsIn'
TFreq = 'H'
StartDate = '11/1/2013'
MultFactor = 1



#%%

# Load data

train = pd.read_csv(path_train)
print(train.columns)
print(train.head())


#%%

train = train.loc[:, ColName].to_frame()
train.index = pd.date_range(start = StartDate, periods = len(train), freq = TFreq)


train.columns = [ColName]
print(train.columns)

train = train.fillna(train.bfill())
train[ColName] = train[ColName]*MultFactor


plt.figure(figsize = (16,8))
plt.plot(train.index, train[ColName])
plt.show()

train['date'] = train.index
train['date'] = train['date'].astype('datetime64[ns]')
train['weekend'] = ((train.date.dt.dayofweek) // 5 == 1).astype(float)

print(train.columns.values)
print(train.head(50))

#%%

train['weekday'] = train['date'].apply(lambda x: x.weekday())
#print(train[:10])

#%%

# Feature engineering with the date
train['year']=train.date.dt.year 
train['month']=train.date.dt.month 
train['day']=train.date.dt.day

#%%

train.head()


#%%

# Show Rolling mean, Rolling Std and Test for the stationnarity
df_date_index = train[['date',ColName]].set_index('date')

print(df_date_index)
df_date_index = df_date_index.fillna(df_date_index.bfill())

#print(df_date_index)

def test_stationarity(timeseries):
    plt.figure(figsize=(50, 8))
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=7)
    rolstd = pd.rolling_std(timeseries, window=7)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = sm.tsa.adfuller(timeseries[ColName], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
#    print(dfoutput)
    
# test_stationarity using Dickey Fuller test

plt.figure(figsize=(50, 8))
#Determing rolling statistics
rolmean = pd.rolling_mean(df_date_index, window=7)
rolstd = pd.rolling_std(df_date_index, window=7)

#Plot rolling statistics:
orig = plt.plot(df_date_index, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = sm.tsa.adfuller(df_date_index[ColName], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)




#%%


"""
We take a simple decomposition on three parts. 
The additive model is Y[t] = T[t] + S[t] + e[t] 
The multiplicative model is Y[t] = T[t] x S[t] x e[t] with:

    T[t]: Trend
    S[t]: Seasonality
    e[t]: Residual



"""


decomposition = sm.tsa.seasonal_decompose(df_date_index, model='multiplicative',freq = 7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
rcParams['figure.figsize'] = 30, 20

plt.subplot(411)
plt.title('Obesered = Trend + Seasonality + Residuals')
plt.plot(df_date_index, label='Observed')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%

"""

We can apply ARIMA Model

"""


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_date_index, order=(1, 0, 1))  
results_AR = model.fit(disp=-1) 

plt.figure(figsize = (16,8))
plt.plot(df_date_index, color = 'blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.show()


#%%
# ACF plot

f = plt.figure(figsize=(15,6))
autocorrelation_plot(df_date_index)
plt.show()

df_date_index.columns

#%%

# Generate a range of p, d q values and work out all permutations of p, d, q in this range.

X = df_date_index.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonality = 24
seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
print(seasonal_pdq)
                
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    


#%%

# Here we identify the p, d, q values to determine the Best Fit ARIMA. 
# The best combination of p, d, q is selected by finding the ARIMA model that generates the least AIC value.
# AIC stands for Akaike Information Criteria

train = df_date_index
warnings.filterwarnings("ignore") # specify to ignore warning messages
df_temp = pd.DataFrame()
param_temp = []
param_seasonal_temp = []
results_temp = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            
            param_temp.append(param)
            param_seasonal_temp.append(param_seasonal)
            results_temp.append(results.aic)
                
        except:
            continue
df_temp['param'] = param_temp
df_temp['param_seasonal'] = param_seasonal_temp
df_temp['AIC_Result'] = results_temp
df_pdq = df_temp

param_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param
param_seasonal_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param_seasonal

#%%

#Now train the ARIMA model using the best fit p, d, q values

pdq_order = param_temp
pdq_seasonal_order = param_seasonal_temp
   
p = list(pdq_order)[0][0]
d = list(pdq_order)[0][1]
q = list(pdq_order)[0][2]


print('ARIMA p, d, q', p,d,q)

p_season = list(pdq_seasonal_order)[0][0]
d_season = list(pdq_seasonal_order)[0][1]
q_season = list(pdq_seasonal_order)[0][2]

print('SARIMAX p,d,q', p_season,d_season,q_season)  
print('before mod')

mod = sm.tsa.statespace.SARIMAX(train,
                                order=(p,d,q),
                                seasonal_order=(p_season, d_season, q_season, seasonality),
                                enforce_stationarity=False,
                                enforce_invertibility=False)


print('after mod')

results = mod.fit()

#print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()
print("before pred")





#%%

# Generate the one step forecast using the trained ARIMA model as above.

print('within ARIMAPredict')
results_arg = results
train2 = df_date_index

pred = results_arg.get_prediction(dynamic=False)
pred_ci = pred.conf_int()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_ylim([100,100])
ax1 = train2.plot(label='observed')
pred.predicted_mean.plot(ax=ax1, label='One-step ahead Forecast', alpha=.7)
ax1.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax1.set_xlabel('Date')
ax1.set_ylabel('Quantity or Value')
plt.legend()
plt.show()    

#%%

# Now Get forecast for future steps (30) ahead in future

forecaststartdate = max(train2.index)+timedelta(days = 1)
print(forecaststartdate)

pred_uc = results.get_forecast(steps=60)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

fig1 = plt.figure()
ax = train2.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel(ColName)

plt.legend()
plt.show()



