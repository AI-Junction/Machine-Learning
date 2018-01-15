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


path_train = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllPPEData\\train_1.csv"

# Load the data
train = pd.read_csv(path_train)
#print(train.columns)



train_record = train[:1]

#%%


train_flattened = pd.melt(train_record[list(train_record.columns[-100:])+['Page']], id_vars='Page', var_name='date', value_name='CPULevel')

#print(train_flattened.columns)
train_flattened.columns = ['CPU', 'date', 'CPULevel']
train_flattened.CPU = 'CPU'
#print(train_flattened)



train_flattened_sorted =  train_flattened.sort(['CPULevel'], ascending = True).reset_index()
ninety_percentile = np.percentile(train_flattened_sorted.CPULevel, 90)
loc = train_flattened[train_flattened.CPULevel > ninety_percentile].index
train_flattened.loc[loc, 'CPULevel'] = ninety_percentile
#print(train_flattened.CPULevel)

train_flattened['CPULevel'] = train_flattened['CPULevel'].apply (lambda x: (x*100)/max(train_flattened['CPULevel']))
#print(np.percentile(train_flattened.CPULevel, 100))
train_flattened_sorted =  train_flattened.sort(['CPULevel'], ascending = True).reset_index()



plt.figure(figsize = (16,8))
plt.scatter(range(train_flattened_sorted.shape[0]), train_flattened_sorted['CPULevel'])
plt.show()


plt.figure(figsize = (16,8))
plt.plot(range(train_flattened.shape[0]), train_flattened['CPULevel'])
plt.show()

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)


#%%

train_flattened.reset_index(drop=False,inplace=True)


#%%


train_flattened['weekday'] = train_flattened['date'].apply(lambda x: x.weekday())
#print(train_flattened[:10])

#%%

# Feature engineering with the date
train_flattened['year']=train_flattened.date.dt.year 
train_flattened['month']=train_flattened.date.dt.month 
train_flattened['day']=train_flattened.date.dt.day

#%%

train_flattened.head()
#print(train_flattened.shape)


#%%

"""
This part allowed us to prepare our data. 
We had created new features that we use in the next steps. 
Days, Months, Years are interesting to forecast with a 
Machine Learning Approach or to do an analysis. 

"""


#%%


# For the next graphics
train_flattened['month_num'] = train_flattened['month']

train_flattened['weekday_num'] = train_flattened['weekday']
train_flattened['weekday'].replace(0,'01 - Monday',inplace=True)
train_flattened['weekday'].replace(1,'02 - Tuesday',inplace=True)
train_flattened['weekday'].replace(2,'03 - Wednesday',inplace=True)
train_flattened['weekday'].replace(3,'04 - Thursday',inplace=True)
train_flattened['weekday'].replace(4,'05 - Friday',inplace=True)
train_flattened['weekday'].replace(5,'06 - Saturday',inplace=True)
train_flattened['weekday'].replace(6,'07 - Sunday',inplace=True)

#print(train_flattened[:10])

#%%

train_group = train_flattened.groupby(["month", "weekday"])['CPULevel'].mean().reset_index()
train_group = train_group.pivot('weekday','month','CPULevel')
train_group.sort_index(inplace=True)

#print(train_group)

#%%

sns.set(font_scale=1) 

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(50, 30))
sns.heatmap(train_group, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('CPU Levels cross Weekdays')
plt.show()

"""
This heatmap show us in average the CPU Values by 
weekdays cross the months. 
"""

#%%

train_day = train_flattened.groupby(["month", "day"])['CPULevel'].mean().reset_index()
train_day = train_day.pivot('day','month','CPULevel')
train_day.sort_index(inplace=True)
train_day = train_day.fillna(train_day.bfill())

#%%

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(50, 30))
sns.heatmap(train_day, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('CPU Levels Months cross days')
plt.show()

"""
With this graph it is possible to see if there are any periods 
with a bigger activity than the rest. 

"""

#%%

"""

III. ML Approach

The first approach introduces is the Machine Learnin Approach. 
We will use just a AdaBoostRegressor 

"""
times_series_means =  pd.DataFrame(train_flattened).reset_index(drop=False)
times_series_means['weekday'] = times_series_means['date'].apply(lambda x: x.weekday())
times_series_means['Date_str'] = times_series_means['date'].apply(lambda x: str(x))
times_series_means[['year','month','day']] = pd.DataFrame(times_series_means['Date_str'].str.split('-',2).tolist(), columns = ['year','month','day'])
date_staging = pd.DataFrame(times_series_means['day'].str.split(' ',2).tolist(), columns = ['day','other'])
times_series_means['day'] = date_staging['day']*1
times_series_means.drop('Date_str',axis = 1, inplace =True)
times_series_means.head()



#%%

"""
The first step for the ML approach is to create the feature 
that we will predict. In our example we don't predict 
the number of CPULevel but the difference between two days. 
The tips to create few features is to take the difference 
between two days and to do a lag. Here we will take a 
lag of "diff" seven times. If you have a weekly pattern 
it is an interesting choice. Here we have few 
data (2 months so 30 values) and it is a contraint. 
I done some test and the number 7 is a good 
choice (weekly pattern?).

"""
times_series_means.reset_index(drop=True,inplace=True)

def lag_func(data,lag):
    lag = lag
    X = lagmat(data["diff"], lag)
    
    lagged = data.copy()
    
    for c in range(1,lag+1):
        lagged["lag%d" % c] = X[:, c-1]
    
#    print(lagged[:10])
    return lagged

def diff_creation(data):
    data["diff"] = np.nan
    data.loc[1:, "diff"] = (data.loc[1:, "CPULevel"].as_matrix() - data.loc[:len(data)-2, "CPULevel"].as_matrix())
    return data

df_count = diff_creation(times_series_means)

# Creation of 30 features with "diff"

lag = 15
lagged = lag_func(df_count,lag)

last_date = lagged['date'].max()
#print(last_date)


#%%

#print(["lag%d" % i for i in range(1,16)] + ['weekday'] + ['day'])
#print(lagged.columns.values)


# Train Test split
def train_test(data_lag):
    xc = ["lag%d" % i for i in range(1,lag+1)] + ['weekday'] + ['day']
    split = 0.70
    xt = data_lag[(lag+1):][xc]
    yt = data_lag[(lag+1):]["diff"]
    isplit = int(len(xt) * split)
    x_train, y_train, x_test, y_test = xt[:isplit], yt[:isplit], xt[isplit:], yt[isplit:]
    return x_train, y_train, x_test, y_test, xt, yt

x_train, y_train, x_test, y_test, xt, yt = train_test(lagged)


#%%

# Linear Model
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def modelisation(x_tr, y_tr, x_ts, y_ts, xt, yt, model0, model1):
    # Modelisation with all product
    model0.fit(x_tr, y_tr)

    prediction = model0.predict(x_ts)
    r2 = r2_score(y_ts.as_matrix(), model0.predict(x_ts))
    mae = mean_absolute_error(y_ts.as_matrix(), model0.predict(x_ts))
    print ("-----------------------------------------------")
    print ("mae with 70% of the data to train:", mae)
    print ("R2 Score:", r2)
    print ("-----------------------------------------------")

    # Model with all data
    model1.fit(xt, yt) 
    
    return model1, prediction, model0

model0 =  AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)
model1 =  AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)

clr, prediction, clr0  = modelisation(x_train, y_train, x_test, y_test, xt, yt, model0, model1)

#print(prediction[:10])
#print(prediction.shape)
#print(type(prediction))



#%%


# Performance 1
plt.style.use('ggplot')
plt.figure(figsize=(50, 12))
line_up, = plt.plot(prediction,label='Prediction')
line_down, = plt.plot(np.array(y_test),label='Reality')
plt.ylabel('Series')
plt.legend(handles=[line_up, line_down])
plt.title('Performance of predictions - Benchmark Predictions vs Reality')
plt.show()



#%%


# Prediction
def pred_df(data,number_of_days):
    data_pred = pd.DataFrame(pd.Series(data["date"][data.shape[0]-1] + timedelta(days=1)),columns = ["date"])
#    print(data_pred[:10])
#    print(pd.Series(data["date"][:10]))
    for i in range(number_of_days):
        inter = pd.DataFrame(pd.Series(data["date"][data.shape[0]-1] + timedelta(days=i+2)),columns = ["date"])
        data_pred = pd.concat([data_pred,inter]).reset_index(drop=True)
    return data_pred

data_to_pred = pred_df(df_count,30)
#print(data_to_pred)
#print(df_count)


#%%


def initialisation_v1(data_lag, data_pred, model, xtrain, ytrain, number_of_days):
    # Initialisation
    model.fit(xtrain, ytrain)
    
    for i in range(number_of_days-1):
        lag1 = data_lag.tail(1)["diff"].values[0]
        lag2 = data_lag.tail(1)["lag1"].values[0]
        lag3 = data_lag.tail(1)["lag2"].values[0]
        lag4 = data_lag.tail(1)["lag3"].values[0]
        lag5 = data_lag.tail(1)["lag4"].values[0]
        lag6 = data_lag.tail(1)["lag5"].values[0]
        lag7 = data_lag.tail(1)["lag6"].values[0]
        lag8 = data_lag.tail(1)["lag7"].values[0]
        
#        if i < 5:
#            print('lag values for i = ', i, ':', lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8)
        
        data_pred['weekday'] = data_pred['date'].apply(lambda x:x.weekday())
        weekday = data_pred['weekday'][0]
        
        row = pd.Series([lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8,weekday]
                        ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
        to_predict = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
        prediction = pd.DataFrame(columns = ['diff'])
        to_predict = to_predict.append([row])
        prediction = pd.DataFrame(model.predict(to_predict),columns = ['diff'])

        # Loop
        if i == 0:
            last_predict = data_lag["CPULevel"][data_lag.shape[0]-1] + prediction.values[0][0]

        if i > 0 :
#            print('last_predict before assignment: ', data_lag["CPULevel"][data_lag.shape[0]-1])
#            print('prediction.values[0][0]: ', prediction.values[0][0])
            last_predict = data_lag["CPULevel"][data_lag.shape[0]-1] + prediction.values[0][0]
#            print('last_predict after assignment: ', last_predict)
        data_lag = pd.concat([data_lag,prediction.join(data_pred["date"]).join(to_predict)]).reset_index(drop=True)
        data_lag["CPULevel"][data_lag.shape[0]-1] = last_predict
        
        data_pred = data_pred[data_pred["date"]>data_pred["date"][0]].reset_index(drop=True)
        
    return data_lag



#%%

#%%

def initialisation_v2(data_lag, data_pred, model, xtrain, ytrain, number_of_days):

    model.fit(xt, yt)

    data_pred['weekday'] = data_pred['date'].apply(lambda x:x.weekday())
    
    for i in range(number_of_days-1):
        lag1 = data_lag.tail(1)["diff"].values[0]
        lag2 = data_lag.tail(1)["lag1"].values[0]
        lag3 = data_lag.tail(1)["lag2"].values[0]
        lag4 = data_lag.tail(1)["lag3"].values[0]
        lag5 = data_lag.tail(1)["lag4"].values[0]
        lag6 = data_lag.tail(1)["lag5"].values[0]
        lag7 = data_lag.tail(1)["lag6"].values[0]
        lag8 = data_lag.tail(1)["lag7"].values[0]
        
#        print('lag values for i = ', i, ':', lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8)
        
        weekday = data_pred['weekday'][0]
        
        row = pd.Series([lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8,weekday]
                        ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
        
#        print(row)
        
        to_predict = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
        prediction = pd.DataFrame(columns = ['diff'])
        to_predict = to_predict.append([row])
        prediction = pd.DataFrame(model.predict(to_predict),columns = ['diff'])
        
        last_predict = data_lag["CPULevel"][data_lag.shape[0]-1] + prediction.values[0][0]
#        print(last_predict)
#        print(data_pred['date'][0])
        row['date'] = ''
        row['date'] = data_pred['date'][0]
        row['diff'] = prediction.values[0][0]
        row['CPULevel'] = last_predict
        data_lag = data_lag.append([row], ignore_index = True)
        data_pred = data_pred[data_pred["date"]>data_pred["date"][0]].reset_index(drop=True)

    return data_lag


    
#%%


def initialisation_v3(data_lag, data_pred, model, xtrain, ytrain, number_of_days):

    model.fit(xt, yt)

    data_pred['weekday'] = data_pred['date'].apply(lambda x:x.weekday())
    
    for i in range(number_of_days-1):
        lag1 = data_lag.tail(1)["diff"].values[0]
        lag2 = data_lag.tail(1)["lag1"].values[0]
        lag3 = data_lag.tail(1)["lag2"].values[0]
        lag4 = data_lag.tail(1)["lag3"].values[0]
        lag5 = data_lag.tail(1)["lag4"].values[0]
        lag6 = data_lag.tail(1)["lag5"].values[0]
        lag7 = data_lag.tail(1)["lag6"].values[0]
        lag8 = data_lag.tail(1)["lag7"].values[0]

        lag9 = data_lag.tail(1)["lag8"].values[0]
        lag10 = data_lag.tail(1)["lag9"].values[0]
        lag11 = data_lag.tail(1)["lag10"].values[0]
        lag12 = data_lag.tail(1)["lag11"].values[0]
        lag13 = data_lag.tail(1)["lag12"].values[0]
        lag14 = data_lag.tail(1)["lag13"].values[0]
        lag15 = data_lag.tail(1)["lag14"].values[0]
        lag16 = data_lag.tail(1)["lag15"].values[0]
        
#        print('lag values for i = ', i, ':', lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8, lag9, lag10, lag11, lag12, lag13, lag14, lag15, lag16)
        
        weekday = data_pred['weekday'][0]
        
        row = pd.Series([lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8,lag9, lag10, lag11, lag12, lag13, lag14, lag15, lag16, weekday]
                        ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','lag9', 'lag10', 'lag11', 'lag12', 'lag13', 'lag14', 'lag15', 'lag16','weekday'])
        
#        print(row)
        
        to_predict = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','lag9', 'lag10', 'lag11', 'lag12', 'lag13', 'lag14', 'lag15', 'lag16','weekday'])
        prediction = pd.DataFrame(columns = ['diff'])
        to_predict = to_predict.append([row])
        prediction = pd.DataFrame(model.predict(to_predict),columns = ['diff'])
        
        last_predict = data_lag["CPULevel"][data_lag.shape[0]-1] + prediction.values[0][0]
#        print(last_predict)
#        print(data_pred['date'][0])
        row['date'] = ''
        row['date'] = data_pred['date'][0]
        row['diff'] = prediction.values[0][0]
        row['CPULevel'] = last_predict
        data_lag = data_lag.append([row], ignore_index = True)
        data_pred = data_pred[data_pred["date"]>data_pred["date"][0]].reset_index(drop=True)

    return data_lag


    
    
#%%
model_fin = AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)

#%%

lagged = initialisation_v3(lagged, data_to_pred, model_fin, xt, yt, 30)


#%%


#%%

df_lagged = lagged[['CPULevel','date']]
df_train = df_lagged[df_lagged['date'].astype('datetime64[ns]') <= last_date]
df_pred = df_lagged[df_lagged['date'].astype('datetime64[ns]') >= last_date]

#print(df_pred)

dfout = pd.rolling_mean(df_lagged['CPULevel'], window=5, min_periods=1, center=False).to_frame().reset_index()

plt.style.use('ggplot')
plt.figure(figsize=(30, 5))
plt.plot(df_train.date,df_train.CPULevel)
plt.plot(df_pred.date,df_pred.CPULevel,color='b')
plt.title('Training time series in red, Prediction on 30 days in blue -- ML Approach')
plt.show()

dfout.columns = ['count', 'rolling_means']
#print(dfout)

plt.figure(figsize = (16,8))
dfout.plot.scatter('count','rolling_means', c='g')


"""

The ML method requires a lot of work. You need 
to create the features, the data to collect 
the prediction, optimisation etc... 
This method done a good results when there 
are a weekly pattern identified or a 
monthly pattern but we need more data.


"""

#%%

"""

V. ARIMA


"""

# Show Rolling mean, Rolling Std and Test for the stationnarity
df_date_index = times_series_means[['date','CPULevel']].set_index('date')

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
    dftest = sm.tsa.adfuller(timeseries['CPULevel'], autolag='AIC')
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
dftest = sm.tsa.adfuller(df_date_index['CPULevel'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)





#%%



"""
Our Time Series is stationary. 
We can to apply the ARIMA Model without transformations.

We can apply ARIMA Model

"""
# Naive decomposition of our Time Series as explained above

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
We take a simple decomposition on three parts. 
The additive model is Y[t] = T[t] + S[t] + e[t] 
The multiplicative model is Y[t] = T[t] x S[t] x e[t] with:

    T[t]: Trend
    S[t]: Seasonality
    e[t]: Residual



"""
#%%
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_date_index, order=(2, 1, 1))  
results_AR = model.fit(disp=-1) 

results_AR_revised = (results_AR.fittedvalues*-1)+50

plt.figure(figsize = (16,8))
plt.plot(df_date_index, color = 'blue')
#plt.plot(results_AR.fittedvalues, color='red')
plt.plot(results_AR_revised, color='red')
plt.show()


#print(results_AR.fittedvalues)
#print(results_AR_revised)             
#print(df_date_index.columns.values)

#%%
# ACF plot

f = plt.figure(figsize=(15,6))
autocorrelation_plot(df_date_index)
plt.show()

df_date_index.columns

#%%

#def GetARIMApdq(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    
#col = 'CPULevel'
#timefreq = 'd'
X = df_date_index.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonality = 7
seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
print(seasonal_pdq)
                
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
#    return pdq, seasonal_pdq, train, test


#%%

# BestFitARIMA

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
                
#            print('ARIMA{}x{}{} - AIC:{}'.format(param, param_seasonal, seasonality, results.aic))
        except:
#                print('in exception')
            continue
#    df_temp.columns = ['param','param_seasonal','AIC_Result']
df_temp['param'] = param_temp
df_temp['param_seasonal'] = param_seasonal_temp
df_temp['AIC_Result'] = results_temp
df_pdq = df_temp
#return df_temp

param_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param
param_seasonal_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param_seasonal

#%%

#def ARIMAResultsShow(train, pdq_order, pdq_seasonal_order, seasonality):
# uncomment later 
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

#    mod = sm.tsa.statespace.SARIMAX(train,
#                                    order=(1,1,1),
#                                    seasonal_order=(1,1,1, 12),
#                                    enforce_stationarity=False,
#                                    enforce_invertibility=False)



print('after mod')

results = mod.fit()

#print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()
print("before pred")





#%%
#def ARIMAPredict(train2, results_arg, predict_start):    

print('within ARIMAPredict')
results_arg = results
train2 = df_date_index

#print(results_arg)
pred = results_arg.get_prediction(dynamic=False)
#print(pred.predicted_mean)
#print(type(pred.predicted_mean))
#print(train2)
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
#return pred

#%%

#def forecast(forecaststartdate, pred1, results2, train2, xlabel, ylabel):  
forecaststartdate = max(train2.index)+timedelta(days = 1)
print(forecaststartdate)
#y_forecasted = pred.predicted_mean
#y_truth = train2.loc[forecaststartdate:]

#print(train2)

# Compute the mean square error
#mse = ((y_forecasted - y_truth) ** 2).mean()
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#pred_dynamic = results.get_prediction(start=pd.to_datetime(forecaststartdate), dynamic=True, full_results=True)
#pred_dynamic = results.get_prediction(start = None, dynamic=True, full_results=True)
#pred_dynamic_ci = pred_dynamic.conf_int()
#
#ax.set_ylim([100,100])
#ax = train2['CPULevel'].plot(label='observed', figsize=(20, 15))
#pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
#
#ax.fill_between(pred_dynamic_ci.index,
#                pred_dynamic_ci.iloc[:, 0],
#                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
#
#ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(forecaststartdate), train2.index[-1],
#                 alpha=.1, zorder=-1)
#
#ax.set_xlabel('xlabel')
#ax.set_ylabel('ylabel')
#
#plt.legend()
#plt.show()


# Extract the predicted and true values of our time series
#y_forecasted = pred_dynamic.predicted_mean
#y_truth = train2[forecaststartdate:]
#
## Compute the mean square error
#mse = ((y_forecasted - y_truth) ** 2).mean()
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



# Get forecast 500 steps ahead in future
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
ax.set_ylabel('CPULevel')

plt.legend()
plt.show()



