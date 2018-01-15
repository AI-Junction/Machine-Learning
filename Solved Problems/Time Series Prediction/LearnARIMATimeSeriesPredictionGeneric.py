# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:28:39 2017

@author: echtpar
"""
##################################
"""

Generic ARIMA / SARIMAX model for any timeseries data

"""
##################################

import warnings
import itertools
import numpy as np
from pylab import rcParams

import pandas as pd
from pandas import read_csv
from pandas import datetime
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame

import matplotlib.pyplot as plt
from matplotlib import pyplot

import statsmodels as sm1
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def timeseriesview(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):

    plt.style.use('fivethirtyeight')

    X  = series[col].resample(timefreq).mean()
    X = X.fillna(X.bfill())

    X.plot(figsize=(15, 6))
    plt.show()
    
#    f, ax = plt.subplots(1,1, figsize=(15,6))
    f = plt.figure(figsize=(15,6))
    autocorrelation_plot(X)
    plt.show()
    return X

def GetARIMApdq(series, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
#    print(train)
#    print(test)
    history = [x for x in train]
#    print(history)
    predictions = list()

    print("Step 1")

#    predictions = []

    # forward walk
    for t in range(len(test)):
#        print("within t loop")
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        
    error = mean_squared_error(test, predictions)
    r2score = r2_score (test, predictions)
    print('Test MSE: %.3f' % error)
    print('Test R2 Score: %.3f' % r2score)
    
    # plot
    pyplot.plot(test, color = 'blue')
    pyplot.plot(predictions, color='red')
    pyplot.show()

    p = d = q = range(0, 2)
    
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    print(pdq)
    
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
    print(seasonal_pdq)
                    
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    return pdq, seasonal_pdq, train, test

def BestFitARIMA(train, pdq, seasonal_pdq, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    
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
                    
                print('ARIMA{}x{}{} - AIC:{}'.format(param, param_seasonal, seasonality, results.aic))
            except:
#                print('in exception')
                continue
#    df_temp.columns = ['param','param_seasonal','AIC_Result']
    df_temp['param'] = param_temp
    df_temp['param_seasonal'] = param_seasonal_temp
    df_temp['AIC_Result'] = results_temp
    return df_temp
        
    
def ARIMAResultsShow(train, pdq_order, pdq_seasonal_order, seasonality):
# uncomment later    
    p = list(pdq_order)[0][0]
    d = list(pdq_order)[0][1]
    q = list(pdq_order)[0][2]

    print(p,d,q)

    p_season = list(pdq_seasonal_order)[0][0]
    d_season = list(pdq_seasonal_order)[0][1]
    q_season = list(pdq_seasonal_order)[0][2]

    print(p_season,d_season,q_season)  
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

    print(results.summary().tables[1])

#    model = ARIMA(y, order=(8,1,0), freq = 'A')
#    model_fit = model.fit(disp=0)
#    print(model_fit.summary())
#    # plot residual errors
#    residuals = DataFrame(model_fit.resid)
#    residuals.plot()
#    pyplot.show()
#    residuals.plot(kind='kde')
#    pyplot.show()
#    print(residuals.describe())
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    print("before pred")
#    start = None
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
#    print(pred)
#    print(pred)
    return results


def ARIMAPredict(train2, results_arg, predict_start):    
    print('within ARIMAPredict')
    print(results_arg)
#    print(pd.to_datetime(predict_start_date))
#    startdatetime = pd.to_datetime(predict_start_date)
#    pred = results.get_prediction(start=pd.to_datetime(predict_start_date), dynamic=False)
#    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
    pred = results_arg.get_prediction(dynamic=False)
    print(pred.predicted_mean)
    print(type(pred.predicted_mean))
    print(train2)
#    fig = plt.figure(figsize = (8, 16))
#    pred = results.get_prediction(dynamic=False)
    pred_ci = pred.conf_int()
#    ax = train['1990':].plot(label='observed')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_ylim([300,380])
#    ax = axes.flatten()
    ax1 = train2.plot(label='observed')
#    df_pred = pd.DataFrame(pred.predicted_mean)
#    z = np.arange(predict_start, len(df_pred)+predict_start)
#    print(z)
#    df_pred.index = z
#    print(len(df_pred))
#    print(df_pred.index)
    pred.predicted_mean.plot(ax=ax1, label='One-step ahead Forecast', alpha=.7)
#    print(pd.DataFrame(pred.predicted_mean))
#    pred_ci.index = z
    ax1.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Quantity or Value')
    plt.legend()
    
    plt.show()    
    return pred

def forecast(forecaststartdate, pred1, results2, train2, xlabel, ylabel):    
    y_forecasted = pred1.predicted_mean
    y_truth = train2.loc[forecaststartdate:]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    pred_dynamic = results2.get_prediction(start=pd.to_datetime(forecaststartdate), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()


    ax = train2['1990':].plot(label='observed', figsize=(20, 15))
    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(forecaststartdate), train2.index[-1],
                     alpha=.1, zorder=-1)
    
    fig = plt.figure()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.legend()
    plt.show()


    # Extract the predicted and true values of our time series
    y_forecasted = pred_dynamic.predicted_mean
    y_truth = train2[forecaststartdate:]
    
    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



    # Get forecast 500 steps ahead in future
    pred_uc = results2.get_forecast(steps=500)
    
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    
    fig1 = plt.figure()
    ax = train2.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.legend()
    plt.show()
    
    

data = sm.datasets.co2.load_pandas()
y = data.data
print(y)


Z = timeseriesview(y, 'co2', timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12)

print(Z)
    
pdq_res, pdq_seas_res, train1, test1 = GetARIMApdq(Z, 'co2', timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12)

#def BestFitARIMA(train, pdq, seasonal_pdq, col, timefreq='MS', predict_start_date = '1998-01-01', seasonality = 12):    

df_pdq = BestFitARIMA(Z, pdq_res, pdq_seas_res, 'co2', 'MS', '1998-01-01', 12)
print(df_pdq)
param_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param
param_seasonal_temp = df_pdq.loc[df_pdq.AIC_Result == min(df_pdq.AIC_Result)].param_seasonal

#param_temp_list = ([str(param_temp)[7], str(param_temp)[10], str(param_temp)[13]])                                 
#param_seasonal_temp_list = ([str(param_seasonal_temp)[7], str(param_seasonal_temp)[10], str(param_seasonal_temp)[13], str(param_seasonal_temp)[16:18]])                                 
                                 
                                 
print(list(param_temp), list(param_seasonal_temp))                                 
                                 
res = ARIMAResultsShow(Z, param_temp, param_seasonal_temp, 12)

start=pd.to_datetime('1998-01-01')
print(start)

#pred = res.get_prediction(pd.to_datetime('1998-01-01 00:00:00'), dynamic=False)
pred = res.get_prediction(dynamic=False)

z = pd.to_datetime('1998-01-01 00:00:00')

pred_ret = ARIMAPredict(Z, res, z)

print(pred_ret.predicted_mean)

forecast(z, pred_ret, res, Z, 'Date', 'Pollution Levels')    






##################################
"""
Shampoo Sales prediction
http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
"""
##################################



from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
 
def parser(x):
#        print(x)
        return datetime.strptime('190'+x, '%Y-%m')
 
path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllShampooSalesData\\sales-of-shampoo-over-a-three-ye.csv"

series = read_csv(path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = read_csv(path, header=0, parse_dates=[0], index_col=0, squeeze=True)

print(series)

print(series.head())
series.plot()
pyplot.show()

autocorrelation_plot(series)
pyplot.show()


model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print(train)
print(test)
history = [x for x in train]
print(history)
predictions = list()


model = ARIMA(history, order=(5,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast()
print(output)
yhat = output[0]
predictions.append(yhat)
obs = test[t]
history.append(obs)
print('predicted=%f, expected=%f' % (yhat, obs))

predictions = []
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()





##################################

"""

CO2 pollution prediction
https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

"""
##################################


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels as sm1
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot


plt.style.use('fivethirtyeight')

data = sm.datasets.co2.load_pandas()
y = data.data



# The 'MS' string groups the data in buckets by start of the month
y = y['co2'].resample('MS').mean()



# The term bfill means that we use the value before filling in missing values
y = y.fillna(y.bfill())

print(y)
#print(type(y))

y.plot(figsize=(15, 6))
plt.show()

autocorrelation_plot(y)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print(seasonal_pdq)
                
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))



warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            print('in exception')
            continue



        
        
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()


model = None
model_fit = None
residuals = None
model = ARIMA(y, order=(8,1,0), freq = 'A')
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())



print(results.summary().tables[1])
        
results.plot_diagnostics(figsize=(15, 12))
plt.show()


pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

plt.show()


y_forecasted = pred.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

pred_dynamic = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


ax = y['1990':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()


# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))



# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()






co2 = data.data
co2.index

rcParams['figure.figsize'] = 11, 9
q = pd.DataFrame(y).reset_index()
print(type(q))
print(q)
w=None
v=None
w = q.co2.apply(lambda x: str(x)[-3:])
v = q['index']
print(v)
print(w)
z=None
z = pd.concat([v,w], axis = 1).reset_index(False)
print(z)
print(q['index'])


decomposition = sm.tsa.seasonal_decompose(z, model='additive')
fig = decomposition.plot()
plt.show()










##################################

"""

Champagne Sales prediction
http://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/

"""
##################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic


path = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllClimateChange-ARIMA\\GlobalLandTemperaturesByCountry.csv"

# File is in csv format so first we have to read data from csv file using following code.
#df = pd.read_csv('GlobalLandTemperaturesByCountry.csv',                 
#                   sep=',',                 
#                   skipinitialspace=True,                 
#                   encoding='utf-8')

df = pd.read_csv(path,                 
                   sep=',',                 
                   skipinitialspace=True,                 
                   encoding='utf-8')




#We must clean values with care otherwise they will create unexpected results. 
#To to this:
df = df.drop('AverageTemperatureUncertainty', axis=1)
df = df[df.Country == 'Canada']
df = df.drop('Country', axis=1)
df = df[df.AverageTemperature.notnull()]

df.index = pd.to_datetime(df.dt)
df = df.drop('dt', axis=1)
df = df.ix['1900-01-01':]
df = df.sort_index()
df.AverageTemperature.fillna(method='pad', inplace=True)


#At this point you might want to see how the plot looks like. 
#Here is the code for that:

plt.plot(df.AverageTemperature)
plt.show()


plt.scatter(df.index, df.AverageTemperature)
plt.show()


# Now lets see the moving average of our timeseries data.
# Rolling Mean/Moving Average
df.AverageTemperature.plot.line(style='b', legend=True, grid=True, 
                                 label='Avg. Temperature (AT)')
ax = df.AverageTemperature.rolling(window=12).mean().plot.line(style='r', 
                                   legend=True, label='Mean AT')
ax.set_xlabel('Date')
plt.legend(loc='best')
plt.title('Temperature Timeseries Visualization')
plt.show()




def stationarity_test(df):
    dftest = adfuller(df)
    print ('Results of Dickey-Fuller Test:', dftest)
    indices = ['Test Statistic', 'p-value',
            'No. Lags Used', 'Number of Observations Used']
    output = pd.Series(dftest[0:4], index=indices)
    for key, value in dftest[4].items():
        output['Critical Value (%s)' % key] = value
    print (output)


stationarity_test(df.AverageTemperature)


# The model is usually referred to as the ARMA(p,q) model where 
# p is the order of the autoregressive part and 
# q is the order of the moving average part.

# Determining this p and q value can be a challenge. 
# So, pandas has a function for finding this. 
# To get the p and q value -

print (arma_order_select_ic(df.AverageTemperature, 
         ic=['aic', 'bic'], trend='nc', 
         max_ar=4, max_ma=4, 
         fit_kw={'method': 'css-mle'}))



#Lets fit the model and make prediction using ARMA.
# Fit the model
ts = pd.Series(df.AverageTemperature, index=df.index)
model = ARMA(ts, order=(3, 3))
results = model.fit(trend='nc', method='css-mle', disp=-1)
print(results.summary2())


# Now, plot the prediction -
# Plot the model
fig, ax = plt.subplots(figsize=(10, 8))
fig = results.plot_predict('01/01/2010', '12/01/2023', ax=ax)
ax.legend(loc='lower left')
plt.title('Temperature Timeseries Prediction')
plt.show()

predictions = results.predict('01/01/2010', '12/01/2016')

print(predictions)

# Lets predict using another method called ARIMA 
# which is also very popular approach. 
# If you look closely then you will see I used q value 4. 
# I did this because pandas will show error if you use p and q as same value. 
# So, I chose 4. So, lets get going - 

# Fit the model ARIMA

model_arima = ARIMA(ts, order=(3, 1, 4))
results_arima = model_arima.fit(disp=-1, transparams=True)
print(results_arima.summary2())

# Plot the modelfig, ax = plt.subplots(figsize=(10, 8))
fig = results_arima.plot_predict('01/01/2010', '12/01/2016', ax=ax)
ax.legend(loc='lower left')
plt.title('Temperature Timeseries Prediction')
plt.show()

predictions = results_arima.predict('01/01/2010', '12/01/2016')    
print(predictions)