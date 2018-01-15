# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:42:27 2017

@author: echtpar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:28:39 2017

@author: echtpar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pathSigma = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllTwo Sigma Financial Modeling Challenge\\train.h5"
pathTemperatures = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllClimateChange-ARIMA\\GlobalLandTemperaturesByCity.csv"

with pd.HDFStore(path, "r") as train:
    df = train.get("train")


print(df.timestamp.head(100))    

timemax = max(df["timestamp"])
timemin = min(df["timestamp"])
xlim = [timemin, timemax]

print(timemax, timemin, xlim)


    
#import kagglegym

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from scipy.stats import norm

# File is in csv format so first we have to read data from csv file using following code.
df = pd.read_csv(pathTemperatures,                 
                   sep=',',                 
                   skipinitialspace=True,                 
                   encoding='utf-8')


df = pd.read_csv(pathTemperatures,                 
#                   parse_dates = ['dt'],              
                   encoding='utf-8')


print(df.dt.head(50))

df.index = pd.to_datetime(df.dt)
df = df.drop('TimeStamp', axis=1)
df = df.ix['2017-01-01':]
print(df.head())
df = df.sort_index()


#At this point you might want to see how the plot looks like. 
#Here is the code for that:

plt.plot(df.NodeTemperature)
plt.show()


# Now lets see the moving average of our timeseries data.
# Rolling Mean/Moving Average
df.NodeTemperature.plot.line(style='b', legend=True, grid=True, 
                                 label='Avg. Temperature (AT)')
ax = df.NodeTemperature.rolling(window=12).mean().plot.line(style='r', 
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


stationarity_test(df.NodeTemperature)


# The model is usually referred to as the ARMA(p,q) model where 
# p is the order of the autoregressive part and 
# q is the order of the moving average part.

# Determining this p and q value can be a challenge. 
# So, pandas has a function for finding this. 
# To get the p and q value -

print (arma_order_select_ic(df.NodeTemperature, 
         ic=['aic', 'bic'], trend='nc', 
         max_ar=4, max_ma=4, 
         fit_kw={'method': 'css-mle'}))



#Lets fit the model and make prediction using ARMA.
# Fit the model
ts = pd.Series(df.NodeTemperature, index=df.index)
model = ARMA(ts, order=(3, 3))
results = model.fit(trend='nc', method='css-mle', disp=-1)
print(results.summary2())


# Now, plot the prediction -
# Plot the model
fig, ax = plt.subplots(figsize=(10, 8))
fig = results.plot_predict('01/07/2017', '31/07/2017', ax=ax)
ax.legend(loc='lower left')
plt.title('Temperature Timeseries Prediction')
plt.show()

predictions = results.predict('01/07/2017', '31/07/2017')



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

predictions = results_arima.predict('01/07/2017', '31/07/2017')    

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

x_train = np.array(df.drop(['NodeTemperature'], axis=1, inplace=True))
y_train = df.NodeTemperature


dtrain = xgb.DMatrix(x_train, y_train)


num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

dtest = xgb.DMatrix(x_test)
y_predict = model.predict(dtest) 