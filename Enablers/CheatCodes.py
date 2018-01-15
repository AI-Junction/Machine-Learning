# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:11:42 2017

@author: echtpar
"""

############################


"""

CHEAT CODES 


References

Analytics Vidhya:
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
https://www.analyticsvidhya.com/blog/2016/01/python-tutorial-list-comprehension-examples/

"""





#############################

"""

IMPORTS

"""

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import BaggingRegressor 
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
import numpy as np


import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *
from PIL import Image


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


### IMPORTS END


"""

NUMPY START

"""

action = np.random.choice(5, 2, p=[0.1,0.2,0.4,0.2,0.1])
print(action)

out = np.random.randn(1)
print(out)

out = np.random.randn(10,2)
print(out)

out = np.random.rand(10,2)
print(out)


out = np.random.randint(2,10, size=(3,4))
print(out)

p = 0.5

x = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]])
print(np.random.rand(*x.shape) < p)
print(np.random.randint(5,4))
print(x.shape)
mask = (np.random.rand(*x.shape) < p) /p 
print(mask)

print(np.mean(np.random.rand(10)), (np.random.rand(10)))
print(*(np.random.rand(5,4)).shape)



from math import sqrt
n = 100
sqrt_n = int(sqrt(n))
no_primes = {j for i in range(2,sqrt_n) for j in range(i*2, n, i)}
print(no_primes)


### NUMPY END






"""

NUMPY MATRIX MULTIPLY START

"""


np.random.seed(10000)
A = np.random.randint(2,10, size=(3,4))
print(A)
print(np.matrix(A))

np.random.seed(10001)
B = np.random.randint(2,10, size=(4,3))
print(B)

C = np.matrix(A)*np.matrix(B)
C_ = A*B


print(C)


D = np.dot(A,B)
print(D)



### NUMPY MATRIX MULTIPLY END


"""

DATES START

"""

from datetime import datetime
import datetime as dt

# ref URL: http://www.marcelscharth.com/python/time.html
# ref URL: https://stackoverflow.com/questions/32168848/how-to-create-a-pandas-datetimeindex-with-year-as-frequency

tmp_date = datetime.strptime('2005-06-01 17:59:00', '%Y-%m-%d %H:%M:%S')
tmp_date = tmp_date.astype('datetime64[ns]')

print(type(tmp_date))
print(tmp_date.date())
print(tmp_date.minute)
print(dt.datetime.today().weekday())
print(tmp_date.weekday())
print(type(tmp_date))

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

print(train_flattened[:10].date)
print(train_flattened[:10].date.dt)
print(train_flattened[:10].date.dt.dayofweek)
print(train_flattened[:10].date.dt.month)
print(train_flattened[:10].date.dt.day)
print(train_flattened[:10].date.dt.year)

train_flattened['year']=train_flattened.date.dt.year 
train_flattened['month']=train_flattened.date.dt.month 
train_flattened['day']=train_flattened.date.dt.day


pickupTime = pd.to_datetime(taxiDB['pickup_datetime'])
taxiDB['src hourOfDay'] = (pickupTime.dt.hour*60.0 + pickupTime.dt.minute)   / 60.0


from statsmodels.tsa.tsatools import lagmat


'''
    's' : second
    'min' : minute
    'H' : hour
    'D' : day
    'w' : week
    'm' : month
    'A' : year 
    'AS' : year start
'''

# months
print(dt.datetime.today())
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'M')
print(z)

#Month Start
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'MS')
print(z)

#seconds
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'S')
print(z)

#min
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'min')
print(z)


#hours
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'H')
print(z)


#Quarterly
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'Q')
print(z)


#Day
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'D')
print(z)


#year end
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'A')
print(z)

#year start
z = pd.date_range(dt.datetime.today(), periods=10, freq = 'AS')
print(z)







### DATES END



"""

ZIP, MAP and LAMBDA


Many novice programmers (and even experienced programmers who are new to python) often get confused when they first see zip, map, and lambda. This post will provide a simple scenario that (hopefully) clarifies how these tools can be used.

To start, assume that you've got two collections of values and you need to keep the largest (or smallest) from each. These could be metrics from two different systems, stock quotes from two different services, or just about anything. For this example we'll just keep it generic.

So, assume you've got a and b: two lists of integers. The goal is to merge these into one list, keeping whichever value is the largest at each index.




"""

a = [1, 2, 3, 4, 5]
b = [2, 2, 9, 0, 9]

#This really isn't difficult to do procedurally. You could write a simple function that compares each item from a and b, then stores the largest in a new list. It might look something like this:

def pick_the_largest(a, b):
    result = []  # A list of the largest values

    # Assume both lists are the same length
    list_length = len(a)
    for i in range(list_length):
        result.append(max(a[i], b[i]))
    return result


#While that's fairly straightforward and easy to read, there is a more concise, more pythonic way to solve this problem.

zip(a, b)

print([x for x in zip(a,b)])

# You now have one list, but it contains pairs of items from a and b. For more information, check out zip in the python documentation.

# lambda is just a shorthand to create an anonymous function. 
# It's often used to create a one-off function (usually for scenarios when you need 
# to pass a function as a parameter into another function). 
# It can take a parameter, and it returns the value of an expression. 
# For more information, see the Python documentation on lambdas.

lambda pair: max(pair)


# map takes a function, and applies it to each item in an iterable (such as a list). 
# You can get a more complete definition of map from the python documentation, 
# but it essentially looks something like this:
    
z = map(  # apply the lambda to each item in the zipped list
        lambda pair: max(pair),  # pick the larger of the pair
        zip(a, b)  # create a list of tuples
    )    

print([x for x in z])


df = pd.DataFrame(np.random.randint(10,size = (4,3)))
print(type(df))
print(df)

df['newcol'] = df[2].apply(lambda x: 2*x)
df['newcol2'] = df['newcol'].map(lambda x: 2*x)

df.columns = ['col1','col2','col3','col4','col5','col6']
print(df.columns)
df['col7'] = df.apply(lambda x: [2*x[0]])
print(df['col7'])


z = df.apply(lambda x: sum(x), axis=0)
print(z)

print(df)
    



### ZIP, MAP and LAMBDA END



"""

STRING OPERATIONS START

"""


a = "01-03-2017"
z = a.split('-', 2)
print(type(z))



import random
sample = list()
n_sample = round(len(prediction) * 0.8)
print(n_sample)
while len(sample) < n_sample:
    index_tmp = random.randrange(len(prediction))
    print('index_tmp', index_tmp) 
    sample.append(prediction[index_tmp])
    print(prediction[index_tmp])



### STRING OPERATIONS END




"""

INDEX OPERATIONS START

"""

dict_test = {'a': [1,2,3,4], 'b' : [5,6,7,8], 'c' : [9,10,11,12]}
dict_test_2 = {'d': [9,10,11,12], 'e' : [5,6,7,8], 'f' : [1,2,3,4]}
print(dict_test_2)    

df_test=None

z = range(10, 14, 1)
print([x for x in z])
df_test = pd.DataFrame(dict_test, index = z)
df_test[['d','e','f']] = pd.DataFrame(dict_test_2, columns = ['d','e','f'], index = df_test.index)
print(df_test)


    
df_test = pd.DataFrame(dict_test, index = z)
print(df_test.index)
print(df_test)

df_test['new_num'] = df_test['a'].apply(lambda x: x+1)
dt_range = pd.date_range(start='23-08-2017', periods = 4)
print(dt_range)

df_test['date_col'] = dt_range
print(df_test)


df_test.reset_index(drop = False, inplace= True)
print(df_test.index)
print(df_test)

df_test.reset_index(drop = True, inplace= True)
print(df_test.index)
print(df_test)


a = [1,2,3,4,5]
z = lagmat(a, 3)




### INDEX OPERATIONS END


"""

PANDAS SERIES AND DATAFRAME START

Ref URL:
https://discuss.analyticsvidhya.com/t/difference-between-map-apply-and-applymap-in-pandas/2365


"""


row_test = None
row_test = pd.Series([1,2,3,4,5,6,7,9,0]
                ,['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])

print(row_test['lag7'])


df_row_test = pd.DataFrame(columns = ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday'])
print(df_row_test)
df_row_test = df_row_test.append([row_test])


train_flattened['month'].replace('11','11 - November',inplace=True)
train_flattened['month'].replace('12','12 - December',inplace=True)

train_group = train_group.pivot('weekday','month','Visits')


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


'''
# Ref Analytics Vidya URL on handling pandas dataset:
# https://www.analyticsvidhya.com/blog/2016/01/python-tutorial-list-comprehension-examples/    
'''


    
#Lets load the dataset:
import pandas as pd
data = pd.read_csv("skills.csv")
print (data)    

#Split text with the separator ';'
data['skills_list'] = data['skills'].apply(lambda x: x.split(';'))
print (data['skills_list'])

print([sport for l in data['skills_list'] for sport in l])


#Initialize the set
skills_unq = set()
#Update each entry into set. Since it takes only unique value, duplicates will be ignored automatically.
skills_unq.update( (sport for l in data['skills_list'] for sport in l) )
print (skills_unq)


### PANDAS SERIES AND DATAFRAME END


"""

GROUP BY START

"""


dict_test = {'a': [1,2,3,4,2], 'b' : [5,6,7,8,5], 'c' : [9,10,11,12,9]}
df_test=None
z = range(10, 15, 1)
df_test = pd.DataFrame(dict_test, index = z)
print(df_test)

df_test_grouped = df_test.groupby(['a', 'c'])['b'].aggregate('max').to_frame().reset_index()
print(df_test_grouped)
df_test_grouped.columns = ['A','B','C']

print(df_test_grouped.index)
print(df_test_grouped.columns.values)
print(df_test_grouped)
print(type(df_test_grouped))
r = range(10,15,1)
print([x for x in r])
df_test_grouped.index = r
print(df_test_grouped.index.values)
print(df_test.iloc[1:,1].as_matrix())
print(df_test.iloc[:len(df_test)-1,1].as_matrix())
print(df_test.iloc[1:,1].as_matrix() - df_test.iloc[:len(df_test)-1,1].as_matrix())
print(df_test.iloc[:,1].as_matrix())
print(type(df_test.iloc[1:,1].as_matrix()))


### GROUP BY END

"""

MATPLOTLIB BY START

"""

import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import pylab
import cv2 as cv2
import numpy as np

fig, axes = plt.subplots(4,4, figsize = (28,28))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
z = axes.flat

print(axes.shape)
img = cv2.imread("C:\\Users\\Public\\Pictures\\Sample Pictures\\Koala.jpg")
axes.flat[0].imshow(img)
axes.flat[1].imshow(img)
axes.flat[2].imshow(img)
axes.flat[3].imshow(img)
axes.flat[4].imshow(img)
axes.flat[5].imshow(img)
axes.flat[6].imshow(img)
axes.flat[7].imshow(img)
axes.flat[8].imshow(img)
axes.flat[9].imshow(img)
axes.flat[10].imshow(img)
axes.flat[11].imshow(img)
axes.flat[12].imshow(img)
axes.flat[13].imshow(img)
axes.flat[14].imshow(img)
axes.flat[15].imshow(img)



f = plt.figure(figsize = (12,6))
plt.imshow(img)
plt.show()



img = cv2.imread("C:\\Users\\Public\\Pictures\\Sample Pictures\\Koala.jpg")
plt.figure(figsize = (16,8))

plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(img)

plt.subplot(223)
plt.imshow(img)

plt.subplot(224)
plt.imshow(img)

plt.show()


x = np.arange(1,10,1)
y = np.arange(11,20,1)
z = range(1,10,1)
q = [1,2,4,6,3,5,9,10,3]

print(x,y)

z = [[1,2],[3,4],[5,6],[7,8]]

plt.figure(figsize = (16,8))

plt.subplot(221)
plt.plot(x,y, color = 'g')

plt.subplot(222)
plt.scatter(x,y)

plt.subplot(223)
r = np.random.normal(size = 1000)
plt.hist(r, normed=True, bins=10)
plt.ylabel('Probability')

plt.subplot(224)
plt.bar(y, height = q)

plt.show()




f, ax = plt.subplots(2,2, figsize = (16,8))
ax.flat[0].plot(x,y, color = 'g')
ax.flat[1].scatter(x,y)

r = np.random.normal(size = 1000)
ax.flat[2].hist(r, normed=True, bins=10)
plt.ylabel('Probability')

ax.flat[3].bar(y, height = q)
plt.show()



import pandas as pd
import seaborn as sns

train = pd.read_csv("C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllMercedesBenzMfgData\\train.csv")

# plot the important features #

train.ix[:10,:10].columns
train.ix[:, 10:].columns
train['X10'].unique()
print(np.sort(train['X10'].unique()).tolist())

for col in train.ix[:, 10:].columns:
    print(col, np.sort(train[col].unique()).tolist())

print(train.columns.values)    
plt.figure(figsize=(30,6))

ax = plt.subplot(1,3,1)
var_name = "X0"
col_order = np.sort(train[var_name].unique()).tolist()
ax = sns.stripplot(x=var_name, y='y', data=train, order=col_order)

ax = plt.subplot(1,3,2)
var_name = "X1"
col_order = np.sort(train[var_name].unique()).tolist()
ax = sns.boxplot(x=var_name, y='y', data=train, order=col_order)


ax = plt.subplot(1,3,3)
var_name = "X2"
col_order = np.sort(train[var_name].unique()).tolist()
ax = sns.violinplot(x=var_name, y='y', data=train, order=col_order)

plt.show()    




#Examples
#--------

#Initialize a 2x2 grid of facets using the tips dataset:


import seaborn as sns; sns.set(style="ticks", color_codes=True)
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time", row="smoker")

#Draw a univariate plot on each facet:


import matplotlib.pyplot as plt
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

#(Note that it's not necessary to re-catch the returned variable; it's
#the same object, but doing so in the examples makes dealing with the
#doctests somewhat less annoying).

#Pass additional keyword arguments to the mapped function:


import numpy as np
bins = np.arange(0, 65, 5)
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=bins, color="r")

print(tips.columns.values)
print(tips.groupby(['sex','smoker'])['total_bill'].aggregate('sum').reset_index())
print(tips.groupby(['time','smoker','day'])['total_bill'].aggregate('sum').reset_index().sort('day', ascending = False))


#Plot a bivariate function on each facet:


g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

#Assign one of the variables to the color of the plot elements:


g = sns.FacetGrid(tips, col="time",  hue="smoker")
g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
      .add_legend())

#Change the size and aspect ratio of each facet:


g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
g = g.map(sns.boxplot, "time", "total_bill")

#Specify the order for plot elements:


g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
g = g.map(plt.hist, "total_bill", bins=bins, color="m")

#Use a different color palette:


kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
                  hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#Use a dictionary mapping hue levels to colors:


pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
                  hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#Additionally use a different marker for the hue levels:


g = sns.FacetGrid(tips, col="sex", hue="time", palette=pal,
                  hue_order=["Dinner", "Lunch"],
                  hue_kws=dict(marker=["^", "v"]))
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#"Wrap" a column variable with many levels into the rows:


attend = sns.load_dataset("attention")
g = sns.FacetGrid(attend, col="subject", col_wrap=5,
                  size=1.5, ylim=(0, 10))
g = g.map(sns.pointplot, "solutions", "score", scale=.7)

#Define a custom bivariate function to map onto the grid:


from scipy import stats
def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)
g = sns.FacetGrid(tips, col="smoker", hue="sex")
g = (g.map(qqplot, "total_bill", "tip", **kws)
      .add_legend())

#Define a custom function that uses a ``DataFrame`` object and accepts
#column names as positional variables:


import pandas as pd
df = pd.DataFrame(
    data=np.random.randn(90, 4),
    columns=pd.Series(list("ABCD"), name="walk"),
    index=pd.date_range("2015-01-01", "2015-03-31",
                        name="date"))
df = df.cumsum(axis=0).stack().reset_index(name="val")
def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
g = sns.FacetGrid(df, col="walk", col_wrap=2, size=3.5)
g = g.map_dataframe(dateplot, "date", "val")

#Use different axes labels after plotting:


g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="g", **kws)
      .set_axis_labels("Total bill (US Dollars)", "Tip"))

#Set other attributes that are shared across the facetes:


g = sns.FacetGrid(tips, col="smoker", row="sex")
g = (g.map(plt.scatter, "total_bill", "tip", color="r", **kws)
      .set(xlim=(0, 60), ylim=(0, 12),
           xticks=[10, 30, 50], yticks=[2, 6, 10]))

#Use a different template for the facet titles:


g = sns.FacetGrid(tips, col="size", col_wrap=3)
g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c")
      .set_titles("{col_name} diners"))

#Tighten the facets:


g = sns.FacetGrid(tips, col="smoker", row="sex",
                  margin_titles=True)
g = (g.map(plt.scatter, "total_bill", "tip", color="m", **kws)
      .set(xlim=(0, 60), ylim=(0, 12),
           xticks=[10, 30, 50], yticks=[2, 6, 10])
      .fig.subplots_adjust(wspace=.05, hspace=.05))




      
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()      

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
plt.figure(); df.plot();

df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
print(df3[:10])
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y='B')

plt.figure();
df.iloc[5].plot(kind='bar'); plt.axhline(0, color='k')


df = pd.DataFrame()
df.plot#.<TAB>
#df.plot.area     df.plot.barh     df.plot.density  df.plot.hist     df.plot.line     df.plot.scatter
#df.plot.bar      df.plot.box      df.plot.hexbin   df.plot.kde      df.plot.pie

plt.figure();
df.iloc[5].plot.bar(); plt.axhline(0, color='k')


df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot.bar();

df2.plot.bar(stacked=True);
df2.plot.barh(stacked=True);


df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

plt.figure();
df4.plot.hist(alpha=0.5)

plt.figure();
df4.plot.hist(stacked=True, bins=20)

#For further graphs refer: https://pandas.pydata.org/pandas-docs/stable/visualization.html
df_temp = pd.DataFrame(np.random.rand(10,3), columns = list('abc'))
print(df_temp)

np.random.seed(12345)
z = list(np.squeeze((np.random.rand(9,1)*10).astype(int)))
print(z)
dict_temp = {'a':z, 'b':list(np.arange(1,10,1)), 'c':list(np.arange(1,10,1))}
print(dict_temp)
df_temp = pd.DataFrame(dict_temp, columns = list('abc')).reset_index()
df_temp.index = df_temp['b']
print(df_temp)

z = list(df_temp.a) + list(df_temp.b)
print(z, type(z))

z = df_temp.a + df_temp.b
print(z, type(z))

z = np.array(list(df_temp.a) + list(df_temp.b))
print(z, type(z))

print(np.percentile(z, 0.5))

df = df_temp.loc[(df_temp['a'] > df_temp['b']), ['c', 'b']]
print(df)

df1 = df_temp.loc[5:, ['c', 'a']]
print(df1)

df2 = df_temp[(df_temp['a'] > df_temp['b'])]
print(df2)

df_temp['new col'] = df_temp['a'].apply(lambda x: x**2)
print(df_temp)




### MATPLOTLIB END

"""

TENSORFLOW START

"""


import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)


result = tf.multiply(x1,x2)
print(result)

sess = tf.Session()
print(sess.run(result))

sess.close()

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)    

### TENSORFLOW END




# Generate a sound
import numpy as np
framerate = 44100
t = np.linspace(0,5,framerate*5)
data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
Audio(data,rate=framerate)

# Can also do stereo or more channels
dataleft = np.sin(2*np.pi*220*t)
dataright = np.sin(2*np.pi*224*t)
Audio([dataleft, dataright],rate=framerate)

Audio("http://www.nch.com.au/acm/8k16bitpcm.wav")  # From URL
Audio(url="http://www.w3schools.com/html/horse.ogg")

Audio('/path/to/sound.wav')  # From file
Audio(filename='/path/to/sound.ogg')

Audio(b'RAW_WAV_DATA..)  # From bytes
Audio(data=b'RAW_WAV_DATA..)





from __future__ import print_function

import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
%matplotlib inline


# Load the example track
y, sr = librosa.load(librosa.util.example_audio_file())

import sounddevice as sd
sd.play(y, sr)


# Play it back!
IPython.display.Audio(data=y, rate=sr)


# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)
sd.play(y_h, sr)


# Play the harmonic component
IPython.display.Audio(data=y_h, rate=sr)


# Play the percussive component
IPython.display.Audio(data=y_p, rate=sr)
sd.play(y_p, sr)



# Pitch shifting?  Let's gear-shift by a major third (4 semitones)
y_shift = librosa.effects.pitch_shift(y, sr, 7)
sd.play(y_shift, sr)



IPython.display.Audio(data=y_shift, rate=sr)

# Or time-stretching?  Let's slow it down
y_slow = librosa.effects.time_stretch(y, 0.5)
sd.play(y_slow, sr)



IPython.display.Audio(data=y_slow, rate=sr)

# How about something more advanced?  Let's decompose a spectrogram with NMF, and then resynthesize an individual component
D = librosa.stft(y)

# Separate the magnitude and phase
S, phase = librosa.magphase(D)

# Decompose by nmf
components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)



