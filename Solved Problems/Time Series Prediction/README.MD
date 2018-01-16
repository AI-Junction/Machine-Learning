Generic Time-series Prediction using ARIMA Model

In real life we come across a number of scenarios where a time series based data needs to be projected and predicted for future time periods that could be vary from days to hours to minutes. For example, the equity analysts want to predict the stock prices based on the historical prices of a given scrip. Similarly, the property dealers want to know how the property prices are likely to vary purely based on the past prices trends etc. And there are many such examples.

Machine learning provides a couple of approaches towards predicting future values of a given time series. We implemented two machine learning algorithms for making such predictions. 

The two algorithms used in our program module are ARIMA (Auto Regression Integrated Moving Average) and Ensemble model based on Decision Trees called Adaboost. The results we found were hugely in favor of the ARIMA model that provided us quite realistic prediction of the future values as compared to Adaboost model. Even the prediction of the time-series values for the same time period as the training dataset also indicated that ARIMA performed much better than Adaboost.

We have tried to make the machine learning module generic in nature such that the user can input any time series data that meets certain specified input criteria. The implemented logic identifies the best hyper parameters that yield the best outcomes from the ARIMA model. The ARIMA model uses multiple hyper parameters of which ‘p’ (number of lags used in Auto Regression model), ’d’ (number of times differencing is done for stationarity) and ’q’ (the number of lagged errors used in Moving Average model) are the main parameters. The implemented logic evaluates the ARIMA results using AIC (Akaike information criteria) by iterating the model with various combinations of the ‘p’, ‘d’ and ‘q’ values. The combination giving the lowest AIC value is chosen for making the ARIMA based future predictions.

The outputs from our implementation of the generic timeseries prediction are illustrated below.

The below graph show the “SMS In” values coming to a cell (tower) on an hourly basis of a telecom operator. 

 ![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)


The Rolling Mean and Rolling Standard Deviation for a window of 7 units is shown below

 


To check the stationarity of the given time series, we perform the Dickey Fuller test. If the ‘Test Statistic’ value is found to be lower than the ‘Critical Values’ then this indicates stationarity and no specific treatment is needed to make the data stationary for feeding it into the ARIMA model. In case, the data is non stationary then this implementation uses the ‘differencing’ method in order to make the data stationary. In common terms the data is stationary if no specific ‘trend’ is observed in the mean or standard deviation values.

Dickey-Fuller test

Test Statistic                -6.036131e+00
p-value                        1.380683e-07
#Lags Used                     2.000000e+01
Number of Observations Used    7.000000e+02
Critical Value (1%)           -3.439726e+00
Critical Value (5%)           -2.865678e+00
Critical Value (10%)          -2.568973e+00
dtype: float64


As a next step we perform the decomposition of the given time series into ‘Trend’, ‘Seasonality’ and ‘Residual’ graphs.

 

Now we train the ARIMA model with some predefined values for ‘p’, ‘d’, ‘q’. We try to get an early view of the ARIMA effectiveness by running the prediction on the same training dataset. The results are seen to be very close to the actual values. In the graph below, the blue color represents the actual values while the red color represents the predicted values
 

To further obtain an indication of the number of lagged values that the ARIMA model should use, we generate the Auto Correlation Graph (ACF). The range where ACF values are positive are the potential candidates for being considered for the ‘p’ value. This ‘p’ value is the number of lags considered in the Auto Regression model of ARIMA.

 
As next step, we generate a range of p, d q values and work out all permutations of p, d, q in this range. We identify that particular combination of p, d, q values which generates the Best Fitting ARIMA model. The best combination of p, d, q is selected by finding the ARIMA model that generates the least AIC value. AIC stands for Akaike Information Criteria. 
Having found the best combination of p, d, q values, we now generate forecast for next 30 steps (hours) ahead in future
 



Business scenario:

Dynamic Traffic Prediction and planning mobile networks is challenging due to the high complexity of the telecom networks. Upcoming 5G networks will be characterized by a high densification of nodes and heterogeneity of layers, applications, and Radio Access Technologies (RAT). In this context, a network planning tool capable of dealing with this complexity is highly convenient.
The objective of this business scenario is to use the traffic information available in the form of Call Data Record (CDR) from the network infrastructure for training the machine learning algorithm. Post completion of the training, the machine learning program is able to predict future network traffic based on the measurement history of the network traffic. 
Such prediction capability is of vital importance in 4G and 5G networks for optimization of the telecom resources. As an example, the predicted traffic patterns of a given geography can be fed to the SDN controller to regulate the network capacity across the given geographical area for various times of the day. This enables the telecom service provider to optimize the network capacity allocation thereby saving power and delivering capacity where the need is. Thus the subscribers of the service provider enjoy a higher quality of service even while the operator saves on the power costs.
The machine learning module was trained using the Open dataset provided by Telecom Italia. The data-set refers to the traffic of voice/sms/data for each squared cell of the Milan area, measured during November and December 2013. It provides information about telecommunication activities in the city of Milano. The entire city if Milano is divided into smaller geographical units identified by square ids. These square ids are mapped to the cell ids falling in the given square. As output, the machine learning module provided the predicted traffic pattern for subsequent days. 
Utility of the solution
•	To use the traffic information available in Call Data Records (CDR) to deploy, configure and optimize network infrastructure.
•	To predict the location based (squared cell) daily/weekly/ monthly network traffic (categorical) based on the measurement history of the network and current trend.
•	Implementation of Machine Learning based solution to the mobile and metro network, in order to optimize and ensure energy efficient resources allocation. 
