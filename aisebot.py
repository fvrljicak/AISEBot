import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import os
import quandl
from datetime import datetime

dates = []
prices = []
datens = []
datenews = []
news = []
trends = []
sentiment	 = []
ns1 = []
ns2 = [[int],ns1]

def load_quandl_prices(dataset, start, transform='rdiff'):
    cache_file = 'prices-cache.csv'
    if os.path.exists(cache_file):
        print ('Loading Prices from cache')
        return pd.read_csv(cache_file, index_col=[0, 1], parse_dates=True)
    else:
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            print ('Downloading Prices for', row['Ticker'])
            all_price_data = quandl.get(
                row['Code'], trim_start=start, transformation=transform, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']

            close_price_data['CloseClass'] = 'Neutral'
            close_price_data.loc[
                close_price_data['Close'] > 0, 'CloseClass'] = 'Gain'
            close_price_data.loc[
                close_price_data['Close'] < 0, 'CloseClass'] = 'Loss'

            prices = prices.append(close_price_data)
        prices.index = pd.MultiIndex.from_arrays(
            [prices.index, prices['Ticker']])
        prices.drop('Ticker', axis=1, inplace=True)
        prices.to_csv(cache_file)
        return prices

def load_quandl_absolute_prices(dataset, start):
    cache_file = 'absolute-prices-cache.csv'
    if os.path.exists(cache_file):
        print ('Loading prices from cache')
        return pd.read_csv(cache_file, parse_dates=True)
    else:
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            print ('Downloading prices for', row['Ticker'])
            all_price_data = quandl.get(
                row['Code'], trim_start=start, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']
            prices = prices.append(close_price_data)
        prices.to_csv(cache_file)
        return prices

def load_quotes_absolute_prices(dataset, start):
	for index, row in dataset.iterrows():
		stock_cache_file = 'quotes/'+row['Ticker']+'-absolute-prices-cache.csv'
		if os.path.exists(stock_cache_file):
			print ('Loading quotes from cache')
			return pd.read_csv(stock_cache_file, parse_dates=True)
		if not(os.path.exists(stock_cache_file)):
			prices = pd.DataFrame()
			quandl_auth = 'T2GAyK64nwsePiJWMq8y'
			print ('Downloading prices for', row['Ticker'])
			all_price_data = quandl.get(row['Code'], trim_start=start, authtoken=quandl_auth)
			close_price_data = all_price_data[['Close']]
			close_price_data['Ticker'] = row['Ticker']
			prices = prices.append(close_price_data)
			prices.to_csv(stock_cache_file)
	return

def load_quotes_prices(dataset, start):
    for index, row in dataset.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-prices-cache.csv'
        if os.path.exists(stock_cache_file):
            print ('Loading quotes from cache')
            return pd.read_csv(stock_cache_file, parse_dates=True)
        if not(os.path.exists(stock_cache_file)):
            prices = pd.DataFrame()
            quandl_auth = 'T2GAyK64nwsePiJWMq8y'
            print ('Downloading prices for', row['Ticker'])
            all_price_data = quandl.get(row['Code'], trim_start=start, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']
            prices = prices.append(close_price_data)
            prices.to_csv(stock_cache_file)
    return

def load_quandl_newsentiment(dataset):
    cache_file = 'NS1/ns1-cache.csv'
    quandl_auth = 'T2GAyK64nwsePiJWMq8y'
    #ns1 = pd.DataFrame()
    i=1
    for index, row in dataset.iterrows():
    	#ns2[i] = ns1
    	ns1 = []
    	ns1 = pd.DataFrame()
    	stock_cache_file = row['NSCode']+'-cache.csv'
    	if not(os.path.exists(stock_cache_file)):
    		print(row['NSCode'])
    		print ('Downloading news for', row['NSCode'])
    		allnews_data = quandl.get(row['NSCode'], authtoken=quandl_auth)
    		ns1 = ns1.append(allnews_data)
    		ns1.to_csv(stock_cache_file)
    	if os.path.exists(stock_cache_file):
    		with open(stock_cache_file, 'r') as csvfile:
    			csvFileReader = csv.reader(csvfile)
    			next(csvFileReader)
    			print ('Loading news from cache ', row['NSCode'])
    			for rows in csvFileReader:
    				datens.append(int(time_to_num(rows[0])))
    				sentiment.append(rows[1])
    			#print (datens, ' ',sentiment,"\n")
		#ns1.to_csv(cache_file)
    return



def time_to_num(time_str):
    yyyy, mm , dd = map(int, time_str.split('-'))
    return yyyy*10000 + mm*100 + dd

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(time_to_num(row[0])))
            prices.append(float(row[1]))
    return

def get_feature_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            datenews.append(int(time_to_num(row[0])))
            news.append(float(row[1]))
    return

def get_price_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(time_to_num(row[0])))
            prices.append(float(row[1]))
    return


def gen_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	
		i=1
		for row in csvFileReader:
			if i<len(prices):
				if float(row[1])>float(prices[i]):
					trends.append(True)
				else:
					trends.append(False)
				i=i+1
			else:
				trends.append(False)
	return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	#plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	#plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	#plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	#plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	#plt.xlabel('Date')
	#plt.ylabel('Price')
	#plt.title('Support Vector Regression')
	#plt.legend()
	#plt.show()
	print (svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0])

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_trend(datenews, news, x):
	datenews = np.reshape(datenews,(len(datenews), 1)) # converting to matrix of n X 1
	print('classifing')
	svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	#boole = [-1,1]
	svc_rbf.fit(datenews,news) 
	print(svc_rbf.predict(x))
	return svc_rbf.predict(x)[0]

DJIA = pd.read_csv('DowJonesIA.csv')
djiaprices = load_quandl_prices(DJIA, start=datetime(2013, 1, 1))
djiabsprices = load_quandl_absolute_prices(DJIA, start=datetime(2013, 1, 1))
djiabsquotes = load_quotes_absolute_prices(DJIA, start=datetime(2013, 1, 1))
djianewsentiment = load_quandl_newsentiment(DJIA)

#########CLASIFICATION
for index, row in DJIA.iterrows():
    stock_cache_file = row['NSCode']+'-cache.csv'
    if os.path.exists(stock_cache_file):
        print ('Loading features from cache ', row['NSCode'])
        get_feature_data(stock_cache_file) # calling gen_data method to add features
        print ("\nThe stock open trend for NBD is:")
        predicted_trend = predict_trend(datenews, news, len(datenews)+1)  
        print ("RBF kernel: $", str(predicted_trend))

get_feature_data('NS1/CSCO_US-cache.csv') # calling gen_data method to add features
i=0
print ("Dates    Sentiment")
while i < len(datenews):
    print (datenews[i],"    ", news[i])
    i=i+1
print ("\nThe stock open trend for NBD is:")
predicted_trend = predict_trend(datenews, news, len(datenews)+1)  
print ("RBF kernel: $", str(predicted_trend))


###########REGRESSION
get_price_data('quotes/CSCO-absolute-prices-cache.csv') # calling get_data method by passing the csv file to it
print ("Dates    Prices ")
i=0
while i < len(dates):
    print (dates[i],"    ", prices[i])
    i=i+1
print ("\nThe stock open price for NBD is:")
predicted_price = predict_price(dates, prices, 28)  
print ("RBF kernel: $", str(predicted_price[0]))
print ("Linear kernel: $", str(predicted_price[1]))
print ("Polynomial kernel: $", str(predicted_price[2]))

	 