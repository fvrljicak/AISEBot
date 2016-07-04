import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import os
import quandl
import math
from datetime import datetime

pd.set_option('display.max_rows', None)
dates = []
prices = []
datens = []
datenews = []
news = []
trends = []
sentiment	 = []
ns1 = []
ns2 = [[int],ns1]

def load_quandl_absolute_prices(dataset, start):
    cache_file = 'absolute-prices-cache.csv'
    if not(os.path.exists(cache_file)):
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            print ('Downloading total absolute prices for', row['Ticker'])
            all_price_data = quandl.get(
                row['Code'], trim_start=start, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']
            prices = prices.append(close_price_data)
        prices.to_csv(cache_file)
    if os.path.exists(cache_file):
        print ('Loading total absolute prices from cache')
        return pd.read_csv(cache_file, index_col=[0], parse_dates=True)

def load_quandl_prices(dataset, start, transform='rdiff'):
    cache_file = 'prices-cache.csv'
    if not(os.path.exists(cache_file)):
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            print ('Downloading total relative prices for', row['Ticker'])
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
    if os.path.exists(cache_file):
        print ('Loading total relative prices from cache')
        return pd.read_csv(cache_file, index_col=[0, 1], parse_dates=True)

def load_quotes_absolute_prices(dataset, start):
    for index, row in dataset.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-absolute-prices-cache.csv'
        if not(os.path.exists(stock_cache_file)):
            prices = pd.DataFrame()
            quandl_auth = 'T2GAyK64nwsePiJWMq8y'
            print ('Downloading absolute prices for', row['Ticker'])
            all_price_data = quandl.get(row['Code'], trim_start=start, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']
            prices = prices.append(close_price_data)
            prices.to_csv(stock_cache_file)
        if os.path.exists(stock_cache_file):
            print ('Loading absolute quotes from cache', row['Ticker'])
            pd.read_csv(stock_cache_file, index_col=[0], parse_dates=True)

def load_quotes_prices(dataset, start,transform='rdiff'):
    for index, row in dataset.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-prices-cache.csv'
        if not(os.path.exists(stock_cache_file)):
            prices = pd.DataFrame()
            quandl_auth = 'T2GAyK64nwsePiJWMq8y'
            print ('Downloading relative prices for', row['Ticker'])
            all_price_data = quandl.get(row['Code'], trim_start=start, transformation=transform, authtoken=quandl_auth)
            close_price_data = all_price_data[['Close']]
            close_price_data['Ticker'] = row['Ticker']
            close_price_data['CloseClass'] = 0
            close_price_data.loc[close_price_data['Close'] > 0, 'CloseClass'] = 1
            close_price_data.loc[close_price_data['Close'] < 0, 'CloseClass'] = -1
            prices = prices.append(close_price_data)
            prices.to_csv(stock_cache_file)
        if os.path.exists(stock_cache_file):
            print ('Loading relative quotes from cache', row['Ticker'])
            pd.read_csv(stock_cache_file, index_col=[0], parse_dates=True)

def load_quandl_newsentiment(dataset):
    cache_file = 'NS1/ns1-cache.csv'
    quandl_auth = 'T2GAyK64nwsePiJWMq8y'
    i=1
    for index, row in dataset.iterrows():
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
            absdates.append(int(time_to_num(row[0])))
            absprices.append(float(row[1]))
    return

def get_relprice_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            reldates.append(int(time_to_num(row[0])))
            relprices.append(float(row[1]))
            trends.append(str(row[3]))
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
	#svr_lin = SVR(kernel= 'linear', C= 1e3)
	#svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	#svr_lin.fit(dates, prices)
	#svr_poly.fit(dates, prices)
	return svr_rbf.predict(x)[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_trend(datenews, news, x):
	datenews = np.reshape(datenews,(len(datenews), 1)) # converting to matrix of n X 1
	print('classifing') 
	svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	svc_rbf.fit(datenews[:-1],news[:-1]) 
	print(svc_rbf.predict(x))
	return svc_rbf.predict(x)[0]

def predict_trend_ns(news_prices, listtrends,ntrain,ntest):
    print('classifing') 
    svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_rbf.fit(news_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_rbf.predict(news_prices[-ntest:])[0]
def predict_price_ns(news_prices, listprices, ntrain, ntest):
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_rbf.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
    return svr_rbf.predict(news_prices[-ntest:])[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

DJIA = pd.read_csv('DowJonesIA.csv')
djiaprices = load_quandl_prices(DJIA, start=datetime(2013, 1, 1))
djiabsprices = load_quandl_absolute_prices(DJIA, start=datetime(2013, 1, 1))
djiabsquotes = load_quotes_absolute_prices(DJIA, start=datetime(2013, 1, 1))
djirelquotes = load_quotes_prices(DJIA, start=datetime(2013, 1, 1))
djianewsentiment = load_quandl_newsentiment(DJIA)

#for index, row in DJIA.iterrows():
#    stock_cache_file = row['NSCode']+'-cache.csv'
#    datenews=[]
#    news=[]
#    if os.path.exists(stock_cache_file):
#        print ('Loading features from cache ', row['NSCode'])
#        get_feature_data(stock_cache_file) 
#        print ("\nThe stock sentiment trend for NBD is:")
#        predicted_trend = predict_trend(datenews, news, len(datenews)+1)  
#        print ("RBF kernel:", str(predicted_trend))

print ("CISCO SVC test")
news=[]
datenews=[]
get_feature_data('NS1/CSCO_US-cache.csv') 
i=0
ns3=[]
ns3=pd.DataFrame(news, index=[datenews])

absprices=[]
absdates=[]
dfabsprices=[]

get_price_data('quotes/CSCO-absolute-prices-cache.csv') # calling get_data method by passing the csv file to it
dfabsprices=pd.DataFrame(absprices, index=[absdates])
left=dfabsprices
#leftstring=left.to_string()
#print(leftstring)
relprices=[]
reldates=[]
trends=[]
dfrelprices=[]
get_relprice_data('quotes/CSCO-prices-cache.csv') # calling get_data method by passing the csv file to it
d={'RelPrices': relprices, 'Trend': trends}
dfrelprices=pd.DataFrame(data=d, index=[reldates])
right=dfrelprices
#rightstring=right.to_string()
#print(rightstring)

#dfprices=[]
left=dfabsprices
right=dfrelprices
pricesmerge = pd.merge(left, right, left_index=True, right_index=True, how='outer')
dfprices = pd.DataFrame()
dfprices = pricesmerge
dfprices = dfprices.fillna(method='pad', axis=0).dropna()


#################### MERGE
ns3string=ns3.to_string()
#print(ns3string)
dfpricesstring=dfprices.to_string()
#print(dfpricesstring)
left=ns3
right=dfprices
news_prices = pd.merge(left, right, left_index=True, right_index=True, how='outer')
dfnews_prices = pd.DataFrame()
dfnews_prices = news_prices
dfnews_prices.columns=['NS','AbsPrice','RelPrice','Trend']
dfnews_prices = dfnews_prices.fillna(method='pad', axis=0).dropna()
news_prices_string=dfnews_prices.to_string()
print(news_prices_string)
listnews=dfnews_prices["NS"].tolist()
listprices=dfnews_prices["AbsPrice"].tolist()
listrelprices=dfnews_prices["RelPrice"].tolist()
listtrends=dfnews_prices["Trend"].tolist()

ntrain=1
ntest=1
print ("\nThe stock sentiment trend and prices for CSCO NBD is:")
testrange=30
i=testrange+1
positive_result=0
accuracygap=0
while i > 1:    
    #########CLASIFICATION
    predicted_trend_ns = predict_trend_ns(dfnews_prices, listtrends, i, i-1)  
    print ("RBF kernel trend for ", dfnews_prices.tail(i-1).index[0], ": ", str(predicted_trend_ns))
    print ("Comparable Real trend", dfnews_prices.tail(i-1).index[0], ": ", dfnews_prices.tail(i-1).values[0][3])
    if predicted_trend_ns==dfnews_prices.tail(i-1).values[0][3]:
        positive_result=positive_result+1
    ###########REGRESSION
    predicted_price_ns = predict_price_ns(dfnews_prices, listprices, i, i-1)  
    print ("RBF kernel price for ", dfnews_prices.tail(i-1).index[0], ": ", str(predicted_price_ns))
    print ("Comparable Real price", dfnews_prices.tail(i-1).index[0], ": ", dfnews_prices.tail(i-1).values[0][1])
    accuracygap=math.fabs(predicted_price_ns-dfnews_prices.tail(i-1).values[0][1])+accuracygap
    i=i-1
testresult=positive_result/testrange
print ("Positive trend prediction rate: ", testresult*100,"%")    
print ("Price prediction AVRG accuracy: ", accuracygap/testrange)    
