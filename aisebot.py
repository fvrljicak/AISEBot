import csv
import numpy as np
from numpy import ndarray
from sklearn.svm import SVR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import math
from datetime import datetime
from functools import wraps
import errno
import os
import signal
from threading import Thread
import time

class TimeoutError(Exception):
    pass
def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

from multiprocessing import Process
from time import sleep

def f(time):
    sleep(time)

predicted_price=float()
predicted_price_lin=float()


pd.set_option('display.max_rows', None)
dates = []
prices = []
datens = []
datenews = []
news = []
news_blank = []
trends = []
sentiment	 = []
ns1 = []
ns2 = [[int],ns1]

startyear=2013
startmonth=1
startday=1
start=startyear*10000+startmonth*100+startday#20130101
endyear=2016
endmonth=12
endday=1
end=endyear*10000+endmonth*100+endday#20161115
currentyear=2017
currentmonth=4
currentday=1
current=currentyear*10000+currentmonth*100+currentday
yeartradingdays=250
testrange=20 #prueba los ultimos 30 dias hasta el end
trainrange=40 #cpu bajo, buen nivel de prediccion
#trainrange=200 #cpu intensivo, falla en pocos minutos
#trainrange=1200 #dataset de 1200 aprox, no se lo banca
#trainrange=testrange*2 # el doble de la cantidad de pruebas para tiempo aceptable, mas datos mas cpu
daytradeinvestns = np.empty((testrange,), float)
daytradeinvest = np.empty((testrange,), float)
daytradegain = np.empty((testrange,), float)
daytradereturn = np.empty((testrange,), float)
buythreshold=0.02#minimo retorno diario esperado (2% = predictedgain over todayopenprice)
buysellcost=0.01#http://www.stockbrokers.com/review/optionshouse?utm_source=stocktrader.com&utm_medium=online-stock-brokers&utm_campaign=optionshouse&ref=st-osb
#Interactive Brokers $0.005 per share
tradeticker = []
tradegain = []
tradepredictedgain = []
tradereturn = []
tradepredictedreturn = []
tradedate = []
tradetrnrange = []
tradetstrange = []
tradebthreshold = []
tradeCparam = []
tradegammaparam = []

i=0
while i < testrange:
    daytradeinvest[i]=0.01 #para evitar error en calculo inicial de retorno
    daytradeinvestns[i]=0.01#para evitar error en calculo inicial de retorno    
    daytradegain[i]=0.01#para evitar error en calculo inicial de retorno    
    i=i+1
        
def load_quandl_absolute_prices(dataset, start, end):
    cache_file = 'absolute-prices-cache.csv'
    if not(os.path.exists(cache_file)):
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            print(start,end)
            print ('Downloading total absolute prices for', row['Ticker'])
            all_price_data = quandl.get(
                row['Code'], trim_start=start, trim_end=end, authtoken=quandl_auth)
            price_data = all_price_data[['Close']]
            open_price_data = all_price_data[['Open']]
            price_data['Ticker'] = row['Ticker']
            prices = prices.append(price_data)
        prices.to_csv(cache_file)
    if os.path.exists(cache_file):
        print ('Loading total absolute prices from cache')
        return pd.read_csv(cache_file, index_col=[0], parse_dates=True)

def load_quandl_prices(dataset, start, end, transform='rdiff'):
    cache_file = 'prices-cache.csv'
    if not(os.path.exists(cache_file)):
        prices = pd.DataFrame()
        quandl_auth = 'T2GAyK64nwsePiJWMq8y'
        for index, row in dataset.iterrows():
            all_price_data = quandl.get(
                row['Code'], trim_start=start, trim_end=end, transformation=transform, authtoken=quandl_auth)
            price_data = all_price_data[['Close']]
            open_price_data = all_price_data[['Open']]
            price_data['Ticker'] = ticker
            price_data['CloseClass'] = 'Neutral'
            price_data.loc[
                price_data['Close'] > 0, 'CloseClass'] = 'Gain'
            price_data.loc[
                price_data['Close'] < 0, 'CloseClass'] = 'Loss'
            prices = prices.append(price_data)
        prices.index = pd.MultiIndex.from_arrays(
            [prices.index, prices['Ticker']])
        prices.drop('Ticker', axis=1, inplace=True)
        prices.to_csv(cache_file)
    if os.path.exists(cache_file):
        print ('Loading total relative prices from cache')
        return pd.read_csv(cache_file, index_col=[0, 1], parse_dates=True)

def load_quotes_absolute_prices(dataset, start, end):
    for index, row in dataset.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-absolute-prices-cache.csv'
        if not(os.path.exists(stock_cache_file)):
            prices = pd.DataFrame()
            quandl_auth = 'T2GAyK64nwsePiJWMq8y'
            print(start,end)
            print ('Downloading absolute prices for', row['Ticker'])
            all_price_data = quandl.get(row['Code'], trim_start=start, trim_end=end, authtoken=quandl_auth)
            price_data = all_price_data[['Open']]
            price_data['Close'] = all_price_data[['Close']]
            price_data['Ticker'] = row['Ticker']
            prices = prices.append(price_data)
            prices.to_csv(stock_cache_file)
        if os.path.exists(stock_cache_file):
            print ('Loading absolute quotes from cache', row['Ticker'])
            pd.read_csv(stock_cache_file, index_col=[0], parse_dates=True)

def load_quotes_prices(dataset, start, end, transform='rdiff'):
    for index, row in dataset.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-prices-cache.csv'
        if not(os.path.exists(stock_cache_file)):
            prices = pd.DataFrame()
            quandl_auth = 'T2GAyK64nwsePiJWMq8y'
            print ('Downloading relative prices for', row['Ticker'])
            all_price_data = quandl.get(row['Code'], trim_start=start, trim_end=end, transformation=transform, authtoken=quandl_auth)
            price_data = all_price_data[['Open']]
            price_data['Close'] = all_price_data[['Close']]
            price_data['Ticker'] = row['Ticker']
            price_data['CloseClass'] = 0
            price_data.loc[price_data['Close'] > 0, 'CloseClass'] = 1
            price_data.loc[price_data['Close'] < 0, 'CloseClass'] = -1
            prices = prices.append(price_data)
            prices.to_csv(stock_cache_file)
        if os.path.exists(stock_cache_file):
            print ('Loading relative quotes from cache', row['Ticker'])
            pd.read_csv(stock_cache_file, index_col=[0], parse_dates=True)

def load_quandl_newsentiment(dataset, start, end):
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
                    date=int(time_to_num(rows[0]))
                    if date > start and date < end:
                #        print(start,date,end)
                        datens.append(date)
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

def get_feature_data(filename, start, end):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        news.append(0)
        for row in csvFileReader:
            date=int(time_to_num(row[0]))
            if date > start and date < end:
                datenews.append(date)
                news.append(float(row[1]))
        news.pop()
    return
def get_feature_data_blank(filename, start, end):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        news_blank.append(0)
        zero=0
        for row in csvFileReader:
            date=int(time_to_num(row[0]))
            if date > start and date < end:
                datenews_blank.append(date)
                news_blank.append(zero)
        news_blank.pop()
    return

def get_price_data(filename,start,end):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            date=int(time_to_num(row[0]))
            if date > start and date < end:
                absdates.append(int(time_to_num(row[0])))
                if isFloat(row[1]):
                    absopenprices.append(float(row[1]))
                else:
                    absopenprices.append(0.0)
                abscloseprices.append(float(row[2]))
    return

def get_relprice_data(filename,start,end):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            date=int(time_to_num(row[0]))
            if date > start and date < end:
                reldates.append(int(time_to_num(row[0])))
                if isFloat(row[1]):
                    relprices.append(float(row[1]))
                else:
                    relprices.append(0.0)
                trends.append(str(row[4]))
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

def predict_trend(news_blank_prices, listtrends,ntrain,ntest):
    #print('classifing') 
    svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_rbf.fit(news_blank_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_rbf.predict(news_blank_prices[-ntest:])[0]
def predict_trend_lin(news_blank_prices, listtrends,ntrain,ntest):
    #print('classifing') 
    svc_lin = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_lin.fit(news_blank_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_lin.predict(news_blank_prices[-ntest:])[0]

def predict_trend_ns(news_prices, listtrends,ntrain,ntest):
    #print('classifing') 
    svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_rbf.fit(news_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_rbf.predict(news_prices[-ntest:])[0]
def predict_trend_ns_lin(news_prices, listtrends,ntrain,ntest):
    #print('classifing') 
    svc_lin = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_lin.fit(news_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_lin.predict(news_prices[-ntest:])[0]
    svr_lin = SVR(kernel= 'linear', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_lin.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
    return svr_lin.predict(news_prices[-ntest:])[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]
def predict_trend_ns_poly(news_prices, listtrends,ntrain,ntest):
    #print('classifing') 
    svc_poly = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svc_poly.fit(news_prices[:-ntrain],listtrends[:-ntrain]) 
    return svc_poly.predict(news_prices[-ntest:])[0]
def predict_price_ns_poly(news_prices, listprices, ntrain, ntest):
    svr_poly = SVR(kernel= 'poly', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_poly.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
    return svr_poly.predict(news_prices[-ntest:])[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_price(news_prices, listprices, ntrain,ntest,c,g):
    global predicted_price
    svr_rbf = SVR(kernel= 'rbf', C=c, gamma=g) # defining the support vector regression models
#    print ("fitting ",ntrain)
    svr_rbf.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
#    print(news_prices[:-ntrain])
#    print(news_prices[-ntest:])
#    print ("predicting")
    predicted_price=svr_rbf.predict(news_prices[-ntest:])[0]
    #print(predicted_price)
    return predicted_price#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_price_lin(news_prices, listprices, ntrain,ntest,c):
    global predicted_price_lin 
    svr_lin = SVR(kernel= 'linear', C=c)
#    print ("fitting ",ntrain)
#    print(news_prices[:-ntrain])
#    print(news_prices[-ntest:])
    svr_lin.fit(news_prices[:-ntrain], listprices[:-ntrain])
#    print ("predicting")
    predicted_price_lin = svr_lin.predict(news_prices[-ntest:])[0]
    #print(predicted_price_lin)
    return predicted_price_lin#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_price_ns(news_prices, listprices, ntrain, ntest):
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
#    print ("fitting ",ntrain)
#    print(news_prices[:-ntrain])
#    print(news_prices[-ntest:])
    svr_rbf.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
#    print ("predicting")
    return svr_rbf.predict(news_prices[-ntest:])[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def predict_price_ns_lin(news_prices, listprices, ntrain, ntest):
    svr_lin = SVR(kernel= 'linear', C= 1e3, gamma= 0.1) # defining the support vector regression models
#    print ("fitting ",ntrain)
#    print(news_prices[:-ntrain])
#    print(listprices[:-ntrain])
    svr_lin.fit(news_prices[:-ntrain], listprices[:-ntrain]) # fitting the data points in the models
#    print ("predicting")
    pricepredict=svr_lin.predict(news_prices[-ntest:])
#    print(news_prices[-ntest:])
#    print(pricepredict[:])
    return pricepredict[0]#, svr_lin.predict(x)[0], svr_poly.predict(x)[0]
def run_with_limited_time(func, args, kwargs, time):
    global predicted_price
    global predicted_price_lin 
    """Runs a function with time limit
    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Thread(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    
    return True

def model_selector(tckr,dfprices,lcprices,trnrange,tstrange,bthreshold,c,g):
    maxgain=0
    accumulatedgain=0
    mthreshold=bthreshold*4
    bthreshold=bthreshold/4
    bestparams=[tckr,trnrange,tstrange,bthreshold,c,g,accumulatedgain]
    maxc=c*4
    definedc=c
    c=c/4
    maxg=g*4
    definedg=g
    g=g/4
    maxtstrange=tstrange*2
    definedtstrange=tstrange
    #tstrange=int(tstrange/2)
    maxtrnrange=trnrange*2
    definedtrnrange=trnrange
    trnrange=int(trnrange/2)
    while bthreshold <= mthreshold:
        i=tstrange
        while trnrange<=maxtrnrange:
            print("MODEL PARAMS Ticker: ",bestparams[0]," EXPECTED GAIN: ",bestparams[6]," TRAIN RANGE: ",trnrange," TEST RANGE: ",tstrange," BUY THRESHOLD: ",bthreshold," C: ",c," gamma: ",g)
            while c<=maxc:
                while g<=maxg:
                    while i > 1:  
                        ticker=tckr
                        todayopenprice=dfprices.tail(i-1).values[0][1]
                        todaycloseprice=lcprices[len(lcprices)-(i-1)]
                        realdaygain=todaycloseprice-todayopenprice
                        rowdate=dfprices.tail(i-1).index[0]
                        
                        global predicted_price
                        run_with_limited_time(predict_price, (dfprices[-(trnrange+i):], lcprices[-(trnrange+i):], i, i-1,c,g),{},2)  
                        global predicted_price_lin
                        run_with_limited_time(predict_price_lin, (dfprices[-(trnrange+i):], lcprices[-(trnrange+i):], i, i-1,c), {},2)    
                        
                        predictedgain=predicted_price-todayopenprice
                        predictedlingain=predicted_price_lin-todayopenprice
                        if predictedgain/todayopenprice>bthreshold and predictedlingain/todayopenprice>bthreshold: #ambas predicciones positivas
                            realdaygainsinns=realdaygain-buysellcost
                            accumulatedgain=accumulatedgain+realdaygainsinns
                        i=i-1
                    if accumulatedgain>=maxgain:
                        maxgain=accumulatedgain
                        bestparams[0]=tckr
                        bestparams[1]=trnrange
                        bestparams[2]=tstrange
                        bestparams[3]=bthreshold
                        bestparams[4]=c
                        bestparams[5]=g
                        bestparams[6]=accumulatedgain
                    accumulatedgain=0
                    g=g+maxg/8
                g=definedg
                c=c+maxc/8
            c=definedc
            trnrange=trnrange+int(maxtrnrange/4)
        trnrange=definedtrnrange
        bthreshold=bthreshold+mthreshold/8
    return bestparams

DJIA = pd.read_csv('DowJonesIA.csv')
djiaprices = load_quandl_prices(DJIA, start=datetime(startyear, startmonth, startday), end=datetime(endyear, endmonth, endday))
djiabsprices = load_quandl_absolute_prices(DJIA, start=datetime(startyear, startmonth, startday), end=datetime(endyear, endmonth, endday))
djiabsquotes = load_quotes_absolute_prices(DJIA, start=datetime(startyear, startmonth, startday), end=datetime(endyear, endmonth, endday))
djirelquotes = load_quotes_prices(DJIA, start=datetime(startyear, startmonth, startday), end=datetime(endyear, endmonth, endday))
djianewsentiment = load_quandl_newsentiment(DJIA,start, end)

while end < current:
    print('Timestamp: %s' % datetime.now())
    totaloperns=0
    totalinvestns=1
    totalgainns=0
    totaloper=0
    totaloperloss=0
    totaloperwin=0
    totaloperchance=0
    totalinvest=1
    totalgain=0
    for index, row in DJIA.iterrows():
        stock_cache_file = 'quotes/'+row['Ticker']+'-prices-cache.csv'
        stock_abscache_file = 'quotes/'+row['Ticker']+'-absolute-prices-cache.csv'
        news_cache_file = row['NSCode']+'-cache.csv'
        print(stock_cache_file)
        print(news_cache_file)
        datenews=[]
        datenews_blank=[]
        news=[]
        news_blank=[]
    
        print (row['NSCode'], " SVC test")
        get_feature_data(news_cache_file, start, end) 
        get_feature_data_blank(news_cache_file, start, end) 
        i=0
        ns3=[]
        ns3_blank=[]
        ns3=pd.DataFrame(news, index=[datenews])
        ns3_blank=pd.DataFrame(news_blank, index=[datenews_blank])
    
        absopenprices=[]
        abscloseprices=[]
        absdates=[]
        dfabsprices=[]
        dfabsopenprices=[]
        dfabscloseprices=[]
    
        get_price_data(stock_abscache_file,start,end) # calling get_data method by passing the csv file to it
        dfabsopenprices=pd.DataFrame(absopenprices, index=[absdates])
        dfabscloseprices=pd.DataFrame(abscloseprices, index=[absdates])
        left=dfabsopenprices
        right=dfabscloseprices
        abspricesmerge = pd.merge(left, right, left_index=True, right_index=True, how='outer')
        dfabsprices = pd.DataFrame()
        dfabsprices = abspricesmerge
    #leftstring=left.to_string()
    #print(leftstring)
        relprices=[]
        reldates=[]
        trends=[]
        dfrelprices=[]
        get_relprice_data(stock_cache_file,start,end) # calling get_data method by passing the csv file to it
        d={'RelPrices': relprices, 'Trend': trends}
        dfrelprices=pd.DataFrame(data=d, index=[reldates])
    #rightstring=right.to_string()
    #print(rightstring)
    
    #dfprices=[]
        left=dfabsprices
        right=dfrelprices
        pricesmerge = pd.merge(left, right, left_index=True, right_index=True, how='outer')
        dfprices = pd.DataFrame()
        dfprices = pricesmerge
        dfprices.columns=['AbsOpenPrice','AbsClosePrice','RelPrice','Trend']
        dfprices = dfprices.fillna(method='pad', axis=0).dropna()
    
    
    #################### MERGE
        ns3string=ns3.to_string()
    #    print(ns3string)
        dfpricesstring=dfprices.to_string()
    #print(dfpricesstring)
        left=ns3
        right=dfprices
        news_prices = pd.merge(left, right, left_index=True, right_index=True, how='outer')
        dfnews_prices = pd.DataFrame()
        dfnews_prices = news_prices
        dfnews_prices.columns=['NS','AbsOpenPrice','AbsClosePrice','RelPrice','Trend']
        dfnews_prices = dfnews_prices.fillna(method='pad', axis=0).dropna()
        listcloseprices=dfnews_prices["AbsClosePrice"].tolist()
        listtrends=dfnews_prices["Trend"].tolist()
    #    print(dfnews_prices)
        dfnews_prices.drop('AbsClosePrice', axis=1, inplace=True)
        dfnews_prices.drop('Trend', axis=1, inplace=True)
        dfnews_prices.drop('RelPrice', axis=1, inplace=True)
    #    news_prices_string=dfnews_prices.to_string()
        
        left=ns3_blank
        right=dfprices
        news_blank_prices = pd.merge(left, right, left_index=True, right_index=True, how='outer')
        dfnews_blank_prices = pd.DataFrame()
        dfnews_blank_prices = news_blank_prices
        dfnews_blank_prices.columns=['NS','AbsOpenPrice','AbsClosePrice','RelPrice','Trend']
        dfnews_blank_prices = dfnews_blank_prices.fillna(method='pad', axis=0).dropna()
        dfnews_blank_prices.drop('AbsClosePrice', axis=1, inplace=True)
        dfnews_blank_prices.drop('Trend', axis=1, inplace=True)
        dfnews_blank_prices.drop('RelPrice', axis=1, inplace=True)
    #    news_blank_prices_string=dfnews_blank_prices.to_string()
    #    print(news_blank_prices_string)
        
        
        listnews=dfnews_prices["NS"].tolist()
        listopenprices=dfnews_prices["AbsOpenPrice"].tolist()
        #lastdayprice=listcloseprices.pop()
        #firstdayprice=listcloseprices.pop(0)
        dfnews_prices_string=dfnews_prices.to_string()
        #print (dfnews_prices,listcloseprices)
        #listrelprices=dfnews_prices["RelPrice"].tolist()
        #lastdaytrend=listtrends.pop()
        #lastdaytrend=listtrends.append(0)
        
        ntrain=1
        ntest=1
        print ("\nThe stock sentiment trend and prices for " ,row['Ticker'], " NBD is:")
    
        i=testrange+1
        positive_result=0
        positive_result_lin=0
        positive_result_ns=0
        positive_result_ns_lin=0
        accuracygapns=0
        accuracygapns_lin=0
        accumulatedgainns=0
        accumulatedoperns=0
        accumulatedinvestns=1
        accuracygap=0
        accuracygap_lin=0
        accumulatedgain=0
        accumulatedoper=0
        accumulatedinvest=1
        definedbuythreshold=buythreshold
        maxthreshold=buythreshold*2
        minthreshold=buythreshold/2
        i=testrange+1
        c= 1e3
        g= 0.1
        ticker=row['Ticker']
        modelparams=model_selector(ticker, dfnews_blank_prices, listcloseprices, trainrange,testrange, buythreshold, c, g)
        trnrange=modelparams[1]
        tstrange=modelparams[2]
        bthreshold=modelparams[3]
        c=modelparams[4]
        g=modelparams[5]
        modelaccumulatedgain=modelparams[6]
        print("MODEL PARAMS Ticker: ",modelparams[0]," EXPECTED GAIN: ",modelaccumulatedgain," TRAIN RANGE: ",trnrange," TEST RANGE: ",tstrange," BUY THRESHOLD: ",bthreshold," C: ",c," gamma: ",g)
        i=tstrange+1
        
        #buythreshold=minthreshold
    #    while buythreshold < maxthreshold:
        while i > 1:  
            #########CLASIFICATION
    #        tomorrowtrend=listtrends[len(listtrends)-(i-1)]
    #        print ("Comparable Real trend",ticker," ", rowdate, ": ", str(tomorrowtrend))
    #        print ("SVC Trend RBF+NS...")
    #        predicted_trend_ns = predict_trend_ns(dfnews_prices, listtrends, i, i-1)  
    #        print ("RBF kernel trend for ",ticker," ", rowdate, ": ", str(predicted_trend_ns))
    #        print ("SVC Trend RBF-NS...")
    #        predicted_trend = predict_trend(dfnews_blank_prices, listtrends, i, i-1)  
    #        print ("RBF sin NS trend for ",ticker," ", rowdate, ": ", str(predicted_trend))
    #        print ("SVC Trend LIN+NS...")
    #        predicted_trend_ns_lin = predict_trend_ns_lin(dfnews_prices, listtrends, i, i-1)  
    #        print ("LIN kernel trend for ",ticker," ", rowdate, ": ", str(predicted_trend_ns_lin))
    #        print ("SVC Trend LIN-NS...")
    #        predicted_trend_lin = predict_trend_lin(dfnews_blank_prices, listtrends, i, i-1)  
    #        print ("LIN sin NS trend for ",ticker," ", rowdate, ": ", str(predicted_trend_lin))
    #        if predicted_trend_ns==tomorrowtrend:
    #            positive_result_ns=positive_result+1
    #        if predicted_trend_ns_lin==tomorrowtrend:
    #            positive_result_ns_lin=positive_result_lin+1
    #        if predicted_trend==tomorrowtrend:
    #            positive_result=positive_result+1
    #        if predicted_trend_lin==tomorrowtrend:
    #            positive_result_lin=positive_result_lin+1
              
        ###########REGRESSION
            yesterdayprice=listcloseprices[len(listcloseprices)-i]
            todayopenprice=dfnews_prices.tail(i-1).values[0][1]
            todaycloseprice=listcloseprices[len(listcloseprices)-(i-1)]
            daytodaygain=todaycloseprice-yesterdayprice
            realdaygain=todaycloseprice-todayopenprice
            rowdate=dfnews_prices.tail(i-1).index[0]
            predicted_price_ns=0
            predicted_price_ns_lin=0
    #       if daytodaygain>buythreshold:
            #print ("Yesterday  price for ",ticker," ", dfnews_prices.tail(i).index[0], ": ", yesterdayprice)
            print ("Today Open price for ",ticker," ", rowdate, ": ", todayopenprice)
            #print ("SVR price RBF+NS...")
            #args=(dfnews_prices[-trainrange:], listcloseprices[-trainrange:], i, i-1)
            #predicted_price_ns = timeout(predict_price_ns,timeout_duration=1,default=None,*args)  
            #predicted_price_ns = predict_price_ns(dfnews_blank_prices[-trainrange:], listcloseprices[-trainrange:], i, i-1)  
            #print ("RBF kernel price for ",ticker," ", rowdate, ": ", str(predicted_price_ns))
            #print ("SVR price RBF-NS...")
            global predicted_price
            run_with_limited_time(predict_price, (dfnews_blank_prices[-(trnrange+i):], listcloseprices[-(trnrange+i):], i, i-1,c,g),{},2)  
            print ("RBF sin NS price for ",ticker," ", rowdate, ": ", str(predicted_price))
            #print ("SVR price LIN+NS...")
            #predicted_price_ns_lin = predict_price_ns_lin(dfnews_prices[-trainrange:], listcloseprices[-trainrange:], i, i-1)  
            #print ("LIN kernel price for ",ticker," ", rowdate, ": ", str(predicted_price_ns_lin))
            #print ("SVR price LIN-NS...")
            global predicted_price_lin
            run_with_limited_time(predict_price_lin, (dfnews_blank_prices[-(trnrange+i):], listcloseprices[-(trnrange+i):], i, i-1,c), {},2)    
            print ("LIN sin NS price for ",ticker," ", rowdate, ": ", str(predicted_price_lin))
            print ("Closing price for    ",ticker," ", rowdate, ": ", todaycloseprice)
            print ("---")
            predictednsgain=predicted_price_ns-todayopenprice
            predictednslingain=predicted_price_ns_lin-todayopenprice
            predictedgain=predicted_price-todayopenprice
            predictedlingain=predicted_price_lin-todayopenprice
            if predictednsgain/todayopenprice>bthreshold and predictednslingain/todayopenprice>bthreshold: #ambas predicciones positivas
                if daytodaygain/todayopenprice < bthreshold:
                    print ("ALERTA@BADPREDICT")
                #daytodaygain=daytodaygain-buysellcost
                daytradeinvestns[i-2]=daytradeinvestns[i-2]+todayopenprice
                realdaygainns=realdaygain-buysellcost
                accumulatedgainns=accumulatedgainns+realdaygainns
                totaloperns=totaloperns+1
                accumulatedoperns=accumulatedoperns+1
                accumulatedinvestns=yesterdayprice
                if totalinvestns<accumulatedinvestns:
                    totalinvestns=accumulatedinvestns
                print ("Day to Day Gain  NS for ",ticker," ", rowdate, ": ", daytodaygain)
                print ("Real Day   Gain  NS for ",ticker," ", rowdate, ": ", realdaygainns)
                print ("Day to Day Gain  NS RBF ",ticker," ", rowdate, ": ", predictednsgain)
                print ("Day to Day sinNS NS RBF ",ticker," ", rowdate, ": ", predictedgain)
                print ("Day to Day Gain  NS LIN ",ticker," ", rowdate, ": ", predictednslingain)
                print ("Day to Day sinNS NS LIN ",ticker," ", rowdate, ": ", predictedlingain)
                accuracygapns=math.fabs(predicted_price_ns-todaycloseprice)+accuracygapns
                accuracygapns_lin=math.fabs(predicted_price_ns_lin-todaycloseprice)+accuracygapns_lin
                print ("Price predict accNS RBF ",ticker," ", rowdate, ": ", math.ceil((1-accuracygapns/(testrange-i+2))*10000)/100,"%")    
                print ("Price predict accNS LIN ",ticker," ", rowdate, ": ", math.ceil((1-accuracygapns_lin/(testrange-i+2))*10000)/100,"%")    
            if predictedgain/todayopenprice>bthreshold and predictedlingain/todayopenprice>bthreshold: #ambas predicciones positivas
                if daytodaygain/todayopenprice < bthreshold:
                    print ("ALERTA@BADPREDICTsinNS")
                #daytodaygain=daytodaygain-buysellcost
                daytradeinvest[i-2]=daytradeinvest[i-2]+todayopenprice
                realdaygainsinns=realdaygain-buysellcost
                daytradegain[i-2]=daytradegain[i-2]+realdaygainsinns
                if realdaygainsinns<0:
                        totaloperloss=totaloperloss+1
                if realdaygainsinns/todayopenprice>=buythreshold:
                        totaloperwin=totaloperwin+1
                accumulatedgain=accumulatedgain+realdaygainsinns
                tradeticker.append(ticker)
                tradegain.append(realdaygainsinns)
                tradepredictedgain.append(predictedgain)
                tradedate.append(rowdate)
                tradepredictedreturn.append(predictedgain/todayopenprice)
                tradereturn.append((todaycloseprice-todayopenprice)/todayopenprice)
                tradetrnrange.append(modelparams[1])
                tradetstrange.append(modelparams[2])
                tradebthreshold.append(modelparams[3])
                tradeCparam.append(modelparams[4])
                tradegammaparam.append(modelparams[5])
                totaloper=totaloper+1
                accumulatedoper=accumulatedoper+1
                accumulatedinvest=yesterdayprice
                if totalinvest<accumulatedinvest:
                    totalinvest=accumulatedinvest
    #            print ("Day to Day Gain  for ",ticker," ", rowdate, ": ", daytodaygain)
    #            print ("Real Day   Gain  for ",ticker," ", rowdate, ": ", realdaygainsinns)
    #            print ("Day to Day Gain  RBF ",ticker," ", rowdate, ": ", predictedgain)
    #            print ("Day to Day sinNS RBF ",ticker," ", rowdate, ": ", predicted_price-dfnews_prices.tail(i).values[0][1])
    #            print ("Day to Day Gain  LIN ",ticker," ", rowdate, ": ", predictedlingain)
    #            print ("Day to Day sinNS LIN ",ticker," ", rowdate, ": ", predicted_price_lin-dfnews_prices.tail(i).values[0][1])
                accuracygap=math.fabs(predicted_price-todaycloseprice)+accuracygap
                accuracygap_lin=math.fabs(predicted_price_lin-todaycloseprice)+accuracygap_lin
    #            print ("Price predict accRBF ",ticker," ", rowdate, ": ", math.ceil((1-accuracygap/(testrange-i+2))*10000)/100,"%")    
    #            print ("Price predict accLIN ",ticker," ", rowdate, ": ", math.ceil((1-accuracygap_lin/(testrange-i+2))*10000)/100,"%")    
            if (todaycloseprice-todayopenprice)/todayopenprice>=buythreshold:
                totaloperchance=totaloperchance+1 
            #testresult_ns=positive_result_ns/(testrange-i+3)
            #testresult_ns_lin=positive_result_ns_lin/(testrange-i+3)
            #testresult=positive_result/(testrange-i+3)
            #testresult_lin=positive_result_lin/(testrange-i+3)
            #print ("Trend predict accRBF ",ticker," ", rowdate, ": ", testresult_ns*100,"%")    
            #print ("Trend predict sinNS  ",ticker," ", rowdate, ": ", testresult*100,"%")    
            #print ("Trend predict accLIN ",ticker," ", rowdate, ": ", testresult_ns_lin*100,"%")    
            #print ("Trend predict sinNS  ",ticker," ", rowdate, ": ", testresult_lin*100,"%")    
            #if testresult!=testresult_ns or testresult_lin!=testresult_ns_lin:
            #    print("ALERTA@NS")
            i=i-1
    #        i=testrange+1
    #        buythreshold=buythreshold+minthreshold/2
        buythreshold=definedbuythreshold
        totalgainns=totalgainns+accumulatedgainns
        totalgain=totalgain+accumulatedgain
        print ("ACCUMULATED GAIN NS          ",ticker," ",accumulatedgainns)    
        if accumulatedoperns>0:
            print ("TOTAL NS CAPITAL used        ",ticker," ",accumulatedinvestns)
            print ("TOTAL NS OPER buy/sell       ",ticker," ",accumulatedoperns)
            print ("TOTAL OPERATIONS failed      ",totaloperloss)
            print ("TOTAL OPERATIONS wins        ",totaloperwin)
            print ("TOTAL NS GAIN obtained       ",ticker," ",accumulatedgainns)
            print ("EFFEC NS TIVE ROI 30day      ",ticker," ",math.ceil((accumulatedgainns/accumulatedinvestns)*100), "%")
            print ("EXPEC NS T ANNUAL RETURN     ",ticker," ",math.ceil((yeartradingdays*(accumulatedgainns/accumulatedinvestns)/testrange)*100), "%")
            totalinvestns=max(daytradeinvestns)
            print ("TOTAL NS CAPITAL  used       ",totalinvestns)
            print ("TOTAL NS OPERATIONS buy/sell ",totaloperns)
            print ("TOTAL NS GAININGS obtained   ",totalgainns)
            print ("EFFEC NS TIVE ROI ",testrange,"day trad ",math.ceil((totalgain/totalinvestns)*100), "%")
            print ("EXPEC NS TED ANNUAL RETURN   ",math.ceil((yeartradingdays*(totalgainns/totalinvestns)/testrange)*100), "%")
            accumulatedoperns=0
        print ("ACCUMULATED GAIN             ",ticker," ",accumulatedgainns)
        if accumulatedoper>0:
            print ("TOTAL CAPITAL used           ",ticker," ",accumulatedinvest)
            print ("TOTAL OPER buy/sell          ",ticker," ",accumulatedoper)
            print ("TOTAL OPERATIONS failed      ",totaloperloss)
            print ("TOTAL OPERATIONS wins        ",totaloperwin)
            print ("TOTAL GAIN obtained          ",ticker," ",accumulatedgain)
            print ("EFFECTIVE ROI 30day          ",ticker," ",math.ceil((accumulatedgain/accumulatedinvest)*100), "%")
            print ("EXPECT ANNUAL RETURN         ",ticker," ",math.ceil((yeartradingdays*(accumulatedgain/accumulatedinvest)/testrange)*100), "%")
            totalinvest=max(daytradeinvest)
            print ("TOTAL CAPITAL  used          ",totalinvest)
            print ("TOTAL OPERATIONS buy/sell    ",totaloper)
            print ("TOTAL GAININGS obtained      ",totalgain)
            print ("EFFECTIVE ROI ",testrange,"day trade    ",math.ceil((totalgain/totalinvest)*100), "%")
            print ("EXPECTED ANNUAL RETURN       ",math.ceil((yeartradingdays*(totalgain/totalinvest)/testrange)*100), "%")
            accumulatedoper=0
        totalinvestns=max(daytradeinvestns)
        totalinvest=max(daytradeinvest)
        i=0
        while i<totaloper:
            print("*DATE: ", tradedate[i]," TICKER: ", tradeticker[i]," ACCURACY: ", tradepredictedgain[i]/tradegain[i])
            print("           BUY THRESHOLD: ",tradebthreshold[i])
            print("          PREDICTED GAIN: ",tradepredictedgain[i],"     REAL GAIN: ",tradegain[i])
            print("        PREDICTED RETURN: ",tradepredictedreturn[i]," REAL RETURN: ",tradereturn[i])
            print("             TRAIN RANGE: ",tradetrnrange[i],"         TEST RANGE: ",tradetstrange[i])
            print("           C MODEL PARAM: ",tradeCparam[i],"    GAMMA MODEL PARAM: ",tradegammaparam[i])
            
            i=i+1
        print ("******* PERIOD REPORT ******* ",end)
        print ("*TOTAL NS CAPITAL  used       ",totalinvestns)
        print ("*TOTAL CAPITAL  used          ",totalinvest)
        print ("*TOTAL NS OPERATIONS buy/sell ",totaloperns)
        print ("*TOTAL OPERATIONS buy/sell    ",totaloper)
        print ("*TOTAL OPERATIONS failed      ",totaloperloss)
        print ("*TOTAL OPERATIONS wins        ",totaloperwin)
        print ("*TOTAL OPERATIONS positive    ",totaloper-totaloperloss)
        print ("*TOTAL OPERATIONS % accuracy  ",(totaloper-totaloperloss)/(0.01+totaloper))
        print ("*TOTAL OPERATIONS chances     ",totaloperchance)
        print ("*TOTAL OPERATIONS % catch     ",(totaloper-totaloperloss)/(0.01+totaloperchance))
        print ("*TOTAL OPERATIONS % predicted ",(totaloperwin)/(0.01+totaloperchance))
        print ("*TOTAL NS GAININGS obtained   ",totalgainns)
        print ("*TOTAL GAININGS obtained      ",totalgain)
        print ("*EFFEC NS TIVE ROI ",testrange,"day trad ",math.ceil((totalgainns/totalinvestns)*10000)/100, "%")
        print ("*EFFECTIVE ROI ",testrange,"day trade    ",math.ceil((totalgain/totalinvest)*10000)/100, "%")
        print ("*EXPEC NS TED ANNUAL RETURN   ",math.ceil((yeartradingdays*(totalgainns/totalinvestns)/testrange)*10000)/100, "%")
        print ("*EXPECTED ANNUAL RETURN       ",math.ceil((yeartradingdays*(totalgain/totalinvest)/testrange)*10000)/100, "%")
        print (daytradeinvest)
        print (daytradegain)    
        i=0
        totaltradereturn=0
        totaltradegain=0
        totaltradeinvest=0
        while i<len(daytradeinvest):
            daytradereturn[i]=daytradegain[i]/daytradeinvest[i]
            print("RETURN: ", daytradereturn[i]," GAIN: " , daytradegain[i], " INVEST: ",daytradeinvest[i])
            totaltradereturn=totaltradereturn+daytradereturn[i]
            totaltradegain=totaltradegain+daytradegain[i]
            totaltradeinvest=totaltradeinvest+daytradeinvest[i]
            i=i+1
        accumulatedinvestns=1
        accumulatedinvest=1
    if endmonth<12:
        endmonth=endmonth+1
    else:
        endmonth=1
        endyear=endyear+1
    end=endyear*10000+endmonth*100+endday
    print('Timestamp: %s' % datetime.now())
    

totalinvestns=max(daytradeinvestns)
totalinvest=max(daytradeinvest)
print ("TOTAL NS CAPITAL  used       ",totalinvestns)
print ("TOTAL CAPITAL  used          ",totalinvest)
print ("TOTAL NS OPERATIONS buy/sell ",totaloperns)
print ("TOTAL OPERATIONS buy/sell    ",totaloper)
print ("TOTAL NS GAININGS obtained   ",totalgainns)
print ("TOTAL GAININGS obtained      ",totalgain)
print ("EFFEC NS TIVE ROI 30day trad ",math.ceil((totalgainns/totalinvestns)*10000)/100, "%")
print ("EFFECTIVE ROI ",testrange,"day trade    ",math.ceil((totalgain/totalinvest)*10000)/100, "%")
print ("EXPEC NS TED ANNUAL RETURN   ",math.ceil((yeartradingdays*(totalgainns/totalinvestns)/testrange)*10000)/100, "%")
print ("EXPECTED ANNUAL RETURN       ",math.ceil((yeartradingdays*(totalgain/totalinvest)/testrange)*10000)/100, "%")
exit()
        