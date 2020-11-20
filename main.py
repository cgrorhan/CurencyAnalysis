import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import math,datetime


style.use('ggplot')

api_key = 'V5T95dSBsFP7Um4YMoTx'
quandl.ApiConfig.api_key=api_key

df = quandl.get('BITFINEX/BTCUSD')
df.dropna(inplace=True)

df['HL_PCT'] = (df['High'] - df['Low']) / df['Last'] *100.0
df['ASKBID_PCT'] = (df['Ask'] - df['Bid']) / df['Ask'] *100.0

df=df[['High','Low','Last','HL_PCT','ASKBID_PCT','Volume']]

forecast_out=int(math.ceil(len(df)*0.005))

df['Label']= df['Last'].shift(-forecast_out)

x=df.drop(columns='Label')
x=scale(x)
y=df.iloc[:,-1]

x_toPredict= x[-forecast_out:]
x=x[:-forecast_out]
y=y[:-forecast_out]


x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)

Accuracy = regressor.score(x_test,y_test)

predictionSet= regressor.predict(x_toPredict)
df['Prediction']=np.nan


last_date = df.iloc[-1].name
lastDateTime= last_date.timestamp()
one_day=86400
nextDateTime= lastDateTime+one_day

print(Accuracy)
for i in predictionSet:
    next_date= datetime.datetime.fromtimestamp(nextDateTime)
    nextDateTime += one_day
    df.loc[next_date] = [np.nan for q in range(len(df.columns)-1 )]+[i]


df['Last'].plot(color='b')
df['Prediction'].plot(color='r')
plt.xlabel('Date')
plt.ylabel('Price(USD)')
plt.legend(loc=4)
plt.show()