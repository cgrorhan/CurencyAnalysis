import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math,datetime


style.use('ggplot')

df_reversed = pd.read_csv('TTKOM.csv')
df = df_reversed[::-1]
df.drop(columns='Change %', inplace=True)


for i in df['Date']:
    datetime_obj= datetime.datetime.strptime(i,"%b %d, %Y")
    df['Date'].loc[df['Date'].values == i] = datetime_obj

df.index=df['Date']

for i in df['Vol.']:
    q=i
    if str(i).endswith('K'):
        i=float(str(i[:-1]))*1000.0
        df['Vol.'].loc[df["Vol."].values==q] = i
    elif str(i).endswith('M'):
        i = float(str(i[:-1])) * 1000000.0
        df['Vol.'].loc[df["Vol."].values == q] = i


df['HL_PCT']=(df['High']-df['Low']) / df['Open']*100.0
df['PCT_CHG']=(df['Price']-df['Open']) / df['Price']*100.0

df=df[['Price','HL_PCT','PCT_CHG','Vol.']]

forecast_out=int(math.ceil(len(df)*0.010))
df['Prediction']=df['Price'].shift(-forecast_out)

x=df.drop(columns='Prediction')
y=df.iloc[:,-1]
"""
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
"""

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x_toPredict=x[-forecast_out:]
x= x[:-forecast_out]
y=y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
accuracy=regressor.score(x_test,y_test)
print(accuracy)

pred_set=regressor.predict(x_toPredict)

df['Prediction']=np.nan

last_date=df.iloc[-1].name
lastDateTime=datetime.datetime.timestamp(last_date)
one_day=86400
nextDateTime=lastDateTime+ one_day

for i in pred_set:
    next_date= datetime.datetime.fromtimestamp(nextDateTime)
    nextDateTime +=one_day
    df.loc[next_date]=[np.nan for x in range(len(df.columns)-1)]+[i]

df['Price'].plot(color='r')
df['Prediction'].plot(color='b')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price(TRY)')
plt.show()