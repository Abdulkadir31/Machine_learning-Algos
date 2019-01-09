import datetime
import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("NSE/RELIANCE", authtoken="zrzu-RebsbmokqDRjwE5",)

df = df[['Open','High','Low','Close','Total Trade Quantity']]
df['HL_percent'] = (df['High']-df['Low'])/df['Low']*100
df['Percent_change'] = (df['Close']-df['Open'])/df['Open']*100
df = df[['Close','HL_percent','Percent_change','Total Trade Quantity']]

forecast_col = 'Close'
df.fillna('0',inplace=True)


forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]




df.dropna(inplace=True)
y = np.array(df['label'])





X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
with open('Stocks_Prediction.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('Stocks_Prediction.pickle','rb')
clf = pickle.load(pickle_in)


forecast_set = clf.predict(X_lately)

#print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name

last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
count = 0
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)

    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]




df['Forecast'].plot()
df.Close=df.Close.astype(float)
df['Close'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()