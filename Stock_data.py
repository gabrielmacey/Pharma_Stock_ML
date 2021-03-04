#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Must have pandas, pymongo, and yfinace installed on environment
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[2]:


#API calls
#AstraZeneca
azn = yf.Ticker("AZN")
#Bristol Myers Squibb
bmy = yf.Ticker("BMY")
#Johnson & Johnson
jnj = yf.Ticker("JNJ")
#Merck
mrk = yf.Ticker("MRK")
#Pfizer
pfe = yf.Ticker("PFE")


# In[3]:


#Extracting history data from stocks (parameters can change)
#Period parameters - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
#Interval parameters - 1m, 2m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo
azn_data = azn.history(interval="1d", period="10y")
bmy_data = bmy.history(interval="1d", period="10y")
jnj_data = jnj.history(interval="1d", period="10y")
mrk_data = mrk.history(interval="1d", period="10y")
pfe_data = pfe.history(interval="1d", period="10y")


# In[4]:


#Converting to a DataFrame
azn_df = pd.DataFrame(azn_data)
bmy_df = pd.DataFrame(bmy_data)
jnj_df = pd.DataFrame(jnj_data)
mrk_df = pd.DataFrame(mrk_data)
pfe_df = pd.DataFrame(pfe_data)


# In[5]:


#Inserting an identifier to know which stock the information came from
azn_df.insert(0, 'stock_name', 'AZN')
bmy_df.insert(0, 'stock_name', 'BMY')
jnj_df.insert(0, 'stock_name', 'JNJ')
mrk_df.insert(0, 'stock_name', 'MRK')
pfe_df.insert(0, 'stock_name', 'PFE')


# In[6]:


#Add DataFrames to a list
dataframes = [azn_df, bmy_df, jnj_df, mrk_df, pfe_df]
#Concatinate this list into one DataFrame
stock_df = pd.concat(dataframes)


# In[7]:


#Reset index
stock_df.reset_index(inplace=True)


# In[8]:


#Remove unnecessary columns
stock_df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)


# In[9]:


stock_df.rename(columns={"Date":"date", 
                         "Open": "open", 
                         "High": "high",
                         "Low": "low",
                         "Close": "close", 
                         "Volume": "volume",}, inplace=True)


# In[10]:


stock_df


# In[50]:


stock_df['date'] = pd.to_datetime(stock_df['date'])


# In[51]:


from pandas.plotting import lag_plot
import numpy as np
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[69]:


azn_df = stock_df[stock_df['stock_name'] == 'AZN']
#azn_df = azn_df.set_index('date')
azn_df


# In[70]:


azn_df.isnull().sum()


# In[71]:


azn_df.shape


# In[72]:


azn_df.corr()


# In[73]:


plt.figure(figsize=(16,8))
lag_plot(azn_df['open'], lag=5)
plt.title('AZN Stock - Autocorrelation plot with lag = 5')
plt.show()


# In[74]:


plt.figure(figsize=(16,8))
plt.plot(azn_df["date"], azn_df["close"])
xticks = pd.date_range(datetime.datetime(2010,1,1), datetime.datetime(2021,1,1), freq='YS')
xticks=xticks.to_pydatetime()
plt.xticks(xticks)
plt.title("AZN stock price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.show()


# In[116]:


X_train, X_test = azn_df[0:int(len(azn_df)*0.8)], azn_df[int(len(azn_df)*0.8):]
X_train = X_train.set_index('date')
X_test = X_test.set_index('date')
X_test


# In[117]:


plt.figure(figsize=(12,8))
ax=X_train.plot(grid=True, figsize=(12,8))
X_test.plot(ax=ax,grid=True)
plt.legend(['X_test', 'X_train'])
plt.show()


# In[118]:


#X_train, X_test = azn_df[0:int(len(azn_df)*0.8)], azn_df[int(len(azn_df)*0.8):]
training_data = X_train['close'].values
test_data = X_test['close'].values


# In[119]:


import warnings
warnings.filterwarnings('ignore')
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))


# In[121]:


X_test


# In[122]:


test_date_range = X_test.index #azn_df[int(len(azn_df)*0.8):].index
test_date_range


# In[129]:


plt.figure(figsize=(12,12))
plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title('AZN Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(2012,2516,50), azn_df.date[2012:2516:50])
plt.legend()
plt.show()


# In[ ]:




