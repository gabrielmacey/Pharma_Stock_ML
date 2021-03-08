import numpy as np
import pandas as pd
import yfinance as yf
from pandas import datetime
import matplotlib.pyplot as plt


def get_stock_data(stock_symbol):
    print(stock_symbol)
    ticker = yf.Ticker(stock_symbol)
    stock_data = ticker.history(interval="1d", period="10y")
    
    stock_df = pd.DataFrame(stock_data)
    stock_df.insert(0, 'stock_name', stock_symbol)
    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={"Date":"date", 
                         "Open": "open", 
                         "High": "high",
                         "Low": "low",
                         "Close": "close", 
                         "Volume": "volume",}, inplace=True)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    return stock_df
    

def run_facebook_prophet_model(stock_df):
    from fbprophet import Prophet
    print(type(stock_df))
    features = stock_df[['date','close']]
    features = features.rename(columns={'date':'ds', 'close':'y'})
    print(features.head(5))
    m = Prophet(daily_seasonality = True) # the Prophet class (model)
    m.fit(features) # fit the model using all data
    future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
    prediction = m.predict(future)
    # m.plot(prediction)
    # plt.title("Prediction of the BMY Stock Price using the Prophet")
    # plt.xlabel("Date")
    # plt.ylabel("Close Stock Price")
    # plt.show()
    # m.plot_components(prediction)
    #plt.show()
    print(prediction.columns)

    plot_df =  pd.DataFrame({'date': features['ds'],
                        'prediction': prediction['yhat'],
                        'Actual': features['y']})
    to_json = plot_df.to_json(orient = 'records')
    return to_json

def run_arima_model(stock_df):
    from  statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    from pandas.plotting import lag_plot
    print(stock_df.isnull().sum())

    """
    plt.figure(figsize=(12,8))
    lag_plot(stock_df['close'], lag=5)
    plt.title('BMY Stock - Autocorrelation plot with lag = 5')
    plt.show()

    plt.figure(figsize=(16,8))
    plt.plot(stock_df["date"], stock_df["close"])
    xticks = pd.date_range(datetime.datetime(2010,1,1), datetime.datetime(2021,1,1), freq='YS')
    xticks=xticks.to_pydatetime()
    plt.xticks(xticks)
    plt.title("BMY stock price over time")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.show()
    """
    
    X_train, X_test = stock_df[0:int(len(stock_df)*0.8)], stock_df[int(len(stock_df)*0.8):]
    X_train = X_train.set_index('date')
    X_test = X_test.set_index('date')

    """ plt.figure(figsize=(12,8))
    ax=X_train.plot(grid=True, figsize=(12,8))
    X_test.plot(ax=ax,grid=True)
    plt.legend(['X_test', 'X_train'])
    plt.show() """

    training_data = X_train['close'].values
    test_data = X_test['close'].values

    #p is the lag order, relationship bewtween current observation and the previous observations, 
    # the model takes a number and calculates the lag
    #d is the difference (subtracting an aobeservation from its previous step) applied to remove trends in the data
    #q Moving avaerage is the relationship betwen observation and the error 
    import warnings
    warnings.filterwarnings('ignore')
    #get the history of all prices from data
    history = [x for x in training_data]
    model_predictions = []
    number_of_test_observations = len(test_data)
    for time_point in range(number_of_test_observations):
        model = ARIMA(history, order=(1,1,1)) 
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))

    test_date_range = X_test.index 

    plot_df =  pd.DataFrame({'date': test_date_range,
                        'prediction': model_predictions,
                        'actual': test_data})
    print(plot_df)
    """ import plotly.express as go
    fig = go.line(plot_df,x='date',y=['prediction','actual'])
    fig.show() """
    to_json = plot_df.to_json(orient='records')
    return to_json


# if __name__ == "__main__":
#     stock_df = get_stock_data('GME')
#     #run_facebook_prophet_model(stock_df)
#     run_arima_model(stock_df)