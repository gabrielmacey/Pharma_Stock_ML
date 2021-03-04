import pandas as pd
import yfinance as yf

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

#Extracting history data from stocks (parameters can change)
#Period parameters - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
#Interval parameters - 1m, 2m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo
azn_data = azn.history(interval="1d", period="10y")
bmy_data = bmy.history(interval="1d", period="10y")
jnj_data = jnj.history(interval="1d", period="10y")
mrk_data = mrk.history(interval="1d", period="10y")
pfe_data = pfe.history(interval="1d", period="10y")

#Converting to a DataFrame
azn_df = pd.DataFrame(azn_data)
bmy_df = pd.DataFrame(bmy_data)
jnj_df = pd.DataFrame(jnj_data)
mrk_df = pd.DataFrame(mrk_data)
pfe_df = pd.DataFrame(pfe_data)

#Inserting an identifier to know which stock the information came from
azn_df.insert(0, 'stock_name', 'AZN')
bmy_df.insert(0, 'stock_name', 'BMY')
jnj_df.insert(0, 'stock_name', 'JNJ')
mrk_df.insert(0, 'stock_name', 'MRK')
pfe_df.insert(0, 'stock_name', 'PFE')

#Add DataFrames to a list
dataframes = [azn_df, bmy_df, jnj_df, mrk_df, pfe_df]
#Concatinate this list into one DataFrame
stock_df = pd.concat(dataframes)

#Reset index
stock_df.reset_index(inplace=True)

#Remove unnecessary columns
stock_df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

stock_df.rename(columns={"Date":"date",
                         "Open": "open",
                         "High": "high",
                         "Low": "low",
                         "Close": "close",
                         "Volume": "volume",}, inplace=True)
