import yfinance as yf, pandas as pd

df = yf.download('BTC-USD', start='2013-01-01', interval='1d', progress=False)
print(type(df))
print(df.columns)
print(df.head())

price_df = df[['Close']].rename(columns={'Close':'Price'})
print(type(price_df))
print(price_df.head())
