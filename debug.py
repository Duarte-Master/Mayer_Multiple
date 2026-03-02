print('debug script start')
import yfinance as yf, pandas as pd, numpy as np

print('imports done')
start_date="2013-01-01"
df=yf.download("BTC-USD", start=start_date, interval="1d", progress=False)
print('df start', df.index.min())
print(df.head())

price_df=df[["Close"]].rename(columns={"Close":"Price"})
if isinstance(price_df.columns,pd.MultiIndex): price_df.columns=price_df.columns.get_level_values(0)
print('price_df start', price_df.index.min())

price_data=pd.to_numeric(price_df["Price"], errors="coerce")
price_data.dropna(inplace=True)
print('after dropna', price_data.index.min(), price_data.index.max(), price_data.iloc[:5])

# reindex
start=price_data.index.min(); end=price_data.index.max()
full=pd.date_range(start=start,end=end,freq='D')
price_data=price_data.reindex(full).ffill()
print('after reindex sample', price_data.iloc[:10])

ma=price_data.rolling(window=200,min_periods=200).mean()
print('ma first non-nan', ma.first_valid_index())
print('ma last', ma.iloc[-1])
print('debug script end')
