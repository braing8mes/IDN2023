import pandas as pd

df = pd.read_csv('setD_combined.csv', parse_dates=['time'])
data = df[df.groupby("index")['index'].transform('size') > 1]
#data = data[data.groupby("index")['index'].transform('size')<100]
data['time_diff'] = data.groupby(['index'])['time'].diff().fillna(pd.Timedelta(seconds=0))
data['time_diff'] = data['time_diff'] / pd.to_timedelta(1, unit='s')
data.to_csv('setD_90_mult.csv', index = False)
