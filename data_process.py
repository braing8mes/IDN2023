import pandas as pd
fraud = pd.read_csv('setD_90_fraud.csv', names = ['index', 'time', 'store', 'location'], parse_dates=['time'])
norm = pd.read_csv('setD_90_norm.csv', names = ['index', 'time', 'store', 'location'], parse_dates=['time'])
fraud['fraud'] = 1
norm['fraud'] = 0
data = pd.concat([fraud, norm])
#data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S.%f')
data = data.sort_values(by = ['index', 'time'], ascending = True)
data.to_csv('setD_combined.csv', index = False)

                   