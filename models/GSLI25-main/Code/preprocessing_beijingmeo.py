
# Download beijing_17_18_meo.csv into "./GSLI24/Code/Data" from 
# https://www.dropbox.com/s/jjta4addnyjndd8

import numpy as np
import pandas as pd

data = pd.read_csv("Data/beijing_17_18_meo.csv")

stations = data.iloc[:, 0].unique()
print(stations)
print(len(stations))

all_data = []
for station in stations:
    result = data[data.iloc[:, 0] == station]
    
    start = "2017-01-30 16:00:00"
    end = "2018-01-31 15:00:00"
    freq = "1H"
    date_range = pd.date_range(start=start, end=end, freq=freq)
    
    res = []
    for date in date_range:
        result_ts = result.loc[result['utc_time'] == str(date)]
        
        notfind = result_ts.isnull().values.all()
        
        if notfind:
            station_one_hour = np.array([-200, -200, -200, -200, -200])
            res.append(station_one_hour)
        else:
            station_one_hour = result_ts[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']]
            station_one_hour = station_one_hour.to_numpy()
            station_one_hour = station_one_hour.reshape(station_one_hour.shape[1])
            res.append(station_one_hour)

    res = np.array(res) # 8781,5
    all_data.append(res)

all_data = np.concatenate(all_data, axis=1)
print(all_data.shape)


locations = data[['latitude', 'longitude']].drop_duplicates()
locations.to_csv('Data/beijingmeo_stations.csv', index=False)

data = all_data
data[np.where(data == 999017)] = -200
data[np.where(data == 999999)] = -200
data[np.where(data != data)] = -200
missing_p = np.where(data == -200)

# np.savetxt("beijingmeo.csv", data, delimiter=",", fmt="%.6f")
# data = np.loadtxt("beijingmeo.csv",delimiter=",")

missing_p = np.where(data == -200)

data_reshape = data.reshape((-1,5))

data_out = []
for f in range(5):
    data_t = data_reshape[:, f]
    ls = []
    for item in data_t:
        if item == -200:
            continue
        ls.append(item)
    ls = np.array(ls)
    mean = np.mean(ls)
    std = np.std(ls)
    data_t = (data_t - mean) / std
    data_out.append(data_t)  


data_out = np.array(data_out).T.reshape((-1, 18*5))

data_out[missing_p] = -200

print(len(missing_p[0]) / (data_out.shape[0] * data_out.shape[1]))
np.savetxt("Data/beijingmeo/beijingmeo_norm.csv", data_out, delimiter=",", fmt="%.6f")