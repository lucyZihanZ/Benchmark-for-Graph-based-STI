# Download airquality.csv into "./GSLI24/Code/Data" from 
# http://research.microsoft.com/apps/pubs/?id=246398


import numpy as np
import pandas as pd

data = pd.read_csv("Data/airquality.csv")
data.fillna(-200, inplace=True)


station_data = pd.read_csv("Data/station.csv")
stations = station_data.iloc[:, 0].unique()

print(len(stations))

all_data = []
for station in stations:
    result = data[data.iloc[:, 0] == station]
    
    start = "2014-10-01 00:00:00"
    end = "2014-12-31 18:00:00"
    freq = "1H"
    date_range = pd.date_range(start=start, end=end, freq=freq)
    
    res = []
    for date in date_range:
        result_ts = result.loc[result['time'] == str(date)]
        
        notfind = result_ts.isnull().values.all()
        
        if notfind:
            station_one_hour = np.array([-200, -200, -200, -200, -200, -200])
            res.append(station_one_hour)
        else:
            station_one_hour = result_ts[['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration']]
            station_one_hour = station_one_hour.to_numpy()
            station_one_hour = station_one_hour.reshape(station_one_hour.shape[1])
            res.append(station_one_hour)

    res = np.array(res) # 2203,6
    all_data.append(res)

all_data = np.concatenate(all_data, axis=1)
print(all_data.shape)

missing_p = np.where(all_data == -200)

data_reshape = all_data.reshape((-1,6))

data_out = []
for f in range(6):
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


data_out = np.array(data_out).T.reshape((-1, 140*6))

data_out[missing_p] = -200

print(len(missing_p[0]) / (data_out.shape[0] * data_out.shape[1]))
np.savetxt("Data/cn/cn_norm.csv", data_out, delimiter=",", fmt="%.6f")