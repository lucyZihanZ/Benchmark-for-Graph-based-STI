

# Download London_historical_aqi_forecast_stations_20180331.csv into "./GSLI24/Code/Data" from 
# https://www.dropbox.com/s/ht3yzx58orxw179

import numpy as np
import pandas as pd

data1 = pd.read_csv("Data/London_historical_aqi_forecast_stations_20180331.csv",delimiter=",")
site1 = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

outputdata = []
for sitename in site1:
    data = data1[data1["station_id"] == sitename]
    data = data[["PM2.5 (ug/m3)", "PM10 (ug/m3)","NO2 (ug/m3)"]].values

    outputdata.append(data)

data = np.hstack(outputdata)
print(data.shape) # 10897 * 13(station) * 3(feature)

data[np.where(data != data)] = -200


missing_p = np.where(data == -200)

data_reshape  = data.reshape((-1,3))

data_out = []
for f in range(3):
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

data_out = np.array(data_out).T.reshape((-1, 13*3))

data_out[missing_p] = -200
np.savetxt("Data/londonaq/londonaq_norm.csv", data_out, delimiter=",", fmt="%.6f")
