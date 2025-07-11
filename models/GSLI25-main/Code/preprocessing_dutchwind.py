
import numpy as np

stations = [240,260,270,280,290,310,370]
alldata = []

for station in stations:
    data = []
    with open('Data/jaar.txt', 'r') as f:
        lines = f.readlines()[33:]
        for line in lines:
            line = line.strip().split(',')
            if int(line[0]) != station:
                continue

            line = line[3:7]
            line = [float(item) for item in line]
            data.append(line)
    data = np.array(data)
    alldata.append(data)

alldata = np.hstack(alldata)

# np.savetxt("dutchind.csv", alldata, delimiter=",", fmt="%.6f")
# data = np.loadtxt("dutchind.csv", delimiter=",")

data = alldata

missing_p = np.where(data == 0)

missp = len(missing_p[0]) / (data.shape[0] * data.shape[1])
print(missp)
data[missing_p] = -200

data_reshape  = data.reshape((-1,4))

data_out = []
for f in range(4):
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

data_out = np.array(data_out).T.reshape((-1, 7*4))
# print(data_out.shape)
data_out[missing_p] = -200

np.savetxt("Data/dutchwind/dutchwind_norm.csv", data_out, delimiter=",", fmt="%.6f")