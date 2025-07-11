
# Download los_adj.csv, los_speed.csv into "./GSLI24/Code/Data" from 
# https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data

import numpy as np

adj = np.loadtxt("Data/los_adj.csv",delimiter=",")
data = np.loadtxt("Data/los_speed.csv",delimiter=",")[1:]

x = []
y = []

# The dataset is pre-interpolation, we should find the real missing data
for j in range(data.shape[1]):
    for i in range(data.shape[0]-2):
        if data[i+1,j] == (data[i,j] + data[i+1, j]) / 2:
            x.append(i+1)
            y.append(j)
x = np.array(x)
y = np.array(y)
miss_p = (x,y)

complete_p = np.ones_like(data)
complete_p[miss_p] = 0
complete_p = np.where(complete_p != 0)

means = np.mean(data[complete_p])
stds = np.std(data[complete_p])

data[complete_p] = (data[complete_p] - means) / stds
data[miss_p] = -200


np.savetxt("Data/los/los_norm.csv", data, delimiter=",", fmt="%.6f")
np.savetxt("Data/los/los_adj.csv", adj, delimiter=",", fmt="%.6f")