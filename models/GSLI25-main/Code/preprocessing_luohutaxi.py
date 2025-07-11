
# Download sz_adj.csv, sz_speed.csv into "./GSLI24/Code/Data" from 
# https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data

import numpy as np

adj = np.loadtxt("Data/sz_adj.csv",delimiter=",")
data = np.loadtxt("Data/sz_speed.csv",delimiter=",")[1:]

miss_p = np.where(data == 0)
complete_p = np.where(data != 0)

# all = data.shape[0] * data.shape[1]
# m = len(miss_p[0])

means = np.mean(data[complete_p])
stds = np.std(data[complete_p])

data[complete_p] = (data[complete_p] - means) / stds
data[miss_p] = -200

for i in range(adj.shape[0]):
    adj[i,i] = 1

np.savetxt("Data/luohutaxi/luohutaxi_norm.csv", data, delimiter=",", fmt="%.6f")
np.savetxt("Data/luohutaxi/luohutaxi_adj.csv", adj, delimiter=",", fmt="%.6f")