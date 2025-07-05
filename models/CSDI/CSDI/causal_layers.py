import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import os


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, activation=F.relu, dropout=0.1):
        super(SpatialEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.convs.append(nn.Linear(in_dim, out_channels))

    def forward(self, x, adj):
        """
        x: (B, N, C)  -- features at one time step
        adj: (N, N) or (B, N, N) -- causal adjacency matrix
        """
        B, N, C = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).repeat(B, 1, 1)  # expand to (B, N, N)

        A_hat = adj + torch.eye(N).to(x.device)  # add self-loop
        D_hat = torch.diag_embed(torch.sum(A_hat, dim=-1) ** -0.5)
        A_norm = D_hat @ A_hat @ D_hat  # symmetric normalization

        out = x
        for layer in self.convs:
            out = A_norm @ out  # graph conv
            out = layer(out)
            out = self.activation(out)
            out = self.dropout(out)
        return out  # shape: (B, N, out_channels)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        """
        x: (B, N, C) — batch × nodes × features
        adj: (N, N) or (B, N, N) — causal graph adjacency matrix
        """
        B, N, C = x.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, N, N)  # expand to batch shape

        A_hat = adj + torch.eye(N).to(adj.device)  # add self-loops
        D_hat = torch.diag_embed(torch.sum(A_hat, dim=-1) ** -0.5)
        A_norm = D_hat @ A_hat @ D_hat  # symmetric normalization

        out = A_norm @ x  # (B, N, C)
        out = self.linear(out)  # (B, N, out_channels)
        return out


# define the adjacent matrix
def geographical_distance(latlon):
    """返回站点之间的地球表面距离（单位：公里）"""
    R = 6371.0088  # Earth radius in kilometers
    radians = np.radians(latlon)
    dist = haversine_distances(radians) * R
    return dist

def thresholded_gaussian_kernel(x, theta=None, threshold=0.1):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    weights[weights < threshold] = 0.0
    return weights

def build_adjacency(location_csv_path, save_path, threshold=0.1):
    df = pd.read_csv(location_csv_path)
    latlon = df[["latitude", "longitude"]].values
    dist = geographical_distance(latlon)
    
    print("📏 平均距离（km）:", np.mean(dist), " | 标准差:", np.std(dist))

    adj = thresholded_gaussian_kernel(dist, threshold=threshold)
    np.fill_diagonal(adj, 1.0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, adj)
    print(f"✅ 邻接矩阵已保存至: {save_path} | shape: {adj.shape}")

    return adj

# 示例调用
if __name__ == "__main__":
    adj = build_adjacency("data/SampleData/pm25_latlng.txt", save_path="data/SampleData/adj_causal.npy", threshold=0.1)


