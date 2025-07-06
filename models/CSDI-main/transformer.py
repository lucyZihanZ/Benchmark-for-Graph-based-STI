import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv,GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj,num_layers=2,
                 dropout=0.5, use_bn=True, heads=2, out_heads=1):
        super(GAT, self).__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, dropout=dropout, heads=out_heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        device = data.device
        edge_index=torch.tensor(np.array(self.adj.nonzero()), dtype=torch.long).to(device)

        batch_size, seq_len, in_channels = data.shape
        x = data.view(batch_size * seq_len, in_channels)

        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = x.view(batch_size, seq_len, -1)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        device = x.device

        # Use coo() to get edge_index from SparseTensor
        row, col = self.adj.nonzero()
        edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long).to(device)
        print(x.shape)
        batch_size, seq_len, in_channels = x.shape

        x = x.view(batch_size * seq_len, in_channels)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = x.view(batch_size, seq_len, -1)
        return x
    

def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    # Avoid division by zero or NaN if norm is zero
    qs_norm = torch.norm(qs, p=2, dim=-1, keepdim=True)
    qs = qs / (qs_norm + 1e-8) # Add small epsilon for stability

    ks_norm = torch.norm(ks, p=2, dim=-1, keepdim=True)
    ks = ks / (ks_norm + 1e-8) # Add small epsilon for stability

    N = qs.shape[0] # Number of query nodes (B*T*N)

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs) # [N_source, H, M] x [N_source, H, D'] -> [H, M, D']
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N_query, H, M] x [H, M, D'] -> [N_query, H, D']
    attention_num += torch.mean(vs, dim=0, keepdim=True) # Add mean of values to numerator

    # denominator
    all_ones = torch.ones([ks.shape[0]], device=ks.device) # ones tensor with size of N_source
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones) # [N_source, H, M] x [N_source] -> [H, M]
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N_query, H, M] x [H, M] -> [N_query, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, -1)  # [N_query, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N # Add N (num query nodes) to denominator
    # Avoid division by zero
    attn_output = attention_num / (attention_normalizer + 1e-8)  # [N_query, H, D']

    # compute attention for visualization if needed
    if output_attn:
        # attention matrix shape [N_query, N_source, H]
        attention=torch.einsum("nhm,lhm->nlh", qs, ks)
        # Average over heads
        attention=attention.mean(dim=-1) #[N_query, N_source]

        # Normalizer for the attention matrix itself
        # Using the same normalizer as the output aggregation, averaged over heads
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N_query, 1]
        # Avoid division by zero
        attention=attention / (normalizer + 1e-8)


    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        # assume query_input and source_input\
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)
        # num_nodes_query = query_input.shape[0]
        # num_nodes_source = source_input.shape[0]

        # query = self.Wq(query_input).reshape(num_nodes_query, self.num_heads, self.out_channels)
        # key   = self.Wk(source_input).reshape(num_nodes_source, self.num_heads, self.out_channels)
        # if self.use_weight:
        #     value = self.Wv(source_input).reshape(num_nodes_source, self.num_heads, self.out_channels)
        # else:
            # value = source_input.reshape(num_nodes_source, 1, self.out_channels) # # Still requires in_channels == out_channels

        # compute full attentive aggregation
        if value.shape[1] == 1 and self.num_heads > 1:
             value = value.repeat(1, self.num_heads, 1) # Repeat along head dimension

        # Check if num_heads match for query, key, value before passing to full_attention_conv
        if not (query.shape[1] == key.shape[1] == value.shape[1]):
             raise ValueError(f"Head dimensions mismatch: query {query.shape}, key {key.shape}, value {value.shape}")

        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, adj, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()
        self.adj = adj # sparse matrix
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

        self.edge_index, self.edge_weight = self.get_edge_index_and_weight()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
#
    def get_edge_index_and_weight(self):
# get the weights of each edge and node
        if isinstance(self.adj, SparseTensor):
            row, col, value = self.adj.coo()
            edge_index = torch.stack([row, col], dim=0)

            if value is None:
                # Use symmetric normalization if no edge weights are present
                deg = torch.bincount(row, minlength=self.adj.size(0)).float()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            else:
                edge_weight = value  # Use original weights

        elif isinstance(self.adj, (np.ndarray, sp.spmatrix)):
            row, col = self.adj.nonzero()
            # row = np.array(row).flatten()
            # col = np.array(col).flatten()
            edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
            deg = np.array(self.adj.sum(axis=1)).flatten()
            deg_inv_sqrt = np.power(deg.astype(np.float64), -0.5) # Use float64 for sqrt stability
            deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0 # Handle zeros or NaNs

            # Apply symmetric normalization: edge_weight_ij = deg_i^{-0.5} * deg_j^{-0.5}
            edge_weight = torch.tensor(deg_inv_sqrt[row] * deg_inv_sqrt[col], dtype=torch.float)

        else:
            raise TypeError(f"Unsupported adjacency matrix type: {type(self.adj)}")

        return edge_index, edge_weight



# x means the input feature matrix
# x.shape = [B, T, N, D]

    def forward(self, x):
      """
      x: Tensor of shape [B, T, N, D]
      Output: Tensor of shape [B, T, N, hidden_channels]
      """
      B, T, N, D = x.shape
      x = x.view(B * T, N, D)  # [B*T, N, D]

      x = self.fcs[0](x)  # [B*T, N, H]
      if self.use_bn:
          x = self.bns[0](x)
      x = self.activation(x)
      x = F.dropout(x, p=float(self.dropout), training=self.training)

    #   layer_ = [x]
      for i, conv in enumerate(self.convs):
          # input_to_current_conv holds the output of the previous layer (or initial transform)
          input_to_current_conv = x # Shape: [B*T, N, hidden_channels]

          # TransConvLayer expects [Num_Nodes, Feature_Dim]. Flatten [B*T, N, H] to [B*T*N, H].
          input_flat = input_to_current_conv.view(B * T * N, -1) # Shape: [B*T*N, hidden_channels]

          # Apply the graph convolutional layer (which is a transformer layer here)
          # The layer operates on the flattened node dimension.
          # The output x_flat will be [B*T*N, hidden_channels]
          # Note: edge_index/edge_weight are currently ignored by TransConvLayer
          x_flat, attn = conv(input_flat, input_flat, edge_index=None, edge_weight=None, output_attn=True)

          # Reshape the output back to [B*T, N, hidden_channels] before residual/norm/act
          x = x_flat.view(B*T, N, -1) # Shape: [B*T, N, hidden_channels]


          # Apply residual connection
          if self.residual:
              # Ensure dimensions match for residual connection: [B*T, N, H] + [B*T, N, H]
              # x is now the output of the conv layer reshaped: [B*T, N, hidden_channels]
              # input_to_current_conv is the input to this conv layer: [B*T, N, hidden_channels]
              x = self.alpha * x + (1 - self.alpha) * input_to_current_conv


          if self.use_bn:
              # Apply LayerNorm after residual
              # self.bns[i + 1] corresponds to the norm after the i-th conv layer
              x = self.bns[i + 1](x)

          if self.use_act:
              x = self.activation(x) # Apply activation after norm

          # Dropout after norm and activation
          x = F.dropout(x, p=float(self.dropout), training=self.training)

          # x is now the processed output of this layer, ready to be the input for the next iteration

            # Reshape back to [B, T, N, hidden_channels] for the final output
          x = x.view(B, T, N, -1)
          return x

    def get_attentions(self, x):
        # layer_, attentions = [], []
        attentions = []
        edge_index = self.edge_index.to(x.device)
        edge_weight = self.edge_weight.to(x.device)
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        # layer_.append(x)
        for i, conv in enumerate(self.convs):
            input_to_conv = x
            x, attn = conv(input_to_conv, input_to_conv, edge_index, edge_weight, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * input_to_conv
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            # layer_.append(x)
        # [num_layers, B*T, N, N] — per-head attention maps
        return torch.stack(attentions, dim=0)