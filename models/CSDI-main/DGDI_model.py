import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer # this is from "https://github.com/lucidrains/linear-attention-transformer?tab=readme-ov-file"

# from performer_pytorch import Performer # use the performer_pytorch package, it is useful for alignment process.
# from xformers.components.attention import MultiHeadDispatch # this way support the local attention.
from trans_layers import GraphTransformer
from generate_adj import *
from layers import *

# Diffusion-based Spatio-Temporal Model with SpatialResidualBlock combining Graph + Temporal + Feature attention
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,              
        nhead=heads,                   
        dim_feedforward=64,            
        activation="gelu"              
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )



def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight) # Kaiming Normal Initialization
    return layer

# similar to FiLM layer
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    

class graph_diff(nn.Module):
    def __init__(self, config, inputdim=2, target_dim=36, device='cuda:0',adj_file=None):
        super().__init__()
        self.channels = config["channels"]
        self.device = device
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # if adj_file == 'AQI_b':
        #     self.adj = get_adj_AQI_b()
        # elif adj_file == 'AQI_t':
        #     self.adj = get_adj_AQI_t()
        if adj_file == 'pems04':
            self.adj = get_similarity_pems04()
        elif adj_file == 'pems08':
            self.adj = get_similarity_pems08()
        elif adj_file == 'discharge' or adj_file == 'pooled':
            self.adj = get_similarity_ssc()
        self.residual_layers = nn.ModuleList(
            [
                SpatialResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    target_dim=target_dim,
                    device=device,
                    adj=self.adj,
                )
                for _ in range(config["layers"])
            ]
        )




    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            # x, skip_connection = layer(x, cond_info, diffusion_emb, self.support)
            x, skip_connection = layer(x, cond_info, diffusion_emb) # no need for self.support
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        h = x.reshape(B, self.channels, K * L) # save intermediate 
        x = x.reshape(B, self.channels, K * L) # prepare for the projection, (B,channel,K*L)
        x = self.output_projection1(x) 
        x = F.relu(x)
        x = self.output_projection2(x) 
        x = x.reshape(B, K, L)
        return x,h
    

class SpatialResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, is_linear=False, adj=None, device=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.adj = adj
        self.device = device
        self.is_linear = is_linear
        # self.temporal_layer = TemporalLearning(channels=channels, nheads=nheads, is_cross=False)
        if is_linear:
            # input layers
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        
        # self.spatial_layer = GCN_MMD(in_channels=channels, hidden_channels=64, out_channels=channels).to(device)
        # self.st_layers = MultiModalFusionTransformer(channels=channels, hidden_dim=64, num_heads=1, adj = adj, num_layers=1, dropout = 0.1)


  
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    # def forward_spatial(self, y, base_shape):
    #     B, channel, K, L = base_shape

    #     if K == 1:
    #         return y
   
    #     y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, K,channel) # (B, L, channel, K)
        
    #     y = self.spatial_layer(y) # 

    #     y=y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
    #     print(f"y.shape before reshape: {y.shape}, total numel={y.numel()}")
    #     print(f"Expected reshape to (B={B}, L={L}, channel={channel}, K={K}), total expected={B*L*channel*K}")
    #     return y # (B, channel, K*L)
    # def forward_spatial(self, y, base_shape):
    #     B, channel, K, L = base_shape
    #     if K == 1:
    #         return y

    #     # (B, channel, K, L) → (B, L, channel, K) → (B*L, K, channel)
    #     y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, K, channel)
    #     # edge_index = torch.stack(self.adj, dim=0).to(self.device)
    #     adj_tensor = torch.tensor(self.adj, dtype=torch.float)
    #     edge_index = adj_tensor.nonzero(as_tuple=False).T.to(self.device)
    #     return y

    # #     # Apply GraphTransformer
    # #     y = self.spatial_layer(y, edge_index)  # output: [B*L, K_new, C_out]

    # #     # Dynamically reshape back
    # #     # B_times_L, K_new, C_out = y.shape
    # #     # assert B_times_L == B * L, f"Mismatch in B*L dimension: got {B_times_L}, expected {B*L}"

    # #     # y = y.reshape(B, L, K_new, C_out).permute(0, 3, 2, 1).reshape(B, C_out, K_new * L)
    # #     y=y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

    # #     # 
    # #     return y


    # # def forward(self, x, cond_info, diffusion_emb, support):
    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        # y = self.forward_spatial(y, base_shape)
        # y = self.st_layers(x, x, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info) 
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    
