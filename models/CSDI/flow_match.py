import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE Encoder for Spatio-Temporal Data [B, T, N, F] → z
class SpatioTemporalVAEEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        B, T, N, F = input_shape
        self.input_dim = T * N * F
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.to_mu = nn.Linear(128, latent_dim)
        self.to_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, mask):
        x = x * mask  # mask out missing values
        x_flat = x.view(x.shape[0], -1)  # flatten to [B, T*N*F]
        h = self.encoder(x_flat)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar


# Simple Flow Network z_src → z_tgt
class FlowNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z_src):
        return self.flow(z_src)


# Decoder z → x_hat
class SimpleDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super().__init__()
        self.output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        self.output_shape = output_shape

    def forward(self, z):
        x_flat = self.decoder(z)
        x_hat = x_flat.view(-1, *self.output_shape)
        return x_hat


# VRFM Loss
def vrfm_loss(z_src, z_tgt, mu_tgt, logvar_tgt, flow_net):
    z_src_flow = flow_net(z_src)
    flow_match_loss = F.mse_loss(z_src_flow, z_tgt)
    kl = -0.5 * torch.sum(1 + logvar_tgt - mu_tgt.pow(2) - logvar_tgt.exp(), dim=1).mean()
    return flow_match_loss + kl


# Training step
def train_step(encoder, flow_net, optimizer, x_src, mask_src, x_tgt, mask_tgt):
    encoder.train()
    flow_net.train()
    z_src, _, _ = encoder(x_src, mask_src)
    z_tgt, mu_tgt, logvar_tgt = encoder(x_tgt, mask_tgt)
    loss = vrfm_loss(z_src, z_tgt, mu_tgt, logvar_tgt, flow_net)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

