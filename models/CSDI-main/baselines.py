import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable, Function
import math
import pdb
import argparse
from dataset_process.dataset_pooled import get_dataloader_pooled
from dataset_process.dataset_discharge import get_dataloader_discharge

# Utility function to calculate the coefficient for the Gradient Reversal Layer
# def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
#     """
#     Calculates a coefficient that changes during training, typically used for
#     the Gradient Reversal Layer. It smoothly transitions from 'low' to 'high'.
#     """
#     return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """
    Calculates a coefficient that changes during training, typically used for
    the Gradient Reversal Layer. It smoothly transitions from 'low' to 'high'.
    """
    # Convert inputs to torch tensors and ensure float32
    iter_num_t = torch.tensor(iter_num, dtype=torch.float32)
    high_t = torch.tensor(high, dtype=torch.float32)
    low_t = torch.tensor(low, dtype=torch.float32)
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    max_iter_t = torch.tensor(max_iter, dtype=torch.float32)

    # Perform calculations using torch operations
    # This ensures all intermediate calculations are in float32
    numerator = 2.0 * (high_t - low_t)
    denominator = 1.0 + torch.exp(-alpha_t * iter_num_t / max_iter_t)
    
    return numerator / denominator - (high_t - low_t) + low_t

# Utility function to initialize weights of network layers
def init_weights(m):
    """
    Initializes weights for different types of layers.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# Gradient Reversal Layer (GRL) implementation
class GradientReversalFunction(Function):
    """
    A custom autograd function for the Gradient Reversal Layer.
    During the forward pass, it acts as an identity.
    During the backward pass, it multiplies the gradients by a negative scalar (coeff).
    """
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.coeff), None

class GRL(nn.Module):
    """
    Gradient Reversal Layer module.
    Wraps the GradientReversalFunction to be used as a PyTorch module.
    """
    def __init__(self):
        super(GRL, self).__init__()
        self.coeff = 1.0 # Initial coefficient, will be updated dynamically

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.coeff)

    def set_coeff(self, coeff):
        """
        Sets the coefficient for gradient reversal.
        """
        self.coeff = coeff

# --- Feature Extractors (Backbones) ---
# These classes are adapted from your original code to serve as feature extractors
# and classifiers. They return both the features before the final classification
# layer and the classification output.

# resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class GRUFeatureExtractor(nn.Module):
  """
  A feature extractor using a GRU network, suitable for sequence data.
  """
  def __init__(self, input_dim, hidden_dim, num_layers=2):
    super(GRUFeatureExtractor, self).__init__()
    self.gru = nn.GRU(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True # Bidirectional helps capture context from past and future
    )
    self.__in_features = hidden_dim * 2 # Multiply by 2 for bidirectional
    self.apply(init_weights)

  def forward(self, x):
    """
    The input x is expected to have shape (batch_size, seq_len, input_dim).
    The output will be the hidden states from the GRU.
    """
    # GRU returns output (batch, seq_len, num_directions * hidden_size)
    # and h_n (num_layers * num_directions, batch, hidden_size)
    features, _ = self.gru(x)
    return features

  def output_num(self):
    return self.__in_features


class ImputationHead(nn.Module):
    """
    A regression head to predict missing values.
    It takes features from the GRU and outputs an imputed value.
    """
    def __init__(self, feature_dim, output_dim=1):
        super(ImputationHead, self).__init__()
        self.impute_layer = nn.Linear(feature_dim, output_dim)
        self.impute_layer.apply(init_weights)

    def forward(self, features):
        """
        Forward pass for the imputation head.
        Args:
            features (torch.Tensor): Features from the GRUFeatureExtractor.
                                     Shape: (batch_size, seq_len, feature_dim)
        Returns:
            torch.Tensor: Imputed values. Shape: (batch_size, seq_len, output_dim)
        """
        imputed_values = self.impute_layer(features)
        return imputed_values


class ConditionalDomainDiscriminator(nn.Module):
    """
    A domain discriminator that distinguishes between source and target domains,
    conditioned on the imputation output.
    """
    def __init__(self, feature_dim, output_dim, hidden_size=512):
        super(ConditionalDomainDiscriminator, self).__init__()
        # Input is the concatenation of features and the imputed values
        self.input_dim = feature_dim + output_dim
        self.grl = GRL()

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1) # Outputs a logit for domain classification
        )
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, features, imputation_output):
        """
        Forward pass for the conditional domain discriminator.
        Args:
            features (torch.Tensor): Features from the backbone network.
            imputation_output (torch.Tensor): Imputed values from the head.
        """
        if self.training:
            self.iter_num += 1
        
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        self.grl.set_coeff(coeff)

        # The GRL reverses the gradient for the feature extractor
        reversed_features = self.grl(features)
        
        # Concatenate features and imputation output for conditioning
        combined_input = torch.cat((reversed_features, imputation_output), dim=-1)

        y = self.discriminator(combined_input) # Output domain logit (source vs. target)
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


# --- Main CDAN Imputation Model ---

class CDANImputationModel(nn.Module):
    """
    Main CDAN model for domain-adaptive spatial-temporal imputation.
    """
    def __init__(self, input_dim, hidden_dim, num_gru_layers=2):
        super(CDANImputationModel, self).__init__()
        self.input_dim = input_dim

        self.feature_extractor = GRUFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers
        )
        
        feature_dim = self.feature_extractor.output_num()

        self.imputation_head = ImputationHead(
            feature_dim=feature_dim,
            output_dim=self.input_dim
        )

        self.domain_discriminator = ConditionalDomainDiscriminator(
            feature_dim=feature_dim,
            output_dim=self.input_dim
        )

    def forward(self, x):
        """
        Forward pass through the entire model.
        Args:
            x (torch.Tensor): Input sequence. Shape: (batch_size, seq_len, input_dim)
        Returns:
            tuple:
                - imputation_output (torch.Tensor): The final imputed sequence.
                - domain_output (torch.Tensor): Logit for domain prediction.
        """
        features = self.feature_extractor(x)
        imputation_output = self.imputation_head(features)
        domain_output = self.domain_discriminator(features, imputation_output)
        
        return imputation_output, domain_output

    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_mult": 1, 'decay_mult': 2},
            {"params": self.imputation_head.parameters(), "lr_mult": 10, 'decay_mult': 2},
            *self.domain_discriminator.get_parameters()
        ]


def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create DataLoaders ---
    # User should replace with their actual data paths
    source_train_loader, source_valid_loader, source_test_loader, source_scaler, source_mean = get_dataloader_discharge(
        args.batch_size, args.device, seed=1, val_len=0.1, num_workers=1, eval_missing_pattern='block', train_missing_pattern='block')
    target_train_loader, target_valid_loader, target_test_scaler, target_scaler, target_mean = get_dataloader_pooled(
        args.batch_size, args.device, seed=1, val_len=0.1, num_workers=1, 
                   eval_missing_pattern='block', train_missing_pattern='block')
    
    # Use the same scaler for both, typically from the source domain
    scaler = source_scaler.to(device)
    mean_scaler = source_mean.to(device)

    # --- Initialize Model, Optimizer, and Loss ---
    model = CDANImputationModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)
    imputation_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        
        # Use the shorter loader to determine epoch length
        train_iterator = iter(zip(source_train_loader, target_train_loader))
        
        for i, (source_batch, target_batch) in enumerate(train_iterator):
            # Move data to device
            source_data = source_batch['observed_data'].to(device).to(torch.float32)
            print(f"source_data{source_data}")
            source_cond_mask = source_batch['cond_mask'].to(device)
            target_data = target_batch['observed_data'].to(device).to(torch.float32)
            target_cond_mask = target_batch['cond_mask'].to(device)
            print(f"target_data{target_data}")
            # print(f"Dtype of source_data: {source_data.dtype}")
            # print(f"Dtype of source_cond_mask: {source_cond_mask.dtype}")
            # print(f"Dtype of target_data: {target_data.dtype}")
            # print(f"Dtype of target_cond_mask: {target_cond_mask.dtype}")

            # Prepare inputs by masking
            source_input = source_data * source_cond_mask
            target_input = target_data * target_cond_mask
            source_input = source_input.to(torch.float32)
            target_input = target_input.to(torch.float32)

            optimizer.zero_grad()

            # --- Forward Pass ---
            # Source domain
            imputation_s, domain_s = model(source_input)
            
            # Target domain
            _, domain_t = model(target_input)

            # --- Loss Calculation ---
            # 1. Supervised Imputation Loss (on source domain only)
            # Loss is calculated on the values that were masked out for imputation
            imputation_target_mask = (1 - source_cond_mask)
            loss_impute = imputation_loss_fn(
                imputation_s * imputation_target_mask,
                source_data * imputation_target_mask
            )
            loss_impute = loss_impute.to(torch.float32)
            # 2. Domain Adversarial Loss
            # Source is domain 0, Target is domain 1
            loss_domain_s = domain_loss_fn(domain_s, torch.zeros_like(domain_s))
            loss_domain_t = domain_loss_fn(domain_t, torch.ones_like(domain_t))
            loss_domain = loss_domain_s + loss_domain_t

            print(f"Dtype of loss_impute: {loss_impute.dtype}")
            print(f"Dtype of loss_domain: {loss_domain.dtype}")
            print(f"Dtype of args.lambda_domain: {type(args.lambda_domain)}") # Check the type of the multiplier


            # 3. Total Loss
            total_loss = loss_impute + args.lambda_domain * loss_domain
            
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}], Impute Loss: {loss_impute.item():.4f}, Domain Loss: {loss_domain.item():.4f}")

        # --- Validation Loop ---
        model.eval()
        total_val_mae = 0
        total_val_mse = 0
        with torch.no_grad():
            for val_batch in target_valid_loader:
                val_data = val_batch['observed_data'].to(device).to(torch.float32)
                val_cond_mask = val_batch['cond_mask'].to(device)
                val_gt_mask = val_batch['gt_mask'].to(device)
                
                
                val_input = val_data * val_cond_mask
                imputation_val, _ = model(val_input)
                # Un-normalize for interpretable error
                imputation_val_un = imputation_val * scaler + mean_scaler
                val_data_un = val_data * scaler + mean_scaler
                # 🔧 人为 mask 掉 cond_mask 中的部分值作为 eval 区域
                random_mask = (torch.rand_like(val_cond_mask) > 0.9).float()  # 10% mask rate
                eval_mask = (val_gt_mask - val_cond_mask).bool()

                if eval_mask.sum() > 0:
                    total_val_mse += nn.functional.mse_loss(
                        imputation_val_un[eval_mask], val_data_un[eval_mask], reduction='sum'
                    ).item()
                    total_val_mae += nn.functional.l1_loss(
                        imputation_val_un[eval_mask], val_data_un[eval_mask], reduction='sum'
                    ).item()
                # Calculate error only on actually observed values that were masked for validation
                # 
                eval_mask = (val_gt_mask * (1 - val_cond_mask)).bool() # all eval_mask are equal to 0.
                print("val_gt_mask sum:", val_gt_mask.sum().item())
                print("val_cond_mask sum:", val_cond_mask.sum().item())
                print("eval_mask sum:", eval_mask.sum().item())
                
        avg_val_mae = total_val_mae / len(target_valid_loader.dataset)
        avg_val_mse = total_val_mse / len(target_valid_loader.dataset)
        print(f"--- Epoch {epoch+1} Validation on Target ---")
        print(f"MAE: {avg_val_mae:.4f}, MSE: {avg_val_mse:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDAN for Spatial-Temporal Imputation")
    
    # --- Paths and Data --
    # --- Model Hyperparameters ---
    parser.add_argument('--input_dim', type=int, default=20, help='Number of features in the time series')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the GRU')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., cuda:0, cpu)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_domain', type=float, default=1.0, help='Weight for the domain adversarial loss')

    args = parser.parse_args()
    main(args)
