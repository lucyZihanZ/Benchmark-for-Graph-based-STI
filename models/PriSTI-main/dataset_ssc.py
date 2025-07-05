# import os
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import pandas as pd
# import pickle
# import torch

# def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
#     rand_for_mask = torch.rand_like(observed_mask) * observed_mask
#     rand_for_mask = rand_for_mask.reshape(-1)
#     sample_ratio = np.random.rand()
#     sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
#     num_observed = observed_mask.sum().item()
#     num_masked = round(num_observed * sample_ratio)
#     rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

#     cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
#     return cond_mask

# def get_block_mask(observed_mask, train_missing_pattern='block'):
#     rand_sensor_mask = torch.rand_like(observed_mask)
#     randint = np.random.randint
#     sample_ratio = np.random.rand()
#     sample_ratio = sample_ratio * 0.15
#     mask = rand_sensor_mask < sample_ratio
#     min_seq = 12
#     max_seq = 24
#     for col in range(observed_mask.shape[1]):
#         idxs = np.flatnonzero(mask[:, col])
#         if not len(idxs):
#             continue
#         fault_len = min_seq
#         if max_seq > min_seq:
#             fault_len = fault_len + int(randint(max_seq - min_seq))
#         idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
#         idxs = np.unique(idxs_ext)
#         idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
#         mask[idxs, col] = True
#     rand_base_mask = torch.rand_like(observed_mask) < 0.05
#     reverse_mask = mask | rand_base_mask
#     block_mask = 1 - reverse_mask.to(torch.float32)

#     cond_mask = observed_mask.clone()
#     mask_choice = np.random.rand()
#     cond_mask = block_mask * cond_mask

#     return cond_mask

# def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
#     if rng is None:
#         rand = np.random.random
#         randint = np.random.randint
#     else:
#         rand = rng.random
#         randint = rng.integers
#     mask = rand(shape) < p 
#     for col in range(mask.shape[1]):
#         idxs = np.flatnonzero(mask[:, col]) 
#         if not len(idxs): 
#             continue
#         fault_len = min_seq 
#         if max_seq > min_seq:
#             fault_len = fault_len + int(randint(max_seq - min_seq)) 
#         idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs]) 
#         idxs = np.unique(idxs_ext) 
#         idxs = np.clip(idxs, 0, shape[0] - 1)
#         mask[idxs, col] = True 
#     mask = mask | (rand(mask.shape) < p_noise) 
#     return mask.astype('uint8')


# class Pooled_Dataset(Dataset):
#     def __init__(self, seed=1, eval_length=16, mode="train", val_len=0.1, test_len=0.2, eval_missing_pattern='block',
#                 train_missing_pattern='point'):
#         """
#     train_missing_pattern: control the stochastic missingness
#     eval_missing_pattern: control the deterministic missingness.
#     Evaluation process: in the target domain without label, only for imputation
#     cond_mask: model input; unknown values
#     eval_mask = gt_mask - cond_mask; hide some values, and use the groundtruth to compare with the prediction.
#     eval: evaluation errors

#         """
#         self.eval_length = eval_length
#         self.train_missing_pattern = train_missing_pattern
#         self.mode = mode
#         self.use_index = []
#         self.cut_length = []

#         df = pd.read_csv('./data/ssc/SSC_pooled.csv', index_col=0) # 将第一列作为索引        
#         df.index = pd.to_datetime(df.index) # 将索引转换为日期时间格式
#         datetime_idx = sorted(df.index)
#         date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
#         df = df.reindex(index=date_range) 

#         start_date = '2015/4/15'
#         end_date = '2022/9/30'
        
#         for col in df.columns:
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col] = df[col].astype(np.float32)
#         #### log transformation #####
#         df = df.loc[start_date:end_date, :] # 特定时间范围的数据
#         ob_mask = ~np.isnan(df.values)
#         df.fillna(method='ffill', axis=0, inplace=True)
#         # change into the float32 format
#         # df = df.fillna(0.).values.astype(np.float32)


#         # SEED = 45678
#         SEED = seed
#         self.rng = np.random.default_rng(SEED)
#         if eval_missing_pattern == 'block':
#             eval_mask = sample_mask(shape=(2726, 20), p=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=self.rng)
#         elif eval_missing_pattern == 'point':
#             eval_mask = sample_mask(shape=(2726, 20), p=0.01, p_noise=0.25, max_seq=12, min_seq=12 * 4, rng=self.rng)
            
#         gt_mask = (1-(eval_mask | (1-ob_mask))).astype('uint8')
#         # gt_mask = (ob_mask & eval_mask).astype('uint8')


#         self.train_mean = np.zeros(20)
#         self.train_std = np.zeros(20)
#         for k in range(20):
#             tmp_data = df.iloc[:, k][ob_mask[:, k] == 1] # tmp_values 中选择所有行的第 k 列的特征数据，并且只选择掩码值为 1 的位置对应的数据，即只选择观察到的值。 c_data 就是第 k 列中观察到的值的数组。
#             self.train_mean[k] = tmp_data.mean()
#             self.train_std[k] = tmp_data.std()
#         path = "./data/ssc/pooled_meanstd.pk"
#         with open(path, "wb") as f:
#             pickle.dump([self.train_mean, self.train_std], f)

#         val_start = int((1 - val_len - test_len) * len(df))
#         test_start = int((1 - test_len) * len(df))
#         c_data = (
#              (df.fillna(0).values - self.train_mean) / self.train_std
#         ) * ob_mask
#         if mode == 'train':
#             self.observed_mask = ob_mask[:val_start]
#             self.gt_mask = gt_mask[:val_start]
#             self.observed_data = c_data[:val_start]
#         elif mode == 'valid':
#             self.observed_mask = ob_mask[val_start: test_start]
#             self.gt_mask = gt_mask[val_start: test_start]
#             self.observed_data = c_data[val_start: test_start]
#         elif mode == 'test':
#             self.observed_mask = ob_mask[test_start:]
#             self.gt_mask = gt_mask[test_start:]
#             self.observed_data = c_data[test_start:]
#         current_length = len(self.observed_mask) - eval_length + 1

#         if mode == "test":
#             n_sample = len(self.observed_data) // eval_length
#             c_index = np.arange(
#                 0, 0 + eval_length * n_sample, eval_length
#             )
#             self.use_index += c_index.tolist()
#             self.cut_length += [0] * len(c_index)
#             if len(self.observed_data) % eval_length != 0:
#                 self.use_index += [current_length - 1]
#                 self.cut_length += [eval_length - len(self.observed_data) % eval_length]
#         elif mode != "test":
#             self.use_index = np.arange(current_length)
#             self.cut_length = [0] * len(self.use_index)

#     def __getitem__(self, org_index):
#         """
#     Training part:
#     cond_mask: control the input of training process
#     target_mask: control which part we mask, and which part we need the model for imputation

#         """
#         index = self.use_index[org_index]
#         ob_data = self.observed_data[index: index + self.eval_length]
#         ob_mask = self.observed_mask[index: index + self.eval_length]
#         ob_mask_t = torch.tensor(ob_mask).float()
#         gt_mask = self.gt_mask[index: index + self.eval_length]
#         gt_mask = torch.tensor(gt_mask).float()
#         if self.mode != 'train':
#             # cond_mask = torch.tensor(gt_mask).to(torch.float32)
#             rand_mask = (torch.rand_like(gt_mask) > 0.1).float()  # 10% dropout
#             cond_mask = gt_mask * rand_mask
#             eval_mask = (gt_mask - cond_mask).clamp(min=0.0)
#         else:
#             if self.train_missing_pattern != 'point':
#                 cond_mask = get_block_mask(ob_mask_t, train_missing_pattern=self.train_missing_pattern)
#                 # eval_mask = (gt_mask - cond_mask).clamp(min=0.0)
#             else:
#                 cond_mask = get_randmask(ob_mask_t)
#             eval_mask = None # 不用于训练阶段
#         domain_indicator = torch.tensor([0])
#         s = {
#             "observed_data": ob_data,
#             "observed_mask": ob_mask,
#             "gt_mask": gt_mask,
#             "timepoints": np.arange(self.eval_length),
#             "cut_length": self.cut_length[org_index],
#             "cond_mask": cond_mask,
#             "domain_indicator": domain_indicator
#         }
#         if eval_mask is not None:
#             s["eval_mask"] = eval_mask

#         return s

#     def __len__(self):
#         return len(self.use_index)


# def get_dataloader_pooled(batch_size, device, seed=1, val_len=0.1, num_workers=4, 
#                    eval_missing_pattern='block', train_missing_pattern='block'):
#     dataset = Pooled_Dataset(seed=seed, mode="train", eval_missing_pattern=eval_missing_pattern,
#                              train_missing_pattern=train_missing_pattern)
#     train_loader = DataLoader(
#         dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
#     )
#     dataset_test = Pooled_Dataset(seed=seed, mode="test", eval_missing_pattern=eval_missing_pattern,
#                              train_missing_pattern=train_missing_pattern)
#     test_loader = DataLoader(
#         dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )
#     dataset_valid = Pooled_Dataset(seed=seed, mode="valid", val_len=val_len, eval_missing_pattern=eval_missing_pattern,
#                              train_missing_pattern=train_missing_pattern)
#     valid_loader = DataLoader(
#         dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )

#     scaler = torch.from_numpy(dataset.train_std).to(device).float()
#     mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

#     return train_loader, valid_loader, test_loader, scaler, mean_scaler

# # test the missingness:
# if __name__ == "__main__":
#     # Create a dummy data file for testing if it doesn't exist
#     # Make sure to adjust this path if your data is elsewhere
#     data_path = './data/ssc/SSC_pooled.csv'
#     if not os.path.exists(data_path):
#         print(f"Creating a dummy CSV at {data_path} for testing.")
#         os.makedirs(os.path.dirname(data_path), exist_ok=True)
#         dummy_data = {
#             'date': pd.date_range(start='2015-04-15', end='2022-09-30', freq='D'),
#             **{f'feature_{i}': np.random.rand(2726) * 100 for i in range(20)}
#         }
#         # Introduce some NaNs to simulate real data
#         for i in range(5):
#             dummy_data[f'feature_{i}'][np.random.choice(len(dummy_data['date']), 50, replace=False)] = np.nan
#         pd.DataFrame(dummy_data).set_index('date').to_csv(data_path)
#         print("Dummy CSV created.")
#     else:
#         print(f"Using existing data file at {data_path}")

#     # --- Test get_randmask ---
#     print("\n--- Testing get_randmask ---")
#     dummy_observed_mask = torch.ones(16, 20).bool() # Fully observed
#     dummy_observed_mask[0, 0] = False # Introduce a natural NaN
#     rand_cond_mask = get_randmask(dummy_observed_mask.float())
#     print(f"rand_cond_mask shape: {rand_cond_mask.shape}")
#     print(f"rand_cond_mask sum (1s): {rand_cond_mask.sum().item()}")
#     # Expected: sum < numel (16*20) due to random masking
#     print(f"rand_cond_mask sample (first few rows): \n{rand_cond_mask[:3, :5]}\n")

#     # --- Test get_block_mask ---
#     print("\n--- Testing get_block_mask ---")
#     block_cond_mask = get_block_mask(dummy_observed_mask.float())
#     print(f"block_cond_mask shape: {block_cond_mask.shape}")
#     print(f"block_cond_mask sum (1s): {block_cond_mask.sum().item()}")
#     # Expected: sum < numel due to block masking
#     print(f"block_cond_mask sample (first few rows): \n{block_cond_mask[:3, :5]}\n")

#     # --- Test sample_mask (used internally by Pooled_Dataset init) ---
#     print("\n--- Testing sample_mask (internal to Pooled_Dataset init) ---")
#     rng_test = np.random.default_rng(123)
#     sample_eval_mask = sample_mask(shape=(10, 5), p=0.1, min_seq=2, max_seq=3, rng=rng_test)
#     print(f"sample_eval_mask shape: {sample_eval_mask.shape}")
#     print(f"sample_eval_mask sum (1s, uint8): {np.sum(sample_eval_mask)}") # Sum of 1s in uint8 array
#     print(f"sample_eval_mask sample: \n{sample_eval_mask}\n")
#     # Expected: sum > 0, some True values

#     # --- Test Pooled_Dataset & DataLoader ---
#     print("\n--- Testing Pooled_Dataset and DataLoader ---")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Test Training DataLoader
#     print("\n--- Train DataLoader Sample ---")
#     train_loader, _, _, _, _ = get_dataloader_pooled(
#         batch_size=4, device=device, eval_missing_pattern='block', train_missing_pattern='block' # Use 'block' for train for demonstration
#     )
#     first_train_batch = next(iter(train_loader))
#     print(f"Train batch 'observed_data' shape: {first_train_batch['observed_data'].shape}")
#     print(f"Train batch 'observed_data' dtype: {first_train_batch['observed_data'].dtype}")
#     print(f"Train batch 'gt_mask' shape: {first_train_batch['gt_mask'].shape}")
#     print(f"Train batch 'gt_mask' sum (1s): {first_train_batch['gt_mask'].sum().item()}")
#     print(f"Train batch 'gt_mask' sample (first sample, first few features): \n{first_train_batch['gt_mask'][0, :3, :5]}\n")
#     print(f"Train batch 'cond_mask' shape: {first_train_batch['cond_mask'].shape}")
#     print(f"Train batch 'cond_mask' sum (1s): {first_train_batch['cond_mask'].sum().item()}")
#     print(f"Train batch 'cond_mask' sample (first sample, first few features): \n{first_train_batch['cond_mask'][0, :3, :5]}\n")
#     # Expected: cond_mask sum should be less than gt_mask sum for training due to introduced missingness.
#     # gt_mask should contain 1s where original data was observed.

#     # Test Validation DataLoader
#     print("\n--- Validation DataLoader Sample ---")
#     _, valid_loader, _, scaler, mean_scaler = get_dataloader_pooled(
#         batch_size=4, device=device,eval_missing_pattern='block', train_missing_pattern='block'
#     )
#     first_valid_batch = next(iter(valid_loader))
#     val_data = first_valid_batch['observed_data'].to(device).to(torch.float32)
#     val_cond_mask = first_valid_batch['cond_mask'].to(device)
#     val_gt_mask = first_valid_batch['gt_mask'].to(device).to(torch.bool)

#     print(f"Valid batch 'observed_data' shape: {val_data.shape}")
#     print(f"Valid batch 'observed_data' dtype: {val_data.dtype}")
#     print(f"Valid batch 'gt_mask' shape: {val_gt_mask.shape}")
#     print(f"Valid batch 'gt_mask' sum (1s): {val_gt_mask.sum().item()}")
#     print(f"Valid batch 'gt_mask' sample (first sample, first few features): \n{val_gt_mask[0, :3, :5]}\n")
#     print(f"Valid batch 'cond_mask' shape: {val_cond_mask.shape}")
#     print(f"Valid batch 'cond_mask' sum (1s): {val_cond_mask.sum().item()}")
#     print(f"Valid batch 'cond_mask' sample (first sample, first few features): \n{val_cond_mask[0, :3, :5]}\n")

#     # CRITICAL TEST: eval_mask in validation
#     eval_mask = ((1 - val_cond_mask).bool() & val_gt_mask).bool()
#     print(f"\n--- Validation eval_mask Check ---")
#     print(f"eval_mask shape: {eval_mask.shape}")
#     print(f"eval_mask sum (True): {eval_mask.sum().item()}")
#     print(f"eval_mask sample (first sample, first few features): \n{eval_mask[0, :, :]}\n")
#     print(f"eval_mask True sum {sum(eval_mask == 1)}")

#     if eval_mask.sum().item() > 0:
#         print("SUCCESS: eval_mask has True values! Evaluation should now work correctly.")
#         # Optional: Test a dummy loss calculation
#         dummy_imputation = torch.randn_like(val_data)
#         masked_imputation = dummy_imputation[eval_mask]
#         masked_gt = val_data[eval_mask]
#         print(f"Dummy masked_imputation shape: {masked_imputation.shape}")
#         print(f"Dummy masked_gt shape: {masked_gt.shape}")
#         dummy_mae = torch.nn.functional.l1_loss(masked_imputation, masked_gt).item()
#         dummy_mse = torch.nn.functional.mse_loss(masked_imputation, masked_gt).item()
#         print(f"Dummy MAE on eval_mask: {dummy_mae:.4f}")
#         print(f"Dummy MSE on eval_mask: {dummy_mse:.4f}")
#     else:
#         print("FAILURE: eval_mask is still all False. Re-check mask generation logic in Pooled_Dataset.__init__ for eval_missing_pattern.")
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torchcde # Import torchcde for continuous-time interpolation
from torch.utils.data import DataLoader, Dataset

# Ensure HDF5 file locking is disabled if you're using HDF5 files elsewhere
# Although this specific script uses CSV, it's good practice if other parts of your
# project might use HDF5.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    """
    Generates a random mask for observed data, introducing missingness.

    Args:
        observed_mask (torch.Tensor): A tensor indicating observed (1) and unobserved (0) values.
        min_miss_ratio (float): Minimum ratio of observed values to mask.
        max_miss_ratio (float): Maximum ratio of observed values to mask.

    Returns:
        torch.Tensor: A conditional mask where 0 indicates a masked (hidden) value.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio - min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    num_masked = min(num_masked, num_observed)
    
    # Use topk to find indices of values to mask
    if num_masked > 0:
        # Get indices of the largest 'num_masked' random values within the observed points
        # These will be set to -1 to indicate they should be masked
        _, indices_to_mask = rand_for_mask.topk(num_masked)
        rand_for_mask[indices_to_mask] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def get_block_mask(observed_mask, target_strategy='block'):
    """
    Generates a block-wise missing mask for observed data.

    Args:
        observed_mask (torch.Tensor): A tensor indicating observed (1) and unobserved (0) values.
        train_missing_pattern (str): Specifies the type of missingness (e.g., 'block').

    Returns:
        torch.Tensor: A conditional mask where 0 indicates a masked (hidden) value.
    """
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15 # Controls the density of initial random points for blocks
    mask = rand_sensor_mask < sample_ratio
    min_seq = 12 # Minimum length of a block of missing data
    max_seq = 24 # Maximum length of a block of missing data

    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    
    # Add some random noise (point-wise missingness)
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    # Apply the generated block mask to the observed mask
    cond_mask = block_mask * cond_mask

    return cond_mask

def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    """
    Generates a mask with contiguous blocks of missing data and random noise.
    Used for evaluation missing patterns.

    Args:
        shape (tuple): The shape of the mask to generate (time_steps, features).
        p (float): Probability of a point being the start of a missing block.
        p_noise (float): Probability of a random point being missing (noise).
        max_seq (int): Maximum length of a missing block.
        min_seq (int): Minimum length of a missing block.
        rng (numpy.random.Generator): Random number generator for reproducibility.

    Returns:
        numpy.ndarray: A uint8 mask where 1 indicates missing, 0 indicates observed.
    """
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers

    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


class Pooled_Dataset(Dataset):
    def __init__(self, seed=1, eval_length=16, mode="train", val_len=0.1, test_len=0.2,
                 missing_pattern='block', target_strategy = 'point',
                 is_interpolate=False): # Added is_interpolate
        """
        Initializes the Pooled_Dataset.

        Args:
            seed (int): Random seed for reproducibility.
            eval_length (int): Length of each evaluation sequence.
            mode (str): Dataset mode ('train', 'valid', 'test').
            val_len (float): Proportion of data for validation.
            test_len (float): Proportion of data for testing.
            eval_missing_pattern (str): Missingness pattern for evaluation (e.g., 'block', 'point').
            train_missing_pattern (str): Missingness pattern for training (e.g., 'block', 'point').
            is_interpolate (bool): If True, compute CDE coefficients for interpolation.
        """
        self.eval_length = eval_length
        self.target_strategy = target_strategy 
        self.mode = mode
        self.use_index = []
        self.cut_length = []
        self.is_interpolate = is_interpolate # Store the interpolation flag
        self.missing_pattern = missing_pattern

        # --- CSV File Loading ---
        # The code already uses pd.read_csv, which is suitable for CSV files.
        # Ensure your CSV file is located at './data/ssc/SSC_pooled.csv'.
        # If your CSV uses a different delimiter (e.g., tab-separated), you can add `sep='\t'`.
        # If your CSV does not have a header row, you can add `header=None`.
        # `index_col=0` means the first column will be used as the DataFrame index (e.g., dates).
        df = pd.read_csv('./data/ssc/SSC_pooled.csv', index_col=0) # Set first column as index
        df.index = pd.to_datetime(df.index) # Convert index to datetime format
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) # Reindex to fill missing dates with NaNs

        start_date = '2015/4/15'
        end_date = '2022/9/30'

        # Ensure numeric types and handle NaNs before normalization
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float32)

        df = df.loc[start_date:end_date, :] # Filter data for specific time range

        # Create observed mask based on NaNs in the filtered DataFrame
        ob_mask = ~np.isnan(df.values)

        # Fill NaNs for the purpose of normalization and subsequent processing
        # This will fill NaNs that were introduced by reindexing or originally present.
        # Use ffill (forward fill) as per original code, but ensure it's on the DataFrame
        # before converting to values for normalization.
        df.fillna(method='ffill', axis=0, inplace=True)
        # If there are still NaNs at the beginning after ffill, fill them with 0 or a sensible value
        df.fillna(0, inplace=True) # Fallback for leading NaNs if ffill doesn't cover them

        # SEED = 45678
        SEED = seed
        self.rng = np.random.default_rng(SEED)

        # Determine the shape for mask generation based on the actual data dimensions
        data_shape = df.shape
        # Assuming the number of features is the second dimension of df.values
        num_features = data_shape[1]
        num_time_steps = data_shape[0]

        if missing_pattern== 'block':
            eval_mask = sample_mask(shape=(num_time_steps, num_features), p=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=self.rng)
        elif missing_pattern == 'point':
            eval_mask = sample_mask(shape=(num_time_steps, num_features), p=0.01, p_noise=0.25, max_seq=12, min_seq=12 * 4, rng=self.rng)
        
        # gt_mask: Ground truth mask. 1 where data is observed AND NOT hidden by eval_mask.
        # This means, values that are originally observed AND are NOT part of the evaluation missingness.
        gt_mask = (ob_mask & (1 - eval_mask)).astype('uint8') # Corrected logic: 1 where observed and NOT eval_mask

        # Calculate mean and std using only observed values from the original data
        # This is done before applying any normalization or filling for the dataset itself.
        # We use the full df here to get the true mean/std for normalization.
        self.train_mean = np.zeros(num_features)
        self.train_std = np.zeros(num_features)
        for k in range(num_features):
            # Only calculate mean/std on originally observed values for each feature
            tmp_data = df.iloc[:, k][ob_mask[:, k] == 1]
            if len(tmp_data) > 0: # Avoid division by zero if a column is entirely missing
                self.train_mean[k] = tmp_data.mean()
                self.train_std[k] = tmp_data.std()
                if self.train_std[k] == 0: # Handle constant features
                    self.train_std[k] = 1e-5
            else: # If a column is entirely missing, set mean/std to 0/1e-5
                self.train_mean[k] = 0.0
                self.train_std[k] = 1e-5

        # Save mean and std for later use
        path = "./data/ssc/pooled_meanstd.pk"
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
        with open(path, "wb") as f:
            pickle.dump([self.train_mean, self.train_std], f)

        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))

        # Normalize the data using the calculated mean and std
        # Apply normalization to the *filled* DataFrame values, then mask with original ob_mask
        c_data = (
             (df.values - self.train_mean) / self.train_std
        ) * ob_mask # Apply original observation mask after normalization

        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = ob_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
        elif mode == 'test':
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = c_data[test_start:]
        current_length = len(self.observed_mask) - eval_length + 1

        if mode == "test":
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(
                0, 0 + eval_length * n_sample, eval_length
            )
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [eval_length - len(self.observed_data) % eval_length]
        elif mode != "test":
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        """
        Retrieves a single sequence from the dataset.

        Args:
            org_index (int): Original index into the use_index list.

        Returns:
            dict: A dictionary containing observed data, masks, timepoints, etc.
        """
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]
        gt_mask = torch.tensor(gt_mask).float()

        if self.mode != 'train':
            # For validation/test, cond_mask is derived from gt_mask with additional dropout
            rand_mask = (torch.rand_like(gt_mask) > 0.1).float()  # 10% dropout
            cond_mask = gt_mask * rand_mask
            eval_mask = (gt_mask - cond_mask).clamp(min=0.0)
        else:
            # For training, cond_mask is generated based on train_missing_pattern
            if self.target_strategy != 'point':
                cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy)
            else:
                cond_mask = get_randmask(ob_mask_t)
            eval_mask = None # Not used directly in training loss calculation in this setup

        domain_indicator = torch.tensor([0]) # Placeholder for domain indicator

        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask,
            "domain_indicator": domain_indicator
        }

        if eval_mask is not None:
            s["eval_mask"] = eval_mask

        # --- NEW: Interpolation coefficients for CDE models ---
        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64) # Ensure float64 for torchcde
            
            # Apply cond_mask to introduce NaNs where data is missing for interpolation
            # The values where cond_mask is 0 become NaN
            itp_data_masked = torch.where(cond_mask == 0, float('nan'), tmp_data)
            
            # Permute to (features, time_steps, 1) as required by torchcde.linear_interpolation_coeffs
            # The unsqueeze(-1) adds the channel dimension (1)
            itp_data_permuted = itp_data_masked.permute(1, 0).unsqueeze(-1)

            # Compute CDE coefficients
            # torchcde.linear_interpolation_coeffs handles NaNs by skipping them
            coeffs = torchcde.linear_interpolation_coeffs(itp_data_permuted)
            
            # Squeeze the last dimension and permute back to (time_steps, features)
            # Convert to float32 and numpy array
            s["coeffs"] = coeffs.squeeze(-1).permute(1, 0).to(torch.float32).numpy()

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader_pooled(batch_size, device, seed=1, val_len=0.1, num_workers=4,
                   missing_pattern='block', is_interpolate=False, target_strategy='block'): # Added is_interpolate
    """
    Creates and returns DataLoader instances for training, validation, and testing.

    Args:
        batch_size (int): Batch size for DataLoaders.
        device (torch.device): Device to move tensors to (e.g., 'cuda' or 'cpu').
        seed (int): Random seed for dataset initialization.
        val_len (float): Proportion of data for validation.
        num_workers (int): Number of worker processes for data loading.
        eval_missing_pattern (str): Missingness pattern for evaluation.
        train_missing_pattern (str): Missingness pattern for training.
        is_interpolate (bool): If True, enable CDE coefficient calculation in dataset.

    Returns:
        tuple: (train_loader, valid_loader, test_loader, scaler, mean_scaler)
    """
    dataset = Pooled_Dataset(seed=seed, mode="train", missing_pattern=missing_pattern,
                             is_interpolate=is_interpolate, target_strategy = target_strategy) # Pass is_interpolate
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = Pooled_Dataset(seed=seed, mode="test", missing_pattern=missing_pattern,
                             target_strategy=target_strategy,
                             is_interpolate=is_interpolate) # Pass is_interpolate
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = Pooled_Dataset(seed=seed, mode="valid", val_len=val_len, missing_pattern=missing_pattern,
                             target_strategy=target_strategy,
                             is_interpolate=is_interpolate) # Pass is_interpolate
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler

# test the missingness and interpolation:
if __name__ == "__main__":
    # Create a dummy data file for testing if it doesn't exist
    data_path = './data/ssc/SSC_pooled.csv'
    if not os.path.exists(data_path):
        print(f"Creating a dummy CSV at {data_path} for testing.")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        dummy_data = {
            'date': pd.date_range(start='2015-04-15', end='2022-09-30', freq='D'),
            **{f'feature_{i}': np.random.rand(2726) * 100 for i in range(20)}
        }
        # Introduce some NaNs to simulate real data
        for i in range(5):
            dummy_data[f'feature_{i}'][np.random.choice(len(dummy_data['date']), 50, replace=False)] = np.nan
        pd.DataFrame(dummy_data).set_index('date').to_csv(data_path)
        print("Dummy CSV created.")
    else:
        print(f"Using existing data file at {data_path}")

    # --- Test get_randmask ---
    print("\n--- Testing get_randmask ---")
    # Create a mask with some actual observed points (1s)
    dummy_observed_mask = torch.ones(16, 20).bool()
    # Simulate some original missingness (0s)
    dummy_observed_mask[0, 0] = False
    dummy_observed_mask[5, 10] = False
    
    rand_cond_mask = get_randmask(dummy_observed_mask.float())
    print(f"rand_cond_mask shape: {rand_cond_mask.shape}")
    print(f"rand_cond_mask sum (1s): {rand_cond_mask.sum().item()}")
    print(f"rand_cond_mask sample (first few rows): \n{rand_cond_mask[:3, :5]}\n")

    # --- Test get_block_mask ---
    print("\n--- Testing get_block_mask ---")
    block_cond_mask = get_block_mask(dummy_observed_mask.float())
    print(f"block_cond_mask shape: {block_cond_mask.shape}")
    print(f"block_cond_mask sum (1s): {block_cond_mask.sum().item()}")
    print(f"block_cond_mask sample (first few rows): \n{block_cond_mask[:3, :5]}\n")

    # --- Test sample_mask (used internally by Pooled_Dataset init) ---
    print("\n--- Testing sample_mask (internal to Pooled_Dataset init) ---")
    rng_test = np.random.default_rng(123)
    sample_eval_mask = sample_mask(shape=(10, 5), p=0.1, min_seq=2, max_seq=3, rng=rng_test)
    print(f"sample_eval_mask shape: {sample_eval_mask.shape}")
    print(f"sample_eval_mask sum (1s, uint8): {np.sum(sample_eval_mask)}")
    print(f"sample_eval_mask sample: \n{sample_eval_mask}\n")

    # --- Test Pooled_Dataset & DataLoader ---
    print("\n--- Testing Pooled_Dataset and DataLoader ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test Training DataLoader with interpolation
    print("\n--- Train DataLoader Sample (with interpolation) ---")
    train_loader, _, _, _, _ = get_dataloader_pooled(
        batch_size=4, device=device, eval_missing_pattern='block',
        train_missing_pattern='block', is_interpolate=True # Enable interpolation
    )
    first_train_batch = next(iter(train_loader))
    print(f"Train batch 'observed_data' shape: {first_train_batch['observed_data'].shape}")
    print(f"Train batch 'gt_mask' shape: {first_train_batch['gt_mask'].shape}")
    print(f"Train batch 'cond_mask' shape: {first_train_batch['cond_mask'].shape}")
    
    # Check for 'coeffs' in the batch
    if 'coeffs' in first_train_batch:
        print(f"Train batch 'coeffs' shape: {first_train_batch['coeffs'].shape}")
        print(f"Train batch 'coeffs' sample (first sample, first few features): \n{first_train_batch['coeffs'][0, :3, :5]}\n")
    else:
        print("ERROR: 'coeffs' not found in train batch when is_interpolate is True.")

    # Test Validation DataLoader with interpolation
    print("\n--- Validation DataLoader Sample (with interpolation) ---")
    _, valid_loader, _, scaler, mean_scaler = get_dataloader_pooled(
        batch_size=4, device=device, eval_missing_pattern='block',
        train_missing_pattern='block', is_interpolate=True # Enable interpolation
    )
    first_valid_batch = next(iter(valid_loader))
    val_data = first_valid_batch['observed_data'].to(device).to(torch.float32)
    val_cond_mask = first_valid_batch['cond_mask'].to(device)
    val_gt_mask = first_valid_batch['gt_mask'].to(device).to(torch.bool)

    print(f"Valid batch 'observed_data' shape: {val_data.shape}")
    print(f"Valid batch 'gt_mask' shape: {val_gt_mask.shape}")
    print(f"Valid batch 'cond_mask' shape: {val_cond_mask.shape}")

    # Check for 'coeffs' in the batch
    if 'coeffs' in first_valid_batch:
        print(f"Valid batch 'coeffs' shape: {first_valid_batch['coeffs'].shape}")
        print(f"Valid batch 'coeffs' sample (first sample, first few features): \n{first_valid_batch['coeffs'][0, :3, :5]}\n")
    else:
        print("ERROR: 'coeffs' not found in valid batch when is_interpolate is True.")

    # CRITICAL TEST: eval_mask in validation
    eval_mask = ((1 - val_cond_mask).bool() & val_gt_mask).bool()
    print(f"\n--- Validation eval_mask Check ---")
    print(f"eval_mask shape: {eval_mask.shape}")
    print(f"eval_mask sum (True): {eval_mask.sum().item()}")
    print(f"eval_mask sample (first sample, first few features): \n{eval_mask[0, :, :]}\n")

    if eval_mask.sum().item() > 0:
        print("SUCCESS: eval_mask has True values! Evaluation should now work correctly.")
        # Optional: Test a dummy loss calculation
        dummy_imputation = torch.randn_like(val_data)
        masked_imputation = dummy_imputation[eval_mask]
        masked_gt = val_data[eval_mask]
        print(f"Dummy masked_imputation shape: {masked_imputation.shape}")
        print(f"Dummy masked_gt shape: {masked_gt.shape}")
        dummy_mae = torch.nn.functional.l1_loss(masked_imputation, masked_gt).item()
        dummy_mse = torch.nn.functional.mse_loss(masked_imputation, masked_gt).item()
        print(f"Dummy MAE on eval_mask: {dummy_mae:.4f}")
        print(f"Dummy MSE on eval_mask: {dummy_mse:.4f}")
    else:
        print("FAILURE: eval_mask is still all False. Re-check mask generation logic in Pooled_Dataset.__init__ for eval_missing_pattern.")
