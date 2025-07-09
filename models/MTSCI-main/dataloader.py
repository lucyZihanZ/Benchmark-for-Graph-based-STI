# import numpy as np
# import pickle as pk

# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from utils.utils import sample_mask


# def generate_val_test_dataloader(
#     dataset_path,
#     seq_len,
#     missing_ratio,
#     missing_pattern="point",
#     batch_size=4,
#     mode="val",
#     num_workers=0,
# ):
#     if mode == "val":
#         with open(dataset_path + "/val_set.pkl", "rb") as fb:
#             data = pk.load(fb)
#         # print("val data shape: ", data.shape)
#         val_SEED = 9101111
#         rng = np.random.default_rng(val_SEED)
#     elif mode == "test":
#         with open(dataset_path + "/test_set.pkl", "rb") as fb:
#             data = pk.load(fb)
#         # print("test data shape: ", data.shape)
#         test_SEED = 9101110
#         rng = np.random.default_rng(test_SEED)
#     else:
#         assert False, "mode must be val or test"

#     X_Tilde = data
#     gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)
#     if missing_pattern == "block":
#         # block missing
#         indicating_mask = sample_mask(
#             shape=data.shape,
#             p=0.0015,
#             p_noise=0.05,
#             min_seq=12,
#             max_seq=12 * 4,
#             rng=rng,
#         )
#     else:
#         # point missing
#         indicating_mask = sample_mask(
#             shape=data.shape,
#             p=0.0,
#             p_noise=missing_ratio,
#             min_seq=12,
#             max_seq=12 * 4,
#             rng=rng,
#         )
#     X = X_Tilde * (1 - indicating_mask)

#     mask = gt_mask * (1 - indicating_mask)

#     print(
#         mode
#         + ": original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
#             1 - np.sum(gt_mask) / gt_mask.size,
#             np.sum(indicating_mask) / indicating_mask.size,
#             missing_pattern,
#             1 - np.sum(mask) / mask.size,
#         )
#     )
#     X = np.nan_to_num(X)
#     X_Tilde = np.nan_to_num(X_Tilde)

#     sample_nums = data.shape[0] // seq_len
#     print(mode + " samples: {}".format(sample_nums))
#     input_X_list, input_mask_list, eval_mask, output_gt_list,output_gt_mask = [], [], [], [],[]
#     for i in range(sample_nums):
#         input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
#         input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
#         eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
#         output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
#         output_gt_mask.append(gt_mask[i * seq_len : (i + 1) * seq_len])

#     X_tensor = torch.from_numpy(np.array(input_X_list)).float()
#     mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
#     eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
#     X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
#     output_gt_mask_tensor = torch.from_numpy(np.array(output_gt_mask)).float()

#     tensor_dataset = TensorDataset(
#         X_tensor, mask_tensor, X_Tilde_tensor, output_gt_mask_tensor, eval_mask_tensor
#     )
#     dataloader = DataLoader(
#         tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#     )
#     return dataloader


# def generate_train_dataloader(
#     dataset_path, seq_len, missing_ratio, missing_pattern, batch_size=4, mode="train"
# ):
#     with open(dataset_path + "/train_set.pkl", "rb") as fb:
#         train_data = pk.load(fb)
#     # print("train data shape: ", train_data.shape)

#     train_SEED = 9101112
#     train_rng = np.random.default_rng(train_SEED)
    
#     X_Tilde = train_data
#     X_Tilde = X_Tilde['X'].astype(np.float32)
#     gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)

#     if missing_pattern == "block":
#         # block missing
#         indicating_mask = sample_mask(
#             shape=train_data.shape,
#             p=0.0015,
#             p_noise=missing_ratio,
#             min_seq=12,
#             max_seq=12 * 4,
#             rng=train_rng,
#         )
#     else:
#         # point missing
#         indicating_mask = sample_mask(
#             shape=train_data['X'].shape,
#             p=0.0,
#             p_noise=missing_ratio,
#             min_seq=12,
#             max_seq=12 * 4,
#             rng=train_rng,
#         )

#     X = X_Tilde * (1 - indicating_mask)

#     mask = gt_mask * (1 - indicating_mask)
#     print(
#         "Train: original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
#             1 - np.sum(gt_mask) / gt_mask.size,
#             np.sum(indicating_mask) / indicating_mask.size,
#             missing_pattern,
#             1 - np.sum(mask) / mask.size,
#         )
#     )

#     X = np.nan_to_num(X)
#     X_Tilde = np.nan_to_num(X_Tilde)

#     train_nums = train_data['X'].shape[0] // seq_len - 1
#     print("train samples: {}".format(train_nums))
#     (
#         input_X_list,
#         input_mask_list,
#         eval_mask,
#         output_gt_list,
#         output_gt_mask,
#         pred_gt_list,
#         pred_gt_mask,
#     ) = ([], [], [], [], [], [],[])
#     for i in range(train_nums):
#         input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
#         input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
#         eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
#         output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
#         output_gt_mask.append(gt_mask[i * seq_len : (i + 1) * seq_len])
#         pred_gt_list.append(X_Tilde[(i + 1) * seq_len : (i + 2) * seq_len])
#         pred_gt_mask.append(gt_mask[(i + 1) * seq_len : (i + 2) * seq_len])

#     X_tensor = torch.from_numpy(np.array(input_X_list)).float()
#     mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
#     eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
#     X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
#     output_gt_mask_tensor = torch.from_numpy(np.array(output_gt_mask)).float()
#     pred_gt_tensor = torch.from_numpy(np.array(pred_gt_list)).float()
#     pred_gt_mask_tensor = torch.from_numpy(np.array(pred_gt_mask)).float()

#     train_dataset = TensorDataset(
#         X_tensor,
#         mask_tensor,
#         eval_mask_tensor,
#         X_Tilde_tensor,
#         output_gt_mask_tensor,
#         pred_gt_tensor,
#         pred_gt_mask_tensor,
#     )
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     return train_dataloader


# if __name__ == "__main__":
#     dataset_path = "../datasets/ETT/"
#     seq_len = 24
#     miss_rate = 0.2
#     batch_size = 32
#     missing_pattern = "point"
#     train_loader = generate_train_dataloader(
#         dataset_path,
#         seq_len,
#         missing_ratio=miss_rate,
#         missing_pattern=missing_pattern,
#         batch_size=batch_size,
#     )
#     val_loader = generate_val_test_dataloader(
#         dataset_path,
#         seq_len,
#         missing_ratio=miss_rate,
#         missing_pattern=missing_pattern,
#         batch_size=batch_size,
#         mode="val",
#     )
#     test_loader = generate_val_test_dataloader(
#         dataset_path,
#         seq_len,
#         missing_ratio=miss_rate,
#         missing_pattern=missing_pattern,
#         batch_size=batch_size,
#         mode="test",
#     )
#     print("len train dataloader: ", len(train_loader))
#     print("len val dataloader: ", len(val_loader))
#     print("len test dataloader: ", len(test_loader))


import numpy as np
import pickle as pk

import torch
from torch.utils.data import DataLoader, TensorDataset
# Assuming utils.utils and sample_mask are correctly defined elsewhere
from utils.utils import sample_mask


def generate_val_test_dataloader(
    dataset_path,
    seq_len,
    missing_ratio,
    missing_pattern="point",
    batch_size=4,
    mode="val",
    num_workers=0,
):
    if mode == "val":
        with open(dataset_path + "/val_set.pkl", "rb") as fb:
            data_dict = pk.load(fb) # Load the dictionary
        # print("val data shape: ", data.shape)
        val_SEED = 9101111
        rng = np.random.default_rng(val_SEED)
    elif mode == "test":
        with open(dataset_path + "/test_set.pkl", "rb") as fb:
            data_dict = pk.load(fb) # Load the dictionary
        # print("test data shape: ", data.shape)
        test_SEED = 9101110
        rng = np.random.default_rng(test_SEED)
    else:
        assert False, "mode must be val or test"

    # Extract the actual data array from the loaded dictionary
    X_Tilde = data_dict['X'].astype(np.float32) 
    
    gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)
    
    # Use X_Tilde.shape instead of data.shape for sample_mask
    if missing_pattern == "block":
        # block missing
        indicating_mask = sample_mask(
            shape=X_Tilde.shape, # Corrected: use X_Tilde.shape
            p=0.0015,
            p_noise=0.005,
            min_seq=12,
            max_seq=12 * 4,
            rng=rng,
        )
    else:
        # point missing
        indicating_mask = sample_mask(
            shape=X_Tilde.shape, # Corrected: use X_Tilde.shape
            p=0.0,
            p_noise=missing_ratio,
            min_seq=12,
            max_seq=12 * 4,
            rng=rng,
        )
    X = X_Tilde * (1 - indicating_mask)

    mask = gt_mask * (1 - indicating_mask)

    print(
        mode
        + ": original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            1 - np.sum(gt_mask) / gt_mask.size,
            np.sum(indicating_mask) / indicating_mask.size,
            missing_pattern,
            1 - np.sum(mask) / mask.size,
        )
    )
    X = np.nan_to_num(X)
    X_Tilde = np.nan_to_num(X_Tilde)

    sample_nums = X_Tilde.shape[0] // seq_len # Use X_Tilde.shape[0]
    print(mode + " samples: {}".format(sample_nums))
    input_X_list, input_mask_list, eval_mask, output_gt_list,output_gt_mask = [], [], [], [],[]
    for i in range(sample_nums):
        input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
        input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
        eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
        output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
        output_gt_mask.append(gt_mask[i * seq_len : (i + 1) * seq_len])

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    output_gt_mask_tensor = torch.from_numpy(np.array(output_gt_mask)).float()

    tensor_dataset = TensorDataset(
        X_tensor, mask_tensor, X_Tilde_tensor, output_gt_mask_tensor, eval_mask_tensor
    )
    dataloader = DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


def generate_train_dataloader(
    dataset_path, seq_len, missing_ratio, missing_pattern, batch_size=4, mode="train"
):
    with open(dataset_path + "/train_set.pkl", "rb") as fb:
        train_data_dict = pk.load(fb) # Renamed to train_data_dict for clarity
    # print("train data shape: ", train_data.shape)

    train_SEED = 9101112
    train_rng = np.random.default_rng(train_SEED)
    
    # Extract the actual data array from the loaded dictionary
    X_Tilde = train_data_dict['X'].astype(np.float32)
    gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)

    # Use X_Tilde.shape instead of train_data.shape for sample_mask
    if missing_pattern == "block":
        # block missing
        indicating_mask = sample_mask(
            shape=X_Tilde.shape, # Corrected: use X_Tilde.shape
            p=0.0015,
            p_noise=missing_ratio,
            min_seq=12,
            max_seq=12 * 4,
            rng=train_rng,
        )
    else:
        # point missing
        indicating_mask = sample_mask(
            shape=X_Tilde.shape, # Corrected: use X_Tilde.shape
            p=0.0,
            p_noise=missing_ratio,
            min_seq=12,
            max_seq=12 * 4,
            rng=train_rng,
        )

    X = X_Tilde * (1 - indicating_mask)

    mask = gt_mask * (1 - indicating_mask)
    print(
        "Train: original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            1 - np.sum(gt_mask) / gt_mask.size,
            np.sum(indicating_mask) / indicating_mask.size,
            missing_pattern,
            1 - np.sum(mask) / mask.size,
        )
    )

    X = np.nan_to_num(X)
    X_Tilde = np.nan_to_num(X_Tilde)

    # Use X_Tilde.shape[0] for sample_nums calculation
    train_nums = X_Tilde.shape[0] // seq_len - 1 
    print("train samples: {}".format(train_nums))
    (
        input_X_list,
        input_mask_list,
        eval_mask,
        output_gt_list,
        output_gt_mask,
        pred_gt_list,
        pred_gt_mask,
    ) = ([], [], [], [], [], [],[])
    for i in range(train_nums):
        input_X_list.append(X[i * seq_len : (i + 1) * seq_len])
        input_mask_list.append(mask[i * seq_len : (i + 1) * seq_len])
        eval_mask.append(indicating_mask[i * seq_len : (i + 1) * seq_len])
        output_gt_list.append(X_Tilde[i * seq_len : (i + 1) * seq_len])
        output_gt_mask.append(gt_mask[i * seq_len : (i + 1) * seq_len])
        pred_gt_list.append(X_Tilde[(i + 1) * seq_len : (i + 2) * seq_len])
        pred_gt_mask.append(gt_mask[(i + 1) * seq_len : (i + 2) * seq_len])

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    output_gt_mask_tensor = torch.from_numpy(np.array(output_gt_mask)).float()
    pred_gt_tensor = torch.from_numpy(np.array(pred_gt_list)).float()
    pred_gt_mask_tensor = torch.from_numpy(np.array(pred_gt_mask)).float()

    train_dataset = TensorDataset(
        X_tensor,
        mask_tensor,
        eval_mask_tensor,
        X_Tilde_tensor,
        output_gt_mask_tensor,
        pred_gt_tensor,
        pred_gt_mask_tensor,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


if __name__ == "__main__":
    dataset_path = "../datasets/ETT/"
    seq_len = 20
    miss_rate = 0.2
    batch_size = 32
    missing_pattern = "point"
    train_loader = generate_train_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
    )
    val_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
