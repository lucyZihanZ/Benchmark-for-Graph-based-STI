import argparse
import json
import yaml
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb


# sys.path.append("../dataloader")
# sys.path.append("../models")
# sys.path.append("../utils")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from dataloader import *
from models import diff_block, model

from utils import utils, timefeatures
# from utils.timefeatures import *


def train(
    model,
    config,
    args,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    current_time=None,
):
    # wandb.init(
    #     project="MTSCI",
    #     name="{}_{}".format(args.dataset, current_time),
    #     config=args,
    # )

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model_{}.pth".format(current_time)

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    alpha = config["alpha"]
    beta = config["beta"]
    for epoch_no in range(config["epochs"]):
        avg_loss, avg_loss_noise, avg_loss_cons = 0.0, 0.0, 0.0
        model.train()
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss_noise, loss_cons = model(train_batch)
            loss = alpha * loss_noise + beta * loss_cons  # compute total loss
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_loss_noise += loss_noise.item()
            avg_loss_cons += loss_cons.item()
        lr_scheduler.step()
        train_loss = avg_loss / (batch_no +1)
        train_loss_noise = avg_loss_noise / (batch_no + 1)
        loss_cl = avg_loss_cons / (batch_no + 1)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                valid_loss = avg_loss_valid / (batch_no + 1)
                print(
                    "Epoch {}: train loss = {} train_loss_noise = {} loss_cl = {} valid loss = {}".format(
                        epoch_no + 1,
                        train_loss,
                        train_loss_noise,
                        loss_cl,
                        valid_loss,
                    )
                )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / (batch_no + 1),
                    "at",
                    epoch_no,
                )
                torch.save(model.state_dict(), output_path)
            # wandb.log(
            #     {
            #         "train_loss": train_loss,
            #         "train_loss_noise": train_loss_noise,
            #         "loss_cl": loss_cl,
            #         "valid_loss": valid_loss,
            #     }
            # )
        else:
            # wandb.log(
            #     {
            #         "train_loss": train_loss,
            #         "train_loss_noise": train_loss_noise,
            #         "loss_cl": loss_cl,
            #     }
            # )
            print(
                "Epoch {}: train loss = {} train_loss_noise = {} loss_cl = {}".format(
                    epoch_no + 1, train_loss, train_loss_noise, loss_cl
                )
            )


def evaluate(
    model,
    test_loader,
    nsample,
    scaler,
    mean_scaler,
    save_result_path,
    current_time=None,
):

    with torch.no_grad():
        model.eval()

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        imputed_data = []
        groundtruth = []
        eval_mask = []
        results = {}
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = (
                    output  # imputed results
                )
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)  # (B,L,K)
                observed_points = observed_points.permute(0, 2, 1)  # (B,L,K)

                samples_median = samples.median(
                    dim=1
                )  # use median as prediction to calculate the RMSE and MAE, include the median values and the indices

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                output = samples_median.values * scaler + mean_scaler
                X_Tilde = c_target * scaler + mean_scaler
                eval_M = eval_points
                imputed_data.append(output.cpu().numpy())
                groundtruth.append(X_Tilde.cpu().numpy())
                eval_mask.append(eval_M.cpu().numpy())

            results["imputed_data"] = np.concatenate(imputed_data, axis=0)
            results["groundtruth"] = np.concatenate(groundtruth, axis=0)
            results["eval_mask"] = np.concatenate(eval_mask, axis=0)

            mae, rmse, mape, mse, r2 = utils.missed_eval_np(
                results["imputed_data"],
                results["groundtruth"],
                1 - results["eval_mask"],
            )

            all_target = torch.cat(all_target, dim=0)  # (B,L,K)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)  # (B,L,K)
            all_observed_point = torch.cat(all_observed_point, dim=0)  # (B,L,K)
            all_observed_time = torch.cat(all_observed_time, dim=0)  # (B,L)
            all_generated_samples = torch.cat(
                all_generated_samples, dim=0
            )  # (B,nsample,L,K)

            CRPS = utils.calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            print(
                "mae = {:.3f}, rmse = {:.3f}, mape = {:.3f}%, mse = {:.3f}, r2 = {:.3f}, CRPS = {:.4f}".format(
                    mae, rmse, mape * 100, mse, r2, CRPS
                )
            )
            np.save(save_result_path + "/result_{}.npy".format(current_time), results)


def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)

    utils.seed_torch(args.seed)
    path = "../config/{}_{}.yaml".format(args.dataset, args.missing_pattern)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print(json.dumps(config, indent=4))

    # load args
    dataset = args.dataset
    dataset_path = args.dataset_path
    seq_len = args.seq_len
    miss_rate = args.missing_ratio
    val_miss_rate, test_miss_rate = args.val_missing_ratio, args.test_missing_ratio
    missing_pattern = args.missing_pattern
    batch_size = config["train"]["batch_size"]

    saving_path = args.saving_path + "/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    save_result_path = args.save_result_path + "/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)

    # load data
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
        missing_ratio=val_miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=test_miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
    # with open(dataset_path + "/scaler.pkl", "rb") as fb:
    #     mean, std = pk.load(fb)
    # print(mean)
    # print(std)
    # mean = np.array(mean).astype(np.float32)
    # std = np.array(std).astype(np.float32)
    try:
        with open(dataset_path + "/scaler.pkl", "rb") as fb:
            loaded_dict = pk.load(fb)

        # --- END OF ADDED/MOVED PRINT STATEMENTS ---

        # Attempt to convert to NumPy array.
        # This will work if loaded_mean/std are floats, lists, or already NumPy arrays.
        # It will fail if they are literally the string 'mean' or 'std'.
        mean = loaded_dict['mean']
        std = loaded_dict['std']
        # mean = mean.astype(np.float32)
        # std = std.astype(np.float32)

        # # Now, convert the NumPy arrays to PyTorch tensors
        # mean = torch.from_numpy(mean).to(args.device)
        # std = torch.from_numpy(std).to(args.device)
        if isinstance(mean, np.ndarray):
            mean = mean.astype(np.float32)
            mean = torch.from_numpy(mean).to(args.device)
        else:
            mean = mean.to(torch.float32).to(args.device)

        if isinstance(std, np.ndarray):
            std = std.astype(np.float32)
            std = torch.from_numpy(std).to(args.device)
        else:
            std = std.to(torch.float32).to(args.device)

    except ValueError as e:
        # This specific error indicates that loaded_mean/std are not numerical
        # and cannot be converted to float (e.g., they are the string 'mean').
        print(f"Error loading scaler.pkl: {e}")
        print("It seems 'mean' or 'std' loaded from scaler.pkl are not numerical values.")
        print("Please ensure your scaler.pkl file contains actual numerical mean and std values,")
        print("not the string labels 'mean' and 'std'.")
        print("You might need to regenerate your scaler.pkl file with correct numerical data.")
        # Exit or handle the error appropriately, as the program cannot proceed without scalers
        raise # Re-raise the exception to stop execution
        

    # mean = torch.from_numpy(mean).to(args.device)
    # std = torch.from_numpy(std).to(args.device)

    mtsci_model = model.MTSCI(
        config, args.device, target_dim=args.feature, seq_len=args.seq_len
    ).to(args.device)

    if args.scratch:
        train(
            mtsci_model,
            config["train"],
            args,
            train_loader,
            valid_loader=val_loader,
            foldername=saving_path,
            current_time=current_time,
        )
        print("load model from", saving_path)
        mtsci_model.load_state_dict(
            torch.load(saving_path + "/model_{}.pth".format(current_time))
        )
    else:
        print("load model from", args.checkpoint_path)
        mtsci_model.load_state_dict(torch.load(args.checkpoint_path))

    evaluate(
        mtsci_model,
        test_loader,
        nsample=args.nsample,
        scaler=std,
        mean_scaler=mean,
        save_result_path=save_result_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTSCI")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="ETT", type=str, help="dataset name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../datasets/SSC/ssc/ssc_pooled/",
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default="../results/",
        help="the save path of imputed data",
    )
    parser.add_argument(
        "--saving_path", type=str, default="../saved_models", help="saving model pth"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../saved_models/test/ssc/point/0.2/model.pth",
    )
    parser.add_argument("--seq_len", type=int, default=24, help="sequence length")
    parser.add_argument("--feature", help="feature nums", type=int, default=20)
    parser.add_argument(
        "--missing_pattern",
        type=str,
        default="block",
        help="missing pattern on training set",
    )
    parser.add_argument(
        "--missing_ratio", type=float, default=0.2, help="missing ratio on training set"
    )
    parser.add_argument(
        "--val_missing_ratio",
        type=float,
        default=0.2,
        help="missing ratio on validation set",
    )
    parser.add_argument(
        "--test_missing_ratio",
        type=float,
        default=0.2,
        help="missing ratio on testing set",
    )
    parser.add_argument("--scratch", action="store_true", help="test or scratch")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(args)

    start_time = time.time()
    main(args)
    print("Spend Time: ", time.time() - start_time)
