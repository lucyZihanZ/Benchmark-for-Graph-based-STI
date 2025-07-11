
import A_dataset
from Models import GSLI
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn


def model_train(configs):
    train_loader, __ = A_dataset.get_dataset(configs)

    model = GSLI.GSLIModel(configs).to(configs.device)

    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch)
    p2 = int(0.9 * configs.epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )

    metirc = nn.MSELoss(reduction="mean")

    model.train()
    for epoch in range(configs.epoch):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (observed_data, observed_tp, observed_mask, gt_mask) in enumerate(train_loader):
            observed_data = observed_data.to(configs.device)
    
            observed_mask = observed_mask.to(configs.device)
            observed_tp = observed_tp.to(configs.device)
    
            iter_count += 1
            model_optim.zero_grad()
            
            B,T,N,F = observed_data.shape
            random_array = torch.rand(B,T,N,F).to(configs.device)
            training_mask = torch.ones_like(random_array).to(configs.device)
            training_mask[random_array <= configs.mask_rate] = 0
            training_mask[random_array > configs.mask_rate] = 1
            training_mask[torch.where(observed_mask == 0)] = 0

            predict_data = model(observed_data, observed_tp, training_mask)
            loss = metirc(predict_data * observed_mask, observed_data * observed_mask)

            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch-1:
            train_loss = np.average(train_loss)
            print("Epoch: {} || cost time: {} || loss: {}".format(epoch + 1, time.time() - epoch_time, train_loss))

        
    return model

def model_test(configs, model):
    __, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    target = []
    forecast = []
    eval_points = []

    for i, (observed_data, observed_tp, observed_mask, gt_mask) in enumerate(test_loader):
        observed_data = observed_data.to(configs.device)
        observed_mask = observed_mask.to(configs.device)
        observed_tp = observed_tp.to(configs.device)

        impute_data = model(observed_data, observed_tp, observed_mask)
        impute_data = observed_mask * observed_data + (1-observed_mask) * impute_data
        
        impute_data = impute_data.detach().to("cpu").numpy()
        observed_mask = observed_mask.detach().to("cpu").numpy()
        gt_mask = gt_mask.detach().numpy()
        observed_data = observed_data.detach().to("cpu").numpy()

        evalmask = gt_mask - observed_mask

        target.append(observed_data)
        forecast.append(impute_data)
        eval_points.append(evalmask)

    target = np.vstack(target)
    forecast = np.vstack(forecast)
    eval_points = np.vstack(eval_points)

    RMSE = calc_RMSE(target, forecast, eval_points) 
    MAE = calc_MAE(target, forecast, eval_points)

    print("RMSE: ", RMSE)
    print("MAE: ", MAE)
    
    if configs.missing_rate == 0:
        forecast = forecast.reshape(forecast.shape[0] * forecast.shape[1], forecast.shape[2] * forecast.shape[3])
        np.savetxt("GSLI_imputed_data.csv", forecast, delimiter=",", fmt="%.6f")

def calc_RMSE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean((target[eval_p] - forecast[eval_p])**2)
    return np.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean(np.abs(target[eval_p] - forecast[eval_p]))
    return error_mean