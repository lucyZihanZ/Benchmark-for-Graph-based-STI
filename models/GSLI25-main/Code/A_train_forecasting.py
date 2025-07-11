
import A_dataset
from torch import optim
import time
import numpy as np
import torch
import torch.nn as nn
from Models import GSLI_forecasting

def model_train(configs):

    train_loader, __ = A_dataset.get_dataset_forecasting(configs)


    model = GSLI_forecasting.GSLIModel(configs).to(configs.device)


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
        for i, (data_x, data_mask_x, data_y, data_mask_y) in enumerate(train_loader):
            data_x = data_x.to(configs.device)
            data_mask_x = data_mask_x.to(configs.device)
            data_y = data_y.to(configs.device)
            data_mask_y = data_mask_y.to(configs.device)

            observed_tp = np.arange(data_x.shape[1])
            observed_tp = np.tile(observed_tp, (data_x.shape[0], 1))

            observed_tp = torch.from_numpy(observed_tp).float().to(configs.device)
        

            iter_count += 1
            model_optim.zero_grad()
            
            B,T,N,F = data_x.shape

            predict_data = model(data_x, observed_tp, data_mask_x)
            loss = metirc(predict_data * data_mask_y, data_y * data_mask_y)
        

            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 10 == 0 or epoch == configs.epoch-1:
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print(train_loss)

        
    return model

def model_test(configs, model):
    __, test_loader = A_dataset.get_dataset_forecasting(configs)
    model.eval()

    target = []
    forecast = []
    eval_points = []

    for i, (data_x, data_mask_x, data_y, data_mask_y) in enumerate(test_loader):
        data_x = data_x.to(configs.device)
        data_mask_x = data_mask_x.to(configs.device)
        data_y = data_y.to(configs.device)
        data_mask_y = data_mask_y.to(configs.device)

        observed_tp = np.arange(data_x.shape[1])
        observed_tp = np.tile(observed_tp, (data_x.shape[0], 1))
        observed_tp = torch.from_numpy(observed_tp).float().to(configs.device)
    

        predict_data = model(data_x, observed_tp, data_mask_x)
        
        predict_data = predict_data.detach().to("cpu").numpy()
        data_y = data_y.detach().to("cpu").numpy()
        data_mask_y = data_mask_y.detach().to("cpu").numpy()


        target.append(data_y)
        forecast.append(predict_data)
        eval_points.append(data_mask_y)

    target = np.vstack(target)
    forecast = np.vstack(forecast)
    eval_points = np.vstack(eval_points)

    RMSE = calc_RMSE(target, forecast, eval_points) 
    MAE = calc_MAE(target, forecast, eval_points)

    print("RMSE: ", RMSE)
    print("MAE: ", MAE)

    print(RMSE)
    print(MAE)
    

def calc_RMSE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean((target[eval_p] - forecast[eval_p])**2)
    return np.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = np.where(eval_points == 1)
    error_mean = np.mean(np.abs(target[eval_p] - forecast[eval_p]))
    return error_mean