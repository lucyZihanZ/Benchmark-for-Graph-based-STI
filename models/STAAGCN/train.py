import numpy as np
import torch.optim as optim
import torch
import random
import matplotlib.pyplot as plt
import math
import datetime
from STAAGCN.utils_graph import RGC
from STAAGCN.earlystopping import EarlyStopping
from STAAGCN.STAA import staaGCN
from STAAGCN.functions import criterion, other_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2021)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = staaGCN()
model.to(device)

print('开始时间-------', datetime.datetime.now())
total_epoch = 500
# total_epoch = 5
batch_size = 256
lr = 0.001

traindata = np.load("train_bj.npy")
testdata = np.load("test_bj.npy")
pm25_train = np.load("pm25_train_bj.npy")
pm25_test = np.load("pm25_test_bj.npy")
max_value = 560
min_value = 2
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
index = [i for i in range(len(testdata))]
random.shuffle(index)
testdata = testdata[index]
pm25_test = pm25_test[index]

def get_batch_feed_dict(k, batch_size, training_data,train_labels_data):
    batch_train_inp = training_data[k:k + batch_size]
    batch_label_inp = train_labels_data[k:k + batch_size]
    feed_dict = (batch_train_inp,batch_label_inp)
    return feed_dict

num_train = 7777
num_valid = 1193
num_test = 1193

train_data = torch.from_numpy(traindata[:num_train, :, :, :]).to(device) # 6990, 12, 34, 26
train_labels_data = torch.from_numpy(pm25_train[:num_train, :, :, :]).to(device) # 6990, 6, 1, 26

valid_data = torch.from_numpy(traindata[num_train:, :, :, :]).to(device)
valid_labels_data = torch.from_numpy(pm25_train[num_train:, :, :]).to(device)

test_data = torch.from_numpy(testdata[:, :, :, :]).to(device)
test_labels_data = torch.from_numpy(pm25_test[:, :, :]).to(device)

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

early_stopping = EarlyStopping(patience=15, verbose=True)

teather_ratio = 0.8
As, Af, At = RGC()
te = 0.0

a_weight = 0.01
b_weight = 0.01
c_weight = 0.01
d_weight = 0.01
aa_weight = 0.001
bb_weight = 0.001
cc_weight = 0.001
dd_weight = 0.01

for i in range(total_epoch):
    print('----------epoch {}-----------{}'.format(i, datetime.datetime.now()))
    i += 1
    teather_ratio = teather_ratio * (0.8 ** (i // 30))
    for j in range(0, num_train, batch_size):
        x = get_batch_feed_dict(j, batch_size, train_data, train_labels_data)
        preds, labels, list_sim_l_sf, list_sim_l_st, list_sim_l_ft, list_sim_l_sft, list_shared_l1l_sf, list_shared_l1l_st, list_shared_l1l_ft, list_shared_l1l_sft = model(
            x, As, Af, At, teather_ratio)

        optimizer.zero_grad()

        preds_loss = criterion(preds, labels)
        similar_loss_sf, similar_loss_st, similar_loss_ft, similar_loss_sft, sharedMarix_l1loss_sf, sharedMarix_l1loss_st, sharedMarix_l1loss_ft, sharedMarix_l1loss_sft = \
            other_loss(list_sim_l_sf, list_sim_l_st, list_sim_l_ft, list_sim_l_sft, list_shared_l1l_sf,
                       list_shared_l1l_st, list_shared_l1l_ft, list_shared_l1l_sft)

        loss = preds_loss + a_weight * similar_loss_sf + b_weight * similar_loss_st + c_weight * similar_loss_ft + d_weight * similar_loss_sft+ \
               aa_weight * sharedMarix_l1loss_sf + bb_weight * sharedMarix_l1loss_st + cc_weight * sharedMarix_l1loss_ft \
               + dd_weight * sharedMarix_l1loss_sft

        loss.backward(retain_graph=True)
        optimizer.step()
        train_losses.append(loss.item())

    for j in range(0, num_valid, batch_size):
        x = get_batch_feed_dict(j, batch_size, valid_data, valid_labels_data)
        preds, labels, list_sim_l_sf, list_sim_l_st, list_sim_l_ft, list_sim_l_sft, list_shared_l1l_sf, list_shared_l1l_st, list_shared_l1l_ft, list_shared_l1l_sft = model(
            x, As, Af, At, te)

        preds_loss = criterion(preds, labels)
        # similar_loss_sf, similar_loss_st, similar_loss_ft, similar_loss_sft, sharedMarix_l1loss_sf, sharedMarix_l1loss_st, sharedMarix_l1loss_ft, sharedMarix_l1loss_sft = other_loss(list_sim_l_sf, list_sim_l_st, list_sim_l_ft, list_sim_l_sft, list_shared_l1l_sf, list_shared_l1l_st, list_shared_l1l_ft, list_shared_l1l_sft)
        similar_loss_sf, similar_loss_st, similar_loss_ft, similar_loss_sft, sharedMarix_l1loss_sf, sharedMarix_l1loss_st, sharedMarix_l1loss_ft, sharedMarix_l1loss_sft = \
            other_loss(list_sim_l_sf, list_sim_l_st, list_sim_l_ft, list_sim_l_sft, list_shared_l1l_sf,
                       list_shared_l1l_st, list_shared_l1l_ft, list_shared_l1l_sft)

        loss = preds_loss + a_weight * similar_loss_sf + b_weight * similar_loss_st + c_weight * similar_loss_ft + d_weight * similar_loss_sft + \
               aa_weight * sharedMarix_l1loss_sf + bb_weight * sharedMarix_l1loss_st + cc_weight * sharedMarix_l1loss_ft \
               + dd_weight * sharedMarix_l1loss_sft

        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    early_stopping(valid_loss, model)
    print_msg = (' train_loss:' + str(train_loss) + ' valid_loss:' + str(valid_loss))
    print(print_msg)
    if early_stopping.early_stop:
        print("Early stopping")
        break

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')
minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')


def MAEloss(preds, labels):
    loss_fn = torch.nn.L1Loss()
    loss = 0.0
    for ps, ls in zip(preds, labels):
        loss = loss + loss_fn(ps.cpu().float() * (max_value - min_value) + min_value,
                              ls.cpu().float() * (max_value - min_value) + min_value)
    return loss


def RMSEloss(preds, labels)                                                                                                                                                                   :
    loss_fn = torch.nn.MSELoss()
    loss = 0.0
    for ps, ls in zip(preds, labels):
        loss = loss + loss_fn(ps.cpu().float() * (max_value - min_value) + min_value,
                              ls.cpu().float() * (max_value - min_value) + min_value)
    return loss


def MAPEloss(preds, labels, null_val = np.nan):
    def loss_fn(preds, labels, null_val):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        diff = torch.abs(labels - preds)/labels
        diff = diff * mask
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        return torch.mean(diff)
    loss = 0.0
    for ps, ls in zip(preds, labels):
        loss = loss + loss_fn(ps.float() * (max_value - min_value) + min_value,
                              ls.float() * (max_value - min_value) + min_value, 0.0)
    return loss


rmses = []
maes = []
mapes = []
test_loss = []
test_loss1 = []
test_loss2 = []

for k in range(0, num_test, batch_size):
    teather_ratio = 0
    x = get_batch_feed_dict(k, batch_size, test_data, test_labels_data)
    preds, labels, _, _, _, _, _, _, _, _ = model(x, As, Af, At, teather_ratio)
    oss = RMSEloss(preds, labels) / 6
    loss1 = MAEloss(preds, labels) / 6
    loss2 = MAPEloss(preds, labels) / 6
    test_loss.append(loss.item())
    test_loss1.append(loss1.item())
    test_loss2.append(loss2.item())

print('===============METRIC===============')
MAE = np.average(test_loss1)
mseLoss = math.sqrt(np.average(test_loss))
MAPE = np.average(test_loss2) * 100
print('MAE = {:.6f}'.format(MAE))
print('RMSE = {:.6f}'.format(mseLoss))
print('MAPE = {:.6f}'.format(MAPE))

