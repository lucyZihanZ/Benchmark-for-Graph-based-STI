
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import torchcde

class LONDON_DATASET(Dataset):
    def __init__(self, configs):
        super(LONDON_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/london/london_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/london/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_london_dataset(configs):
    dataset = LONDON_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

class BEIJINGMEO_DATASET(Dataset):
    def __init__(self, configs):
        super(BEIJINGMEO_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/beijingmeo/beijingmeo_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/beijingmeo/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_beijingmeo_dataset(configs):
    dataset = BEIJINGMEO_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


class DUTCH_DATASET(Dataset):
    def __init__(self, configs):
        super(DUTCH_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/dutch/dutch_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/dutch/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_dutch_dataset(configs):
    dataset = DUTCH_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


class LOS_DATASET(Dataset):
    def __init__(self, configs):
        super(LOS_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/los/los_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/los/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0

        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()

        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_los_dataset(configs):
    dataset = LOS_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

class LUOHU_DATASET(Dataset):
    def __init__(self, configs):
        super(LUOHU_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/luohu/luohu_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/luohu/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0

        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_luohu_dataset(configs):
    dataset = LUOHU_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

class CN_DATASET(Dataset):
    def __init__(self, configs):
        super(CN_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/cn/cn_norm.csv", delimiter=",")
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/cn/mask/"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")
        self.gt_mask = np.ones_like(self.data)
        self.gt_mask[np.where(self.data == -200)] = 0

        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)
        self.mask = self.mask.reshape(self.mask.shape[0], self.configs.num_nodes, self.configs.feature)
        self.gt_mask = self.gt_mask.reshape(self.gt_mask.shape[0], self.configs.num_nodes, self.configs.feature)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0] // self.configs.seq_len

    def __getitem__(self, index):
        data_res = self.data[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        mask_res = self.mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]
        observed_tp = np.arange(self.configs.seq_len)
        gt_mask_res = self.gt_mask[index * self.configs.seq_len: (index+1) * self.configs.seq_len]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()
        observed_tp = torch.from_numpy(observed_tp).float()
        gt_mask_res = torch.from_numpy(gt_mask_res).float()
        return data_res, observed_tp, mask_res, gt_mask_res

def get_cn_dataset(configs):
    dataset = CN_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader




def get_dataset(configs):
    if configs.dataset == "london":
        return get_london_dataset(configs)
    if configs.dataset == "los":
        return get_los_dataset(configs)
    if configs.dataset == "luohu":
        return get_luohu_dataset(configs)
    if configs.dataset == "beijingmeo":
        return get_beijingmeo_dataset(configs)
    if configs.dataset == "dutch":
        return get_dutch_dataset(configs)
    if configs.dataset == "cn":
        return get_cn_dataset(configs)


class LONDON_DATASET_FORECASTING_TRAIN(Dataset):
    def __init__(self, configs):
        super(LONDON_DATASET_FORECASTING_TRAIN, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/london/london_norm.csv", delimiter=",")
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)

        self.mask = np.ones_like(self.data)
        self.mask[np.where(self.data == -200)] = 0
        x, y = [], []
        mask_x, mask_y = [], []

        index = 0
        while index + configs.seq_len + configs.pred_len <= self.data.shape[0]:
          
            x.append(self.data[index: index + configs.seq_len])
            mask_x.append(self.mask[index: index + configs.seq_len])
            y.append(self.data[index + configs.seq_len: index + configs.seq_len + configs.pred_len])
            mask_y.append(self.mask[index + configs.seq_len: index + configs.seq_len + configs.pred_len])

            index += (configs.seq_len + configs.pred_len)

        ins_num = len(x)
        self.x = np.array(x)[0: int(ins_num*0.7)]
        self.mask_x = np.array(mask_x)[0: int(ins_num*0.7)]
        self.y = np.array(y)[0: int(ins_num*0.7)]
        self.mask_y = np.array(mask_y)[0: int(ins_num*0.7)]

        
    def __len__(self):
        # Needs to be divisible
        return self.x.shape[0]

    def __getitem__(self, index):
        data_x = self.x[index]
        data_mask_x = self.mask_x[index]
        data_y = self.y[index]
        data_mask_y = self.mask_y[index]

        data_x = torch.from_numpy(data_x).float()
        data_mask_x = torch.from_numpy(data_mask_x).float()
        data_y = torch.from_numpy(data_y).float()
        data_mask_y = torch.from_numpy(data_mask_y).float()
        return data_x, data_mask_x, data_y, data_mask_y

class LONDON_DATASET_FORECASTING_TEST(Dataset):
    def __init__(self, configs):
        super(LONDON_DATASET_FORECASTING_TEST, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/london/london_norm.csv", delimiter=",")
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)

        self.mask = np.ones_like(self.data)
        self.mask[np.where(self.data == -200)] = 0
        x, y = [], []
        mask_x, mask_y = [], []

        index = 0
        while index + configs.seq_len + configs.pred_len <= self.data.shape[0]:
          
            x.append(self.data[index: index + configs.seq_len])
            mask_x.append(self.mask[index: index + configs.seq_len])
            y.append(self.data[index + configs.seq_len: index + configs.seq_len + configs.pred_len])
            mask_y.append(self.mask[index + configs.seq_len: index + configs.seq_len + configs.pred_len])

            index += (configs.seq_len + configs.pred_len)

        ins_num = len(x)
        self.x = np.array(x)[int(ins_num*0.7):]
        self.mask_x = np.array(mask_x)[int(ins_num*0.7):]
        self.y = np.array(y)[int(ins_num*0.7):]
        self.mask_y = np.array(mask_y)[int(ins_num*0.7):]

    
    def __len__(self):
        # Needs to be divisible
        return self.x.shape[0]

    def __getitem__(self, index):
        data_x = self.x[index]
        data_mask_x = self.mask_x[index]
        data_y = self.y[index]
        data_mask_y = self.mask_y[index]

        data_x = torch.from_numpy(data_x).float()
        data_mask_x = torch.from_numpy(data_mask_x).float()
        data_y = torch.from_numpy(data_y).float()
        data_mask_y = torch.from_numpy(data_mask_y).float()
        return data_x, data_mask_x, data_y, data_mask_y


def get_london_dataset_forecasting(configs):
    dataset = LONDON_DATASET_FORECASTING_TRAIN(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    dataset = LONDON_DATASET_FORECASTING_TEST(configs)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

class LUOHU_DATASET_FORECASTING_TRAIN(Dataset):
    def __init__(self, configs):
        super(LUOHU_DATASET_FORECASTING_TRAIN, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/luohu/luohu_norm.csv", delimiter=",")
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)

        self.mask = np.ones_like(self.data)
        self.mask[np.where(self.data == -200)] = 0
        x, y = [], []
        mask_x, mask_y = [], []

        index = 0
        while index + configs.seq_len + configs.pred_len <= self.data.shape[0]:
          
            x.append(self.data[index: index + configs.seq_len])
            mask_x.append(self.mask[index: index + configs.seq_len])
            y.append(self.data[index + configs.seq_len: index + configs.seq_len + configs.pred_len])
            mask_y.append(self.mask[index + configs.seq_len: index + configs.seq_len + configs.pred_len])

            index += (configs.seq_len + configs.pred_len)

        ins_num = len(x)
        self.x = np.array(x)[0: int(ins_num*0.7)]
        self.mask_x = np.array(mask_x)[0: int(ins_num*0.7)]
        self.y = np.array(y)[0: int(ins_num*0.7)]
        self.mask_y = np.array(mask_y)[0: int(ins_num*0.7)]

        
    def __len__(self):
        # Needs to be divisible
        return self.x.shape[0]

    def __getitem__(self, index):
        data_x = self.x[index]
        data_mask_x = self.mask_x[index]
        data_y = self.y[index]
        data_mask_y = self.mask_y[index]

        data_x = torch.from_numpy(data_x).float()
        data_mask_x = torch.from_numpy(data_mask_x).float()
        data_y = torch.from_numpy(data_y).float()
        data_mask_y = torch.from_numpy(data_mask_y).float()
        return data_x, data_mask_x, data_y, data_mask_y

class LUOHU_DATASET_FORECASTING_TEST(Dataset):
    def __init__(self, configs):
        super(LUOHU_DATASET_FORECASTING_TEST, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/luohu/luohu_norm.csv", delimiter=",")
        self.data = self.data.reshape(self.data.shape[0], self.configs.num_nodes, self.configs.feature)

        self.mask = np.ones_like(self.data)
        self.mask[np.where(self.data == -200)] = 0
        x, y = [], []
        mask_x, mask_y = [], []

        index = 0
        while index + configs.seq_len + configs.pred_len <= self.data.shape[0]:
          
            x.append(self.data[index: index + configs.seq_len])
            mask_x.append(self.mask[index: index + configs.seq_len])
            y.append(self.data[index + configs.seq_len: index + configs.seq_len + configs.pred_len])
            mask_y.append(self.mask[index + configs.seq_len: index + configs.seq_len + configs.pred_len])

            index += (configs.seq_len + configs.pred_len)

        ins_num = len(x)
        self.x = np.array(x)[int(ins_num*0.7):]
        self.mask_x = np.array(mask_x)[int(ins_num*0.7):]
        self.y = np.array(y)[int(ins_num*0.7):]
        self.mask_y = np.array(mask_y)[int(ins_num*0.7):]

        
    def __len__(self):
        # Needs to be divisible
        return self.x.shape[0]

    def __getitem__(self, index):
        data_x = self.x[index]
        data_mask_x = self.mask_x[index]
        data_y = self.y[index]
        data_mask_y = self.mask_y[index]

        data_x = torch.from_numpy(data_x).float()
        data_mask_x = torch.from_numpy(data_mask_x).float()
        data_y = torch.from_numpy(data_y).float()
        data_mask_y = torch.from_numpy(data_mask_y).float()
        return data_x, data_mask_x, data_y, data_mask_y


def get_luohu_dataset_forecasting(configs):
    dataset = LUOHU_DATASET_FORECASTING_TRAIN(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    dataset = LUOHU_DATASET_FORECASTING_TEST(configs)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


def get_dataset_forecasting(configs):
    if configs.dataset == "london":
        return get_london_dataset_forecasting(configs)
    if configs.dataset == "luohu":
        return get_luohu_dataset_forecasting(configs)

def get_london_adj():
    adj_in = np.loadtxt("Data/london/london_adj.csv", delimiter=",")
    return adj_in
def get_beijingmeo_adj():
    adj_in = np.loadtxt("Data/beijingmeo/beijingmeo_adj.csv", delimiter=",")
    return adj_in
def get_dutch_adj():
    adj_in = np.loadtxt("Data/dutch/dutch_adj.csv", delimiter=",")
    return adj_in
def get_los_adj():
    adj_in = np.loadtxt("Data/los/los_adj.csv", delimiter=",")
    return adj_in
def get_luohu_adj():
    adj_in = np.loadtxt("Data/luohu/luohu_adj.csv", delimiter=",")
    return adj_in
def get_cn_adj():
    adj_in = np.loadtxt("Data/cn/cn_adj.csv", delimiter=",")
    return adj_in


def get_adj(configs):
    if configs.dataset == "london":
        return get_london_adj()
    if configs.dataset == "los":
        return get_los_adj()
    if configs.dataset == "luohu":
        return get_luohu_adj()
    if configs.dataset == "beijingmeo":
        return get_beijingmeo_adj()
    if configs.dataset == "dutch":
        return get_dutch_adj()
    if configs.dataset == "cn":
        return get_cn_adj()
