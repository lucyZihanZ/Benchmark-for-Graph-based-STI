import torch
import numpy as np
import A_train
import argparse

parser = argparse.ArgumentParser(description='GSLI')

parser.add_argument('--device', type=str, default="cuda", help='input sequence length')
parser.add_argument('--batch', type=int, default=16, help='input batch size')
parser.add_argument('--missing_rate', type=float, default=0.1, help='missing percent for experiment')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# input data setting: 
# dutchwind:{seq_len:48 node:7, feature:4} 
# beijingmeo:{seq_len:48 node:18, feature:5} 
# londonaq:{seq_len:48 node:13, feature:3} 
# cn: {seq_len:8 node:140, feature:6} 
# los: {seq_len:16 node:207, feature:1} 
# luohutaxi: {seq_len:16 node:156, feature:1} 
parser.add_argument('--dataset', type=str, default="beijingmeo", help='dataset name')
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--num_nodes', type=int, default=18, help='input node num')
parser.add_argument('--feature', type=int, default=5, help='input feature dim')

#######################################################################################
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learing rate')
parser.add_argument('--epoch', type=int, default=100, help='traning epoch')
parser.add_argument('--mask_rate', type=float, default=0.2, help='training mask rate')

parser.add_argument('--timeemb', type=int, default=128, help='side information timeemb dimension')
parser.add_argument('--featureemb', type=int, default=16, help='side information dimension')
parser.add_argument('--nheads', type=int, default=8, help='number of head for attention')
parser.add_argument('--proj_t', type=int, default=128, help='proj_t for feature self-attention')
parser.add_argument('--channels', type=int, default=128, help='channel dimension')
parser.add_argument('--layers', type=int, default=3, help='numeber of layers')

if __name__ == '__main__':
    configs = parser.parse_args()
    print(configs)

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed_all(configs.seed)

    model = A_train.model_train(configs)
    A_train.model_test(configs, model)

