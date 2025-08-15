import torch
import torch.nn.functional as F



def criterion(preds, labels):
    loss_fn = torch.nn.MSELoss()
    loss = 0.0
    for ps, ls in zip(preds, labels):
        loss = loss + loss_fn(ps.cpu().float(), ls.cpu().float())
    return loss


def sim_loss(input1, input2):
    loss_fn = torch.nn.MSELoss()
    input1= input1 - torch.mean(input1, dim=1, keepdim=True)
    input2 = input2 - torch.mean(input2, dim=1, keepdim=True)
    input1 = F.normalize(input1, p=2, dim=2)
    input2 = F.normalize(input2, p=2, dim=2)
    cov1 = torch.matmul(input1, input1.permute(0, 2, 1))
    cov2 = torch.matmul(input2, input2.permute(0, 2, 1))
    return loss_fn(cov1, cov2)


def other_loss(input1, input2, input3, input4, input5, input6, input7, input8):
    similar_loss_sf = 0.0
    similar_loss_st = 0.0
    similar_loss_ft = 0.0
    similar_loss_sft = 0.0
    sharedMarix_l1loss_sf = 0.0
    sharedMarix_l1loss_st = 0.0
    sharedMarix_l1loss_ft = 0.0
    sharedMarix_l1loss_sft = 0.0
    for i1, i2, i3, i4, i5, i6, i7, i8 in zip(input1, input2, input3, input4, input5, input6, input7, input8):
        # features_loss = features_loss + i1.cpu().float()
        similar_loss_sf = similar_loss_sf + i1.cpu().float()
        similar_loss_st = similar_loss_st + i2.cpu().float()
        similar_loss_ft = similar_loss_ft + i3.cpu().float()
        similar_loss_sft = similar_loss_sft + i4.cpu().float()
        sharedMarix_l1loss_sf = sharedMarix_l1loss_sf + i5.cpu().float()
        sharedMarix_l1loss_st = sharedMarix_l1loss_st + i6.cpu().float()
        sharedMarix_l1loss_ft = sharedMarix_l1loss_ft + i7.cpu().float()
        sharedMarix_l1loss_sft = sharedMarix_l1loss_sft + i8.cpu().float()
    return similar_loss_sf, similar_loss_st, similar_loss_ft, similar_loss_sft, sharedMarix_l1loss_sf, sharedMarix_l1loss_st, sharedMarix_l1loss_ft, sharedMarix_l1loss_sft







