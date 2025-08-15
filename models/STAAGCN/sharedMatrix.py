import torch
from torch.nn.utils.prune import l1_unstructured

class SMLinear1(torch.nn.Module):
    def __init__(self, code_size):
        super(SMLinear1, self).__init__()
        self.code_size = code_size
        self.linear_matrix = torch.nn.Linear(self.code_size, self.code_size, bias=False)
        self.mask = torch.ones([self.code_size, self.code_size], dtype=torch.bool)
        mask_indices = [[range(self.code_size)], [range(self.code_size)]]
        self.mask[mask_indices] = 0
        def backward_hood(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.linear_matrix.weight.data[self.mask] = 0
        self.linear_matrix.weight.register_hook(backward_hood)

        l1_unstructured(self.linear_matrix, name='weight', amount=0.0001)

    def l1_loss(self):
        loss = 0.0
        for weight_ in self.linear_matrix.weight:
            loss = loss + weight_.abs().sum()
        return loss

    def forward(self, input_data1, input_data2):
        return self.linear_matrix(input_data1), self.linear_matrix(input_data2)


class SMLinear2(torch.nn.Module):
    def __init__(self, code_size):
        super(SMLinear2, self).__init__()
        self.code_size = code_size
        self.linear_matrix = torch.nn.Linear(self.code_size, self.code_size, bias=False)
        self.mask = torch.ones([self.code_size, self.code_size], dtype=torch.bool)
        mask_indices = [[range(self.code_size)], [range(self.code_size)]]
        self.mask[mask_indices] = 0
        def backward_hood(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.linear_matrix.weight.data[self.mask] = 0
        self.linear_matrix.weight.register_hook(backward_hood)

        l1_unstructured(self.linear_matrix, name='weight', amount=0.0001)

    def l1_loss(self):
        loss = 0.0
        for weight_ in self.linear_matrix.weight:
            loss = loss + weight_.abs().sum()
        return loss

    def forward(self, input_data1, input_data2, input_data3):
        return self.linear_matrix(input_data1), self.linear_matrix(input_data2), self.linear_matrix(input_data3)





