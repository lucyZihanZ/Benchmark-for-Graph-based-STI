import torch


class Attention(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MultiheadSoftClustering(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(MultiheadSoftClustering, self).__init__()
        self.project1 = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
        self.project2 = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
        self.project3 = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
        self.project4 = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w1 = self.project1(z)
        beta1 = torch.softmax(w1, dim=1)
        w2 = self.project2(z)
        beta2 = torch.softmax(w2, dim=1)
        w3 = self.project3(z)
        beta3 = torch.softmax(w3, dim=1)
        w4 = self.project4(z)
        beta4 = torch.softmax(w4, dim=1)
        beta = (beta1 + beta2 + beta3 + beta4) / 4
        return beta



