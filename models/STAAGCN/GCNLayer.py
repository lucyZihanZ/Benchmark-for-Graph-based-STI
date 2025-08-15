import torch
import math


class GCNLayer(torch.nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases):
        super(MGCNLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels  # num of relationships
        self.num_bases = num_bases
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_feat, self.out_feat))
        self.w_comp = torch.nn.Parameter(torch.FloatTensor(self.num_rels, self.num_bases))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-std, std)

    # def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, adj_As, adj_Af, adj_At):
    #     weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
    #     weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, adj_As, adj_Af, adj_At):
        weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)

        # adj_As
        weight_adj_As = weight[0]
        support_adj_As = torch.matmul(x1, weight_adj_As.to(self.device))
        output_adj_As = torch.matmul(adj_As.to(self.device), support_adj_As)

        # adj_Af
        weight_adj_Af = weight[1]
        support_adj_Af = torch.matmul(x2, weight_adj_Af.to(self.device))
        output_adj_Af = torch.matmul(adj_Af.to(self.device), support_adj_Af)

        # adj_At
        weight_adj_At = weight[2]
        support_adj_At = torch.matmul(x3, weight_adj_At.to(self.device))
        output_adj_At = torch.matmul(adj_At.to(self.device), support_adj_At)

        # adj_As--adj_Af
        weight_adj_As_Af = weight[3]
        support_As_Af_shared = torch.matmul(x4, weight_adj_As_Af.to(self.device))
        support_Af_As_shared = torch.matmul(x5, weight_adj_As_Af.to(self.device))
        output_As_Af_shared = torch.matmul(adj_As.to(self.device), support_As_Af_shared)
        output_Af_As_shared = torch.matmul(adj_Af.to(self.device), support_Af_As_shared)

        # adj_As--adj_At
        weight_adj_As_At = weight[4]
        support_As_At_shared = torch.matmul(x6, weight_adj_As_At.to(self.device))
        support_At_As_shared = torch.matmul(x7, weight_adj_As_At.to(self.device))
        output_As_At_shared = torch.matmul(adj_As.to(self.device), support_As_At_shared)
        output_At_As_shared = torch.matmul(adj_At.to(self.device), support_At_As_shared)

        # adj_Af--adj_At
        weight_adj_Af_At = weight[5]
        support_Af_At_shared = torch.matmul(x8, weight_adj_Af_At.to(self.device))
        support_At_Af_shared = torch.matmul(x9, weight_adj_Af_At.to(self.device))
        output_Af_At_shared = torch.matmul(adj_Af.to(self.device), support_Af_At_shared)
        output_At_Af_shared = torch.matmul(adj_At.to(self.device), support_At_Af_shared)

        # adj_As--adj_Af--adj_At
        weight_adj_As_Af_At = weight[6]
        support_As_Af_At_shared = torch.matmul(x10, weight_adj_As_Af_At.to(self.device))
        support_Af_As_At_shared = torch.matmul(x11, weight_adj_As_Af_At.to(self.device))
        support_At_As_Af_shared = torch.matmul(x12, weight_adj_As_Af_At.to(self.device))
        output_As_Af_At_shared = torch.matmul(adj_As.to(self.device), support_As_Af_At_shared)
        output_Af_As_At_shared = torch.matmul(adj_Af.to(self.device), support_Af_As_At_shared)
        output_At_As_Af_shared = torch.matmul(adj_At.to(self.device), support_At_As_Af_shared)

        # return output_adj_As, output_adj_Af, output_adj_At, output_As_Af_shared, output_Af_As_shared, \
        #        output_As_At_shared, output_At_As_shared, output_Af_At_shared, output_At_Af_shared, \
        #        output_As_Af_At_shared, output_Af_As_At_shared, output_At_As_Af_shared
        return output_adj_As, output_adj_Af, output_adj_At, output_As_Af_shared, output_Af_As_shared, \
               output_As_At_shared, output_At_As_shared, output_Af_At_shared, output_At_Af_shared, output_As_Af_At_shared, output_Af_As_At_shared, output_At_As_Af_shared










