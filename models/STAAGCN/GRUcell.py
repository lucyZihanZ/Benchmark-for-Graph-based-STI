import torch
from STAAGCN.functions import sim_loss
from STAAGCN.sharedMatrix import SMLinear1, SMLinear2
from STAAGCN.GCNLayer import GCNLayer

class Cells(torch.nn.Module):
    def __init__(self, hidden_size, input_size, nhid_1, nhid_2, num_rels, num_bases):
        super(Cells, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.nhid_1 = nhid_1
        self.nhid_2 = nhid_2
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.mgcn_1 = GCNLayer(self.input_size, self.nhid_1, self.num_rels, self.num_bases)
        self.mgcn_2 = GCNLayer(self.input_size, self.nhid_1, self.num_rels, self.num_bases)
        self.shared_mat1_sf = SMLinear1(35)
        self.shared_mat1_st = SMLinear1(35)
        self.shared_mat1_ft = SMLinear1(35)
        self.shared_mat2 = SMLinear2(35)

        self.Wr_As = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_As = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_As = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_As_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_As_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_As_Af = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_As_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_As_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_As_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

        self.Wr_As_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wz_As_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)
        self.Wh_As_Af_At = torch.nn.Linear(self.hidden_size + self.nhid_2, self.hidden_size, bias=False)

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, adj_As, adj_Af, adj_At,
                hidden_As, hidden_Af, hidden_At, hidden_C_sf, hidden_C_st, hidden_C_ft, hidden_C_sft):

        # 卷积

        o_As1, o_Af1, o_At1, o_As_Af1, o_Af_As1, o_As_At1, o_At_As1, \
        o_Af_At1, o_At_Af1, o_As_Af_At1, o_As_At_Af1, o_At_As_Af1 = self.mgcn_1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, adj_As, adj_Af, adj_At)


        o_As2, o_Af2, o_At2, o_As_Af2, o_Af_As2, o_As_At2, o_At_As2, \
        o_Af_At2, o_At_Af2, o_As_Af_At2, o_As_At_Af2, o_At_As_Af2 = self.mgcn_2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
                                                            adj_As ** 2, adj_Af ** 2, adj_At ** 2)

        # 两层卷积
        out_As = o_As1 + o_As2
        out_Af = o_Af1 + o_Af2
        out_At = o_At1 + o_At2
        out_As_Af = o_As_Af1 + o_As_Af2
        out_Af_As = o_Af_As1 + o_Af_As2
        out_As_At = o_As_At1 + o_As_At2
        out_At_As = o_At_As1 + o_At_As2
        out_Af_At = o_Af_At1 + o_Af_At2
        out_At_Af = o_At_Af1 + o_At_Af2
        out_As_Af_At = o_As_Af_At1 + o_As_Af_At2
        out_As_At_Af = o_As_At_Af1 + o_As_At_Af2
        out_At_As_Af = o_At_As_Af1 + o_At_As_Af2

        # 共享对角矩阵
        o_As_Af_m_shared, o_Af_As_m_shared = self.shared_mat1_sf(out_As_Af.permute(0, 2, 1), out_Af_As.permute(0, 2, 1))
        o_As_At_m_shared, o_At_As_m_shared = self.shared_mat1_st(out_As_At.permute(0, 2, 1), out_At_As.permute(0, 2, 1))
        o_Af_At_m_shared, o_At_Af_m_shared = self.shared_mat1_ft(out_Af_At.permute(0, 2, 1), out_At_Af.permute(0, 2, 1))
        o_As_Af_At_m_shared, o_As_At_Af_m_shared, o_At_As_Af_m_shared = self.shared_mat2(out_As_Af_At.permute(0, 2, 1), out_As_At_Af.permute(0, 2, 1), out_At_As_Af.permute(0, 2, 1))

        # 相似约束
        sim_los_sf = sim_loss(o_As_Af_m_shared.permute(0, 2, 1), o_Af_As_m_shared.permute(0, 2, 1))
        sim_los_st = sim_loss(o_As_At_m_shared.permute(0, 2, 1), o_At_As_m_shared.permute(0, 2, 1))
        sim_los_ft = sim_loss(o_Af_At_m_shared.permute(0, 2, 1), o_At_Af_m_shared.permute(0, 2, 1))
        sim_los_sft = sim_loss(o_As_Af_At_m_shared.permute(0, 2, 1), o_As_At_Af_m_shared.permute(0, 2, 1)) + sim_loss(o_As_Af_At_m_shared.permute(0, 2, 1), o_At_As_Af_m_shared.permute(0, 2, 1)) \
                      + sim_loss(o_As_At_Af_m_shared.permute(0, 2, 1), o_At_As_Af_m_shared.permute(0, 2, 1))

        # 共享特征
        com_sf = (o_As_Af_m_shared.permute(0, 2, 1) + o_Af_As_m_shared.permute(0, 2, 1)) / 2
        com_st = (o_As_At_m_shared.permute(0, 2, 1) + o_At_As_m_shared.permute(0, 2, 1)) / 2
        com_ft = (o_Af_At_m_shared.permute(0, 2, 1) + o_At_Af_m_shared.permute(0, 2, 1)) / 2
        com_sft = (o_As_Af_At_m_shared.permute(0, 2, 1) + o_As_At_Af_m_shared.permute(0, 2, 1) + o_At_As_Af_m_shared.permute(0, 2, 1)) / 3

        # l1 loss
        sMarix_l1_sf = self.shared_mat1_sf.l1_loss()
        sMarix_l1_st = self.shared_mat1_st.l1_loss()
        sMarix_l1_ft = self.shared_mat1_ft.l1_loss()
        sMarix_l1_sft = self.shared_mat2.l1_loss()

        # 时空卷积
        x1_As = torch.cat([out_As, hidden_As.to(self.device)], dim=2)
        r_As = torch.sigmoid(self.Wr_As(x1_As))
        z_As = torch.sigmoid(self.Wz_As(x1_As))
        x2_As = torch.cat([r_As * hidden_As.to(self.device), out_As], dim=2)
        h_As_ = torch.tanh(self.Wh_As(x2_As))
        h_As = (1 - z_As) * hidden_As.to(self.device) + z_As * h_As_

        x1_Af = torch.cat([out_Af, hidden_Af.to(self.device)], dim=2)
        r_Af = torch.sigmoid(self.Wr_Af(x1_Af))
        z_Af = torch.sigmoid(self.Wz_Af(x1_Af))
        x2_Af = torch.cat([r_Af * hidden_Af.to(self.device), out_Af], dim=2)
        h_Af_ = torch.tanh(self.Wh_Af(x2_Af))
        h_Af = (1 - z_Af) * hidden_Af.to(self.device) + z_Af * h_Af_

        x1_At = torch.cat([out_At, hidden_At.to(self.device)], dim=2)
        r_At = torch.sigmoid(self.Wr_At(x1_At))
        z_At = torch.sigmoid(self.Wz_At(x1_At))
        x2_At = torch.cat([r_At * hidden_At.to(self.device), out_At], dim=2)
        h_At_ = torch.tanh(self.Wh_At(x2_At))
        h_At = (1 - z_At) * hidden_At.to(self.device) + z_At * h_At_

        x1_com_sf = torch.cat([com_sf, hidden_C_sf.to(self.device)], dim=2)
        r_com_sf = torch.sigmoid(self.Wr_As_Af(x1_com_sf))
        z_com_sf = torch.sigmoid(self.Wz_As_Af(x1_com_sf))
        x2_com_sf = torch.cat([r_com_sf * hidden_C_sf.to(self.device), com_sf], dim=2)
        h_com_sf_ = torch.tanh(self.Wh_As_Af(x2_com_sf))
        h_com_sf = (1 - z_com_sf) * hidden_C_sf.to(self.device) + z_com_sf * h_com_sf_

        x1_com_st = torch.cat([com_st, hidden_C_st.to(self.device)], dim=2)
        r_com_st = torch.sigmoid(self.Wr_As_At(x1_com_st))
        z_com_st = torch.sigmoid(self.Wz_As_At(x1_com_st))
        x2_com_st = torch.cat([r_com_st * hidden_C_st.to(self.device), com_st], dim=2)
        h_com_st_ = torch.tanh(self.Wh_As_At(x2_com_st))
        h_com_st = (1 - z_com_st) * hidden_C_st.to(self.device) + z_com_st * h_com_st_

        x1_com_ft = torch.cat([com_ft, hidden_C_ft.to(self.device)], dim=2)
        r_com_ft = torch.sigmoid(self.Wr_Af_At(x1_com_ft))
        z_com_ft = torch.sigmoid(self.Wz_Af_At(x1_com_ft))
        x2_com_ft = torch.cat([r_com_ft * hidden_C_ft.to(self.device), com_ft], dim=2)
        h_com_ft_ = torch.tanh(self.Wh_Af_At(x2_com_ft))
        h_com_ft = (1 - z_com_ft) * hidden_C_ft.to(self.device) + z_com_ft * h_com_ft_

        x1_com_sft = torch.cat([com_sft, hidden_C_sft.to(self.device)], dim=2)
        r_com_sft = torch.sigmoid(self.Wr_As_Af_At(x1_com_sft))
        z_com_sft = torch.sigmoid(self.Wz_As_Af_At(x1_com_sft))
        x2_com_sft = torch.cat([r_com_sft * hidden_C_sft.to(self.device), com_sft], dim=2)
        h_com_sft_ = torch.tanh(self.Wh_As_Af_At(x2_com_sft))
        h_com_sft = (1 - z_com_sft) * hidden_C_sft.to(self.device) + z_com_sft * h_com_sft_


        return h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft, \
               sim_los_sf, sim_los_st, sim_los_ft, sim_los_sft, sMarix_l1_st, sMarix_l1_sf, sMarix_l1_ft, sMarix_l1_sft
