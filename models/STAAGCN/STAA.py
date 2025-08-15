import torch
import torch.nn.functional as F
from STAAGCN.GRUcell import Cells
from STAAGCN.attention import Attention, MultiheadSoftClustering
import random


class staaGCN(torch.nn.Module):
    def __init__(self):
        super(staaGCN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 64
        self.m = 29
        self.forecast = 6
        self.stations = 35
        self.input_size = 29
        self.nhid_1 = 64
        self.nhid_2 = 64
        self.dropout = 0.5
        self.num_rels = 7
        # self.num_rels = 6
        self.num_bases = 3
        self.Wt = torch.nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.Wt_ = torch.nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bt = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.vt = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.ws = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wf = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wt = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wsf = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wst = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wft = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.wsft = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.bfc = torch.nn.Parameter(torch.FloatTensor(self.stations))
        self.att = Attention(self.hidden_size, 32)
        self.mltiattsoft = MultiheadSoftClustering(self.m, 8)
        self.cells = Cells(self.hidden_size, self.input_size, self.nhid_1, self.nhid_2, self.num_rels, self.num_bases)
        self.cells2 = Cells(self.hidden_size, self.hidden_size, self.nhid_1, self.nhid_2, self.num_rels, self.num_bases)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -0.2, 0.2)

    def input_transform(self, x):
        local_inputs, labels = x
        #  16, 12, 34, 26      16, 6, 1, 26
        local_inputs = local_inputs.permute(1, 0, 2, 3)  # (12h, batch, feature, stations)
        labels = labels.permute(1, 0, 2, 3)  # (6h, batch, 1features, stations)
        n_input_encoder = local_inputs.data.size(2)  # 28features
        batch_size = local_inputs.data.size(1)
        _local_inputs = local_inputs.contiguous().view(-1, n_input_encoder, self.stations)
        _local_inputs = torch.split(_local_inputs, batch_size, 0)
        encoder_inputs = _local_inputs
        _labels = labels.contiguous().view(-1, self.stations)
        _labels = torch.split(_labels, batch_size, 0)  # （batch, 1, stations）
        _lastinp = local_inputs[11:12, :, 0:1, :]  # 1, 256, 1, 35
        _lastinp = _lastinp.contiguous().view(-1, 1, self.stations)
        _lastinp = torch.split(_lastinp, batch_size, 0)
        decoder_inputs = list(_lastinp) + list(_labels[:-1])  # 6*(256, 1, 35)
        return encoder_inputs, _labels, decoder_inputs

    def Encoder(self, encoder_inputs, As, Af, At):
        Inputs = encoder_inputs # 12 (batch, feature, stations)
        batch_size = Inputs[0].data.size(0)
        stations = Inputs[0].data.size(2)
        lasth_As = torch.rand(batch_size, stations, self.hidden_size)
        lasth_Af = torch.rand(batch_size, stations, self.hidden_size)
        lasth_At = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_sf = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_st = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_ft = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_sft = torch.rand(batch_size, stations, self.hidden_size)
        hlist_As = []
        hlist_Af = []
        hlist_At = []
        hlist_com_sf = []
        hlist_com_st = []
        hlist_com_ft = []
        hlist_com_sft = []
        list_similar_loss_sf = []
        list_similar_loss_st = []
        list_similar_loss_ft = []
        list_similar_loss_sft = []
        list_sharedMarix_l1loss_sf = []
        list_sharedMarix_l1loss_st = []
        list_sharedMarix_l1loss_ft = []
        list_sharedMarix_l1loss_sft = []
        for flinput in Inputs:
            # 12 (batch, feature, stations)
            flinputx = flinput.permute(0, 2, 1) # stations, feature,
            flinputx = torch.as_tensor(flinputx, dtype=torch.float32).to(self.device)
            S = self.mltiattsoft(flinputx)

            As_new = As.to(self.device) * S.to(self.device)
            As_new = F.normalize(As_new)
            Af_new = Af.to(self.device) * S.to(self.device)
            Af_new = F.normalize(Af_new)
            At_new = At.to(self.device) * S.to(self.device)
            At_new = F.normalize(At_new)

            # h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft, sim_los_sf, sim_los_st, sim_los_ft, sim_los_sft, sMarix_l1_st, sMarix_l1_sf, sMarix_l1_ft, sMarix_l1_sft = self.cells(
            #     flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx,
            #     flinputx, flinputx, As_new, Af_new, At_new, lasth_As, lasth_Af, lasth_At, lasth_com_sf, lasth_com_st,
            #     lasth_com_ft, lasth_com_sft)
            h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft, sim_los_sf, sim_los_st, sim_los_ft, sim_los_sft, sMarix_l1_sf, sMarix_l1_st, sMarix_l1_ft, sMarix_l1_sft = self.cells(
                flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, flinputx, As_new, Af_new, At_new, lasth_As, lasth_Af, lasth_At, lasth_com_sf, lasth_com_st,
                lasth_com_ft, lasth_com_sft)

            lasth_As = h_As
            lasth_At = h_At
            lasth_Af = h_Af
            lasth_com_sf = h_com_sf
            lasth_com_st = h_com_st
            lasth_com_ft = h_com_ft
            lasth_com_sft = h_com_sft
            hlist_As.append(h_As)
            hlist_At.append(h_At)
            hlist_Af.append(h_Af)
            hlist_com_sf.append(h_com_sf)
            hlist_com_st.append(h_com_st)
            hlist_com_ft.append(h_com_ft)
            hlist_com_sft.append(h_com_sft)
            list_similar_loss_sf.append(sim_los_sf)
            list_similar_loss_st.append(sim_los_st)
            list_similar_loss_ft.append(sim_los_ft)
            list_similar_loss_sft.append(sim_los_sft)
            list_sharedMarix_l1loss_sf.append(sMarix_l1_sf)
            list_sharedMarix_l1loss_st.append(sMarix_l1_st)
            list_sharedMarix_l1loss_ft.append(sMarix_l1_ft)
            list_sharedMarix_l1loss_sft.append(sMarix_l1_sft)
        # return hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, \
        #        list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft, \
        #        list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft
        return hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft,\
               list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft,\
               list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft


    # def Decoder(self, As, Af, At, decoder_inputs, encoder_inputs, hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, teather_ratio):
    def Decoder(self, As, Af, At, decoder_inputs, encoder_inputs, hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, teather_ratio):
        Inputs = encoder_inputs
        batch_size = Inputs[0].data.size(0)
        stations = Inputs[0].data.size(2)
        lasth_As = torch.rand(batch_size, stations, self.hidden_size)
        lasth_Af = torch.rand(batch_size, stations, self.hidden_size)
        lasth_At = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_sf = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_st = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_ft = torch.rand(batch_size, stations, self.hidden_size)
        lasth_com_sft = torch.rand(batch_size, stations, self.hidden_size)
        predicts = []
        for dint in decoder_inputs:
            etlist = []
            # for h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft in zip(hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft):
            for h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft in zip(hlist_As, hlist_Af, hlist_At,
                                                                                     hlist_com_sf, hlist_com_st,
                                                                                     hlist_com_ft, hlist_com_sft):
                # et = torch.stack([h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft], dim=1)
                et = torch.stack([h_As, h_Af, h_At, h_com_sf, h_com_st, h_com_ft, h_com_sft], dim=1)
                et, _ = self.att(et)
                etlist.append(et)

            sumalfa1 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist1 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_As.to(self.device), self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist1.append(torch.exp(alfa))
                sumalfa1 = sumalfa1 + torch.exp(alfa)
            C1 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist1):
                alfa = torch.div(alfa, sumalfa1)
                C1 = C1 + alfa * et

            sumalfa2 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist2 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_Af.to(self.device), self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist2.append(torch.exp(alfa))
                sumalfa2 = sumalfa2 + torch.exp(alfa)
            C2 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist2):
                alfa = torch.div(alfa, sumalfa2)
                C2 = C2 + alfa * et

            sumalfa3 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist3 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_At.to(self.device), self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist3.append(torch.exp(alfa))
                sumalfa3 = sumalfa3 + torch.exp(alfa)
            C3 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist3):
                alfa = torch.div(alfa, sumalfa3)
                C3 = C3 + alfa * et

            sumalfa4 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist4 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(
                    torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_com_sf.to(self.device),
                                                                             self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist4.append(torch.exp(alfa))
                sumalfa4 = sumalfa4 + torch.exp(alfa)
            C4 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist4):
                alfa = torch.div(alfa, sumalfa4)
                C4 = C4 + alfa * et

            sumalfa5 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist5 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(
                    torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_com_st.to(self.device),
                                                                             self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist5.append(torch.exp(alfa))
                sumalfa5 = sumalfa5 + torch.exp(alfa)
            C5 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist5):
                alfa = torch.div(alfa, sumalfa5)
                C5 = C5 + alfa * et

            sumalfa6 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist6 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(
                    torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_com_ft.to(self.device),
                                                                             self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist6.append(torch.exp(alfa))
                sumalfa6 = sumalfa6 + torch.exp(alfa)
            C6 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist6):
                alfa = torch.div(alfa, sumalfa6)
                C6 = C6 + alfa * et

            sumalfa7 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            alfalist7 = []
            for et in etlist:
                alfa = torch.matmul(torch.tanh(
                    torch.matmul(et.to(self.device), self.Wt) + torch.matmul(lasth_com_sft.to(self.device),
                                                                             self.Wt_) + self.bt), self.vt)
                alfa = alfa.reshape(batch_size, -1, 1)
                alfalist7.append(torch.exp(alfa))
                sumalfa7 = sumalfa7 + torch.exp(alfa)
            C7 = torch.zeros(batch_size, stations, self.hidden_size).to(self.device)
            for et, alfa in zip(etlist, alfalist7):
                alfa = torch.div(alfa, sumalfa7)
                C7 = C7 + alfa * et

            ran_t = random.random()
            is_teather = ran_t < teather_ratio
            dint = dint.reshape(batch_size, self.stations, 1).float().repeat(1, 1, self.hidden_size)
            C1 = (dint if is_teather else C1)
            C2 = (dint if is_teather else C2)
            C3 = (dint if is_teather else C3)
            C4 = (dint if is_teather else C4)
            C5 = (dint if is_teather else C5)
            C6 = (dint if is_teather else C6)
            C7 = (dint if is_teather else C7)

            # d_As, d_Af, d_At, d_sf, d_st, d_ft, d_sft, _, _, _, _, _, _, _, _ = self.cells2(C1, C2, C3, C4, C4, C5, C5, C6, C6, C7, C7, C7, As, Af, At, lasth_As, lasth_Af, lasth_At, lasth_com_sf, lasth_com_st, lasth_com_ft, lasth_com_sft)
            d_As, d_Af, d_At, d_sf, d_st, d_ft, d_sft, _, _, _, _, _, _, _, _ = self.cells2(C1, C2, C3, C4, C4, C5, C5,
                                                                                            C6, C6, C7, C7, C7, As, Af,
                                                                                            At, lasth_As, lasth_Af,
                                                                                            lasth_At, lasth_com_sf,
                                                                                            lasth_com_st, lasth_com_ft, lasth_com_sft)
            lasth_As = d_As
            lasth_Af = d_Af
            lasth_At = d_At
            lasth_com_sf = d_sf
            lasth_com_st = d_st
            lasth_com_ft = d_ft
            lasth_com_sft = d_sft
            # ypredict = torch.tanh(torch.matmul(d_As, self.ws) + torch.matmul(d_Af, self.wf) + torch.matmul(d_At, self.wt) + self.bfc)
            # ypredict = torch.tanh(torch.matmul(d_As, self.ws) + torch.matmul(d_Af, self.wf) + torch.matmul(d_At, self.wt) + torch.matmul(d_st, self.wst) + torch.matmul(d_sf, self.wsf) + torch.matmul(d_ft, self.wft) + torch.matmul(d_sft, self.wsft))
            ypredict = torch.tanh(
                torch.matmul(d_As, self.ws) + torch.matmul(d_Af, self.wf) + torch.matmul(d_At, self.wt) + torch.matmul(d_sf, self.wsf) + torch.matmul(
                    d_st, self.wst) + torch.matmul(d_ft, self.wft) + torch.matmul(d_sft, self.wsft))
            predicts.append(ypredict)
        return predicts

    def forward(self, x, As, Af, At, teather_ratio):
        encoder_inputs, labels, decoder_inputs = self.input_transform(x)
        # hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft, list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft = self.Encoder(
        #     encoder_inputs, As, Af, At)
        hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft, list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft = self.Encoder(
            encoder_inputs, As, Af, At)
        # predicts = self.Decoder(As, Af, At, decoder_inputs, encoder_inputs, hlist_As, hlist_Af, hlist_At, hlist_com_sf, hlist_com_st, hlist_com_ft, hlist_com_sft, teather_ratio)
        predicts = self.Decoder(As, Af, At, decoder_inputs, encoder_inputs, hlist_As, hlist_Af, hlist_At, hlist_com_sf,
                                hlist_com_st, hlist_com_ft, hlist_com_sft, teather_ratio)
        # return predicts, labels, list_similar_loss, list_sharedMarix_l1loss
        # return predicts, labels, list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft, \
        #        list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft
        return predicts, labels, list_similar_loss_sf, list_similar_loss_st, list_similar_loss_ft, list_similar_loss_sft,\
               list_sharedMarix_l1loss_sf, list_sharedMarix_l1loss_st, list_sharedMarix_l1loss_ft, list_sharedMarix_l1loss_sft














