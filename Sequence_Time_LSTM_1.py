
import torch
import torch.nn as nn
import numpy as np
import math
from lstm import Time_LSTM_1

# TIME_EMBED_SIZE = config.TIME_EMBED_SIZE
# BATCHSIZE = config.BATCHSIZE
# INPUT_LENGTH = config.INPUT_LENGTH
# OUTPUT_LENGTH = config.OUTPUT_LENGTH
# device = config.device

class RnnModelInterp(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RnnModelInterp, self).__init__()
        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']

        self.hid2category = nn.Linear(kwargs["h_size"], kwargs["nb_classes"])
        self.hid2measures = nn.Linear(kwargs["h_size"], 22)

        self.cells = nn.ModuleList()

        self.cells.append(
            Time_LSTM_1(kwargs["nb_measures"], kwargs["h_size"], kwargs["TIME_EMBED_SIZE"]))
        for _ in range(1, kwargs['nb_layers']):
            self.cells.append(Time_LSTM_1(kwargs["h_size"], kwargs["h_size"], kwargs["TIME_EMBED_SIZE"]))
        # for cell in self.cells:
        #     jozefowicz_init(cell.bias_hh)

class Sequence_Time_LSTM_1(RnnModelInterp):
    """ Minimal RNN """

    # def __init__(self, **kwargs):
    #     # super(MinimalRNN, self).__init__(MinimalRNNCell, **kwargs)
    def __init__(self, **kwargs):
        super(Sequence_Time_LSTM_1, self).__init__(**kwargs)
        dev = next(self.parameters()).device
        # layers, time, hidden
        self.h_t = torch.zeros(kwargs['nb_layers'], 1, kwargs['h_size']).to(dev)
        self.c_t = torch.zeros(kwargs['nb_layers'], 1, kwargs['h_size']).to(dev)
        self.nb_measures = kwargs['nb_measures']


    def reset_parameters(self, hid):
        # stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv = 1.0 / math.sqrt(hid[0].shape[1])
        self.h_t.data.uniform_(-stdv, stdv)
        self.c_t.data.uniform_(-stdv, stdv)
        # self.bias.data.uniform_(stdv)
        return self.c_t

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        # i_mask = torch.ones(
        #     batch_size, self.hid2measures.out_features, device=dev)
        i_mask = torch.ones(
            # batch_size, self.hid2measures.out_features, device=dev)
            batch_size, self.nb_measures, device=dev)
        r_mask = [
            # torch.ones(batch_size, cell.hidden_size, device=dev)
            torch.ones(batch_size, cell.hidden_length, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            # state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
            state.append(torch.zeros(batch_size, cell.hidden_length, device=dev))
        return state

    def forward(self, _cat_seq, _val_seq, delta, task_name):
        out_cat_seq, out_val_seq= [], []
        # delta = torch.Tensor(delta)
        # delta.cuda(1)
        hidden0 = self.init_hidden_state(_val_seq.shape[1])
        c0 = self.init_hidden_state(_val_seq.shape[1])
        hidden1 = self.init_hidden_state(_val_seq.shape[1])
        c1 = self.init_hidden_state(_val_seq.shape[1])
        hidden2 = self.init_hidden_state(_val_seq.shape[1])
        c2 = self.init_hidden_state(_val_seq.shape[1])


        masks = self.dropout_mask(_val_seq.shape[1])


        cat_seq = _cat_seq.copy()
        val_seq = _val_seq.copy()
        hidden_shared_arr = []
        c_shared_arr = []

        # if len(hidden_shared_arr)==0 or len(c_shared_arr)==0:
        for i in range(len(val_seq)):
            hidden_shared_arr.append(self.init_hidden_state(_val_seq.shape[1]))
            c_shared_arr.append(self.init_hidden_state(_val_seq.shape[1]))
        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            if task_name==0:
               hidden_shared,c_shared,\
                o_cat, o_val, hidden0, c0 = self.predict(cat_seq[i], val_seq[i],hidden_shared_arr[i], hidden0,
                                                    masks, delta[i],c_shared_arr[i],c0)
            if task_name==1:
                hidden_shared,c_shared,\
                o_cat, o_val, hidden1, c1 = self.predict(cat_seq[i], val_seq[i],hidden_shared_arr[i], hidden1,
                                                    masks, delta[i],c_shared_arr[i],c1)
            if task_name==2:
                hidden_shared,c_shared,\
                o_cat, o_val, hidden2, c2 = self.predict(cat_seq[i], val_seq[i],hidden_shared_arr[i], hidden2,
                                                    masks, delta[i],c_shared_arr[i],c2)


            hidden_shared_arr[i] = hidden_shared
            c_shared_arr[i] = c_shared
            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = np.isnan(val_seq[j])
            # val_seq[j][idx] = val_seq[i][idx]
            val_seq[j][idx] = o_val.data.cpu().numpy()[:, 2:][idx]
            idx = np.isnan(cat_seq[j])
            cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]

        return torch.stack(out_cat_seq), torch.stack(out_val_seq)

    # cat_seq[i], val_seq[i], hidden_shared_arr[i], hidden0,
    # masks, delta[i], c_shared_arr[i], c0
    def predict(self, i_cat, i_val, hidden_shared,hidden, masks, delta, c_shared,c):
        i_mask, r_mask = masks
        h_t = hidden[0].new(i_val) * i_mask
        next_hid_shared = []
        next_c_shared = []
        next_hid = []
        next_c = []

        for cell, prev_h_shared,prev_h, c_next_shared,c_next, mask in \
                zip(self.cells,hidden_shared, hidden,c_shared, c, r_mask):

            # prev_h, c_next, delta = cell(h_t, (prev_h * mask, c_next), delta)
            prev_h_shared, c_next_shared,\
                prev_h, c_next, \
                    = cell(h_t,(prev_h_shared*mask,c_next_shared),(prev_h*mask,c_next),delta)

            next_hid_shared.append(prev_h_shared)
            next_c_shared.append(c_next_shared)

            next_hid.append(prev_h)
            next_c.append(c_next)

        o_cat1 = nn.functional.softmax(self.hid2category(prev_h), dim=-1)
        o_val1 = self.hid2measures(prev_h)

        return next_hid_shared,next_c_shared,\
               o_cat1, o_val1, next_hid, next_c