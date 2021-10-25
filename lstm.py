
import torch
import math
import torch.nn as nn
class Time_LSTM_1(torch.nn.Module):

    def __init__(self, input_length, hidden_length, TIME_EMBED_SIZE):
        super(Time_LSTM_1, self).__init__()
        self.hidden_length = hidden_length
        self.input_length = input_length

        self.mutivariate = nn.Linear(self.input_length, self.input_length, bias=True)
        # feature attention
        self.linear_feature_attention_W_x = nn.Linear(self.input_length, self.input_length, bias=True)
        self.sigmoid_feature_attention = nn.Sigmoid()
        # shared_layer attention
        self.linear_shared_layer_attention_W_x = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_shared_layer_attention = nn.Sigmoid()

        # input gate components
        self.linear_input_W_x = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_forget_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # time gate components
        self.linear_time_W_x = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_time_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_time = nn.Sigmoid()
        self.sigmoid_time_inner = nn.Sigmoid()

        # cell memory components
        self.linear_cell_W_x = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell = nn.Tanh()

        # out gate components
        self.linear_output_W_x = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_output_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out = nn.Sigmoid()

        self.linear_input_W_x = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input = nn.Sigmoid()

        # =====================================================================

        # input gate shared components
        self.linear_input_W_x_shared = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_input_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)

        # forget gate shared components
        self.linear_forget_W_x_shared = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_forget_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)

        # time gate shared components
        self.linear_time_W_x_shared = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_time_W_t_shared = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_time_shared = nn.Sigmoid()

        # cell memory shared components
        self.linear_cell_W_x_shared = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_cell_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)

        # out gate shared components
        self.linear_output_W_x_shared = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_output_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_W_t_shared = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.activation_final = nn.Tanh()


    # def reset_parameters(self):
    #     # stdv = 1.0 / math.sqrt(self.hidden_size)
    #     stdv = 1.0 / math.sqrt(self.hidden_length)
    #     self.weight_i2h.data.uniform_(-stdv, stdv)
    #     self.weight_h2h.data.uniform_(-stdv, stdv)
    #
    #     self.bias.data.uniform_(stdv)


    def input_gate(self, x, h, c_prev,shared):
        if shared:
            x_temp = self.linear_input_W_x_shared(x)
            h_temp = self.linear_input_W_h_shared(h)
            c_temp = self.linear_input_w_c_shared(c_prev)
            i = self.sigmoid_input(x_temp + h_temp + c_temp)
            return i
        else:
            x_temp = self.linear_input_W_x(x)
            h_temp = self.linear_input_W_h(h)
            c_temp = self.linear_input_w_c(c_prev)
            i = self.sigmoid_input(x_temp + h_temp + c_temp)
            return i

    def forget_gate(self, x, h, c_prev,shared):
        if shared:
            x = self.linear_forget_W_x_shared(x)
            h = self.linear_forget_W_h_shared(h)
            c = self.linear_forget_w_c_shared(c_prev)
            f = self.sigmoid_forget(x + h + c)
            return f
        else:
            x = self.linear_forget_W_x(x)
            h = self.linear_forget_W_h(h)
            c = self.linear_forget_w_c(c_prev)
            f = self.sigmoid_forget(x + h + c)
            return f


    def time_gate(self, x, delt,shared):
        if shared:

            x = self.linear_time_W_x_shared(x)
            tt = self.linear_time_W_t_shared(delt)
            t = self.sigmoid_time_inner(tt)
            T = self.sigmoid_time(x + t)
            return T
        else:
            x = self.linear_time_W_x(x)
            tt = self.linear_time_W_t(delt)
            t = self.sigmoid_time_inner(tt)
            T = self.sigmoid_time(x + t)
            return T

    def cell_memory_gate(self, i, f, x, h, c_prev, T,shared):
        if shared:
            x = self.linear_cell_W_x_shared(x)
            h = self.linear_cell_W_h_shared(h)
            k = self.tanh_cell(x + h)
            g = (i * T * k)
            c = (f * c_prev)
            c_next = (g + c)
            return c_next
        else:

            x = self.linear_cell_W_x(x)
            h = self.linear_cell_W_h(h)
            k = self.tanh_cell(x + h)
            g = (i * T * k)
            c = (f * c_prev)
            c_next = (g + c)
            return c_next

    def out_gate(self, x, h, c_prev, delt,shared):
        if shared:
            x = self.linear_output_W_x_shared(x)
            t = self.linear_output_W_t_shared(delt)
            h = self.linear_output_W_h_shared(h)
            c = self.linear_output_w_c_shared(c_prev)
            o = self.sigmoid_out(x + t + h + c)
            return o
        else:
            x = self.linear_output_W_x(x)
            t = self.linear_output_W_t(delt)
            h = self.linear_output_W_h(h)
            c = self.linear_output_w_c(c_prev)
            o = self.sigmoid_out(x + t + h + c)
            return o

    def shared_layer(self,x,tuple_in,delt,shared):
        (h, c_prev) = tuple_in
        i = self.input_gate(x, h, c_prev,shared)
        f = self.forget_gate(x, h, c_prev,shared)
        T = self.time_gate(x, delt,shared)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev, T,shared)
        o = self.out_gate(x, h, c_prev, delt,shared)
        h_next = o * self.activation_final(c_next)

        return h_next, c_next
    def specific_layer(self,x,h_shared,tuple_in,delt,shared):
        (h, c_prev) = tuple_in
        input_concat = torch.cat((x, h_shared), axis=1)

        # x = (x,h_shared)
        i = self.input_gate(input_concat, h, c_prev,shared)
        f = self.forget_gate(input_concat, h, c_prev,shared)
        T = self.time_gate(input_concat, delt,shared)
        c_next = self.cell_memory_gate(i, f, input_concat, h, c_prev, T,shared)
        o = self.out_gate(input_concat, h, c_prev, delt,shared)
        h_next = o * self.activation_final(c_next)
        return h_next, c_next
    def forward(self, x,tuple_shared, tuple_in, delt):
        # x_attention = self.linear_feature_attention_W_x(x)
        # x_attention_sig = self.sigmoid_feature_attention(x_attention)
        # x = self.mutivariate(x)
        h_next_shared, c_next_shared = self.shared_layer(x,tuple_shared,delt,True)
        h_next_shared_attention = self.linear_shared_layer_attention_W_x(h_next_shared)
        h_next_shared_sig = self.sigmoid_shared_layer_attention(h_next_shared_attention)
        h_next, c_next = self.specific_layer(x,h_next_shared_sig, tuple_in, delt,False)
        # h_next2, c_next2 = self.specific_layer(x_attention_sig,h_next_shared_sig, tuple_in2, delt)
        return h_next_shared, c_next_shared,h_next, c_next


