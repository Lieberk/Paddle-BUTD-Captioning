# [2018]《Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering》
import paddle
import paddle.nn as nn
from .AttModel import AttModel, Attention
import paddle.nn.functional as F


class UpDownCore(nn.Layer):
    def __init__(self, opt):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.attention = Attention(opt)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = paddle.concat([prev_h, fc_feats, xt], 1)

        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = paddle.concat([att, h_att], 1)

        _, (h_lang, c_lang) = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, 1, self.training)
        state = (paddle.stack([h_att, h_lang]), paddle.stack([c_att, c_lang]))

        return output, state


class UpDownModel(AttModel):
    def __init__(self, opt):
        super(UpDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = UpDownCore(opt)
