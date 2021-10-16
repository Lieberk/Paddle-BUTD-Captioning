from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import misc.utils as utils

from .CaptionModel import CaptionModel


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        lengths = att_masks.sum(1)
        pack_sequence = paddle.zeros([lengths.sum(), att_feats.shape[-1]])
        st = 0
        for i, l in enumerate(lengths):
            l = l.astype('int32').item()
            pack_sequence[st:l+st] = att_feats[i][0:l]
            st = l + st

        pack_feats = module(pack_sequence)
        max_att_len = att_feats.shape[1]
        att_size = pack_feats.shape[-1]
        pad_feats = paddle.zeros([att_feats.shape[0], max_att_len, att_size])

        st = 0
        for i, l in enumerate(lengths):
            l = l.astype('int32').item()
            pack_f = pack_feats[st:l+st]
            pz = paddle.zeros([max_att_len-l, att_size])
            pack_f_pz = paddle.concat([pack_f, pz], 0).unsqueeze(0)

            if i == 0:
                pad_feats = pack_f_pz
            else:
                pad_feats = paddle.concat([pad_feats, pack_f_pz], 0)

            st = l + st

        return pad_feats
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length  # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1D(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1D(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab

    def init_hidden(self, bsz):
        x = paddle.zeros([self.num_layers, bsz, self.rnn_size], dtype="float32")
        return (paddle.create_parameter(x.shape,
                                        dtype=str(x.numpy().dtype)),
                paddle.create_parameter(x.shape,
                                        dtype=str(x.numpy().dtype))
                )

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.shape[0]
        seq_per_img = seq.shape[0] // batch_size

        state = self.init_hidden(batch_size * seq_per_img)  # state[i][j]: i=0表示h, i=1表示c, j表示层数

        outputs = paddle.zeros([batch_size * seq_per_img, seq.shape[1] - 1, self.vocab_size + 1])

        if seq_per_img > 1:
            fc_feats, att_feats, att_masks = utils.repeat_tensors(seq_per_img, [fc_feats, att_feats, att_masks])
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(seq.shape[1] - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = paddle.uniform([batch_size * seq_per_img], min=0.0, max=1.0)
                sample_mask = (sample_prob < self.ss_prob).astype('int64')
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero()
                    it = seq[:, i].clone()
                    prob_prev = paddle.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    prob_tensor = paddle.multinomial(prob_prev, 1).index_select(sample_ind).squeeze(-1)
                    for k, ind in enumerate(sample_ind):
                        it[ind] = prob_tensor[k]
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            output = output.unsqueeze(1)
            if i == 0:
                outputs = output
            else:
                outputs = paddle.concat([outputs, output], 1)

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), 1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt=None):

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.shape[0]

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = \
            self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the ' \
                                                 'road. can be dealt with in future if needed '
        seqs = paddle.zeros([batch_size, self.seq_length], dtype='int64')
        seqLogprobs = paddle.zeros([self.seq_length, batch_size])

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand([beam_size, p_fc_feats.shape[1]])
            tmp_att_feats = p_att_feats[k:k + 1].expand([beam_size, p_att_feats.shape[-2], p_att_feats.shape[-1]])
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand([beam_size, pp_att_feats.shape[-2], pp_att_feats.shape[-1]])
            tmp_att_masks = p_att_masks[k:k + 1].expand([
                beam_size, p_att_masks.shape[-1]]) if att_masks is not None else None
            self.done_beams[k] = self.beam_search(state, tmp_fc_feats,
                                                  tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)

            tokens = self.done_beams[k][0]['seq']

            seqs[k, :len(tokens)] = tokens
        return seqs, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = opt.get('sample_n', 1)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.shape[0]
        state = self.init_hidden(batch_size * sample_n)

        if sample_n > 1:
            fc_feats, att_feats, att_masks = utils.repeat_tensors(sample_n, [fc_feats, att_feats, att_masks])
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = paddle.zeros([batch_size * sample_n, self.seq_length], dtype='int64')
        seqLogprobs = paddle.zeros([batch_size * sample_n, self.seq_length])

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = paddle.zeros([batch_size * sample_n], dtype='int32')

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            unfilg = (it > 0).astype('int64')
            if t == 0:
                unfinished = unfilg
            else:
                unfinished = unfinished * unfilg

            seq[:, t] = it
            sampleLogprobs = sampleLogprobs.unsqueeze(1)
            if t == 0:
                seqLogprobs = sampleLogprobs
            else:
                seqLogprobs = paddle.concat([seqLogprobs, sampleLogprobs], 1)

            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        sl = seqLogprobs.shape[-1]
        if sl != self.seq_length:
            seqL_zeros = paddle.zeros([batch_size * sample_n, self.seq_length - sl])
            seqLogprobs = paddle.concat([seqLogprobs, seqL_zeros], 1)

        return seq, seqLogprobs


class Attention(nn.Layer):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.shape[0] // att_feats.shape[-1]
        att = p_att_feats.reshape([-1, att_size, self.att_hid_size])

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = paddle.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.reshape([-1, self.att_hid_size])  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.reshape([-1, att_size])  # batch * att_size

        weight = F.softmax(dot, 1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.reshape([-1, att_size])
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.reshape([-1, att_size, att_feats.shape[-1]])  # batch * att_size * att_feat_size
        att_res = paddle.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res
