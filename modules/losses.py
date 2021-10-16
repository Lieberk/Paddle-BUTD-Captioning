import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.reshape([-1])
        reward = reward.reshape([-1])
        mask = (seq > 0).astype('float32')
        fill_ones = paddle.ones([mask.shape[0], 1], dtype='float32')
        mask = paddle.concat([fill_ones, mask[:, :-1]], 1).reshape([-1])
        output = - input * reward * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output


class LanguageModelCriterion(nn.Layer):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]
        target = F.one_hot(target, input.shape[-1])
        output = -input.multiply(target).sum(-1) * mask
        output = paddle.sum(output) / paddle.sum(mask)
        return output
