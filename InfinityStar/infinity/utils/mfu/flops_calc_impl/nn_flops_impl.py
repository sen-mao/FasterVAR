# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import torch.nn as nn

def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    gates_size = w_ih.shape[0]
    # matrix matrix mult ih state and internal state
    flops += 2 * w_ih.shape[0] * w_ih.shape[1] - gates_size
    # matrix matrix mult hh state and internal state
    flops += 2 * w_hh.shape[0] * w_hh.shape[1] - gates_size
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: rnn_forward_hook,
    nn.GRU: rnn_forward_hook,
    nn.LSTM: rnn_forward_hook,
    nn.RNNCell: rnn_cell_forward_hook,
    nn.LSTMCell: rnn_cell_forward_hook,
    nn.GRUCell: rnn_cell_forward_hook,
}

