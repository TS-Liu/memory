# -*- coding: utf-8 -*-
"""
__AUTHOR__: Alanili
__EMAIL__: waajoenglei@gmail.com
"""
from onmt.modules import attention
from onmt.modules import layers
from onmt.Models import EncoderBase
from onmt.Models import DecoderState
from onmt.Utils import aeq

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = onmt.modules.LayerNorm(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class EncoderLayer(nn.Module):
    """
    sigle encoder layer used for Encoder
    """

    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma = attention.MultiheadAttention(hidden_size,
                                               hidden_size,
                                               hidden_size,
                                               dropout)
        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_prenorm = layers.LayerNorm(hidden_size)
        self.ffn_prenorm = layers.LayerNorm(hidden_size)
        self.ma_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, bias):
        """
        Args:
            x: current input Tensor with shape [batch, length, channels]
            bias: self attention bias with shape [batch, 1, length]
            pv: previous value
        Returns:
            the transformed result with shape as x
        """
        # multihead attention
        y, _ = self.ma(self.ma_prenorm(x), None,
                       self.num_heads, bias)
        x = self.ma_postdropout(y) + x
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans


class DecoderLayer(EncoderBase):

    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma_l1 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ma_l2 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)

        # self.ma_m1 = attention.Memory_MultiheadAttention(hidden_size,
        #                                                  hidden_size,
        #                                                  hidden_size,
        #                                                  dropout)
        # self.ma_m2 = attention.Memory_MultiheadAttention(hidden_size,
        #                                                  hidden_size,
        #                                                  hidden_size,
        #                                                  dropout)
        # self.ma_m3 = attention.Memory_MultiheadAttention(hidden_size * 2,
        #                                                  hidden_size,
        #                                                  hidden_size,
        #                                                  dropout)

        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_l1_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l2_prenorm = layers.LayerNorm(hidden_size)

        # self.ma_m1_prenorm = layers.LayerNorm(hidden_size)
        # self.ma_m2_prenorm = layers.LayerNorm(hidden_size)
        # self.ma_m3_prenorm = layers.LayerNorm(hidden_size * 2)

        self.ffn_prenorm = layers.LayerNorm(hidden_size)

        self.ma_l1_postdropout = nn.Dropout(dropout)
        self.ma_l2_postdropout = nn.Dropout(dropout)

        # self.ma_m1_postdropout = nn.Dropout(dropout)
        # self.ma_m2_postdropout = nn.Dropout(dropout)
        # self.ma_m3_postdropout = nn.Dropout(dropout)

        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_bias,
                encoder_decoder_bias, previous_input=None):
        # self multihead attention
        norm_x = self.ma_l1_prenorm(x)
        all_inputs = norm_x
        if previous_input is not None:
            all_inputs = torch.cat((previous_input, norm_x), dim=1)
            self_attention_bias = None
        y, _ = self.ma_l1(norm_x, all_inputs, self.num_heads, self_attention_bias)
        x = self.ma_l1_postdropout(y) + x
        # encoder decoder multihead attention
        y, attn = self.ma_l2(self.ma_l2_prenorm(x), encoder_output,
                             self.num_heads, encoder_decoder_bias)
        x = self.ma_l2_postdropout(y) + x

        # # memory
        # s_norm_x = self.ma_m1_prenorm(outputs_m)
        #
        # s_y, s_ = self.ma_m1(s_norm_x, outputs_memory, outputs_memory, self.num_heads, esc_bias)
        # s_x = self.ma_m1_postdropout(s_y) + outputs_m
        #
        # t_norm_x = self.ma_m1_prenorm(outputt_m)
        #
        # t_y, t_ = self.ma_m2(t_norm_x, outputt_memory, outputt_memory, self.num_heads, etc_bias)
        # t_x = self.ma_m1_postdropout(t_y) + outputt_m
        #
        # t_x = torch.cat((s_x, t_x), dim=4)
        # outputs = torch.cat((x, x), dim=2)
        # # encoder decoder multihead attention
        # y, attn = self.ma_m3(self.ma_m3_prenorm(outputs), t_x, outputt_m,
        #                      self.num_heads, et_bias)
        # o = self.ma_m3_postdropout(y) + x.unsqueeze(2).unsqueeze(2)
        #
        # b, l, d = x.size()
        # x = o.view(b, l, d)

        # ffn layer
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x

        return ans, attn, all_inputs


class MemoryLayer(EncoderBase):

    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(MemoryLayer, self).__init__()
        self.num_heads = num_heads
        self.ma_l1 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ma_l2 = attention.MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)

        self.ffn = layers.ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_l1_prenorm = layers.LayerNorm(hidden_size)
        self.ma_l2_prenorm = layers.LayerNorm(hidden_size)

        self.ffn_prenorm = layers.LayerNorm(hidden_size)

        self.ma_l1_postdropout = nn.Dropout(dropout)
        self.ma_l2_postdropout = nn.Dropout(dropout)

        self.ffn_postdropout = nn.Dropout(dropout)

        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.u1 = nn.Linear(hidden_size, hidden_size)
        self.v1 = nn.Linear(hidden_size, hidden_size)

        self.Tanh = nn.Tanh()

        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.s = nn.Sigmoid()

    def forward(self, output, outputs, outputs_m, outputt_m, src_memory_bank, encoder_decoder_bias, sc_bias, s_bias):
        # self multihead attention
        batch, lens, dim = src_memory_bank.size()

        outputs = outputs.view(batch, lens, 1, dim).view(batch * lens, 1, dim)
        outputs_m = outputs_m.view(batch, lens, 7, dim).view(batch * lens, 7, dim)
        norm_x = self.ma_l1_prenorm(outputs)
        y, _ = self.ma_l1(norm_x, outputs_m, self.num_heads, sc_bias)
        x = self.ma_l1_postdropout(y) + outputs
        s_x = x.view(batch, lens, dim)

        src_memory_bank = src_memory_bank.unsqueeze(2).repeat(1, 1, 2, 1).view(batch, lens * 2, dim)
        s_x = s_x.unsqueeze(2).repeat(1, 1, 2, 1).view(batch, lens*2, dim)

        s_t_m = self.Tanh(self.w1(src_memory_bank)+self.u1(s_x)+self.v1(outputt_m))
        #s_t_m = torch.cat((outputt_m, s_x), dim=2)

        s_t_norm_x = self.ma_l2_prenorm(output)
        s_t_y, s_t_ = self.ma_l2(s_t_norm_x, s_t_m, self.num_heads, s_bias)
        s_t_x = self.ma_l2_postdropout(s_t_y)

        y = self.ffn(self.ffn_prenorm(s_t_x))
        output_h = self.ffn_postdropout(y) + s_t_x

        B = self.s(self.w(output) + self.u(output_h))
        ans = (1 - B) * output + B * output_h




        return ans, s_t_, B




class TransformerEncoder(EncoderBase):
    """
    encoder of transformer
    """

    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, dropout) for _ in range(num_layers)])
        self.layer_norm = layers.LayerNorm(hidden_size)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).float()
        bias = Variable(torch.unsqueeze(mask * -1e9, 1))
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.layer_stack[i](out, bias)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()


class TransformerDecoder(nn.Module):
    """
    decoder of transformer
    """

    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings, src_embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.src_embeddings = src_embeddings

        self.layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, dropout) for _ in range(num_layers)])

        self.memory = MemoryLayer(hidden_size, dropout)

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = layers.LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, train=False,
                memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        src = state.src
        src_m = state.src_m
        tgt_m = state.tgt_m
        src_len, src_batch, _ = src.size()
        # if base:
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_m_words = src_m[:, :, 0].transpose(0, 1)
        tgt_m_words = tgt_m[:, :, 0].transpose(0, 1)

        # else:
        #     tgt_m_words = tgt_m.transpose(0, 2).contiguous().view(src_batch, src_len, 2, 1).transpose(2, 3)


        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
            #tgt_m = torch.cat([state.previous_tm_input, tgt_m], 0)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        _, __, tgt_embedding_dim = emb.size()
        embt_m = self.embeddings(tgt_m)
        embs_m = self.src_embeddings(src_m)
        embs = self.src_embeddings(src)

        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
            #embt_m = embt_m[state.previous_tm_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        outputt_m = embt_m.transpose(0, 1).contiguous().view(src_batch, src_len*2, tgt_embedding_dim)
        outputs_m = embs_m.transpose(0, 1).contiguous().view(src_batch, src_len*7, tgt_embedding_dim)
        outputs = embs.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx

        src_pad_mask = Variable(src_words.data.eq(padding_idx).float())

        src_m_pad_mask = Variable(src_m_words.data.eq(padding_idx).float()).view(src_batch, src_len, 7).view(src_batch*src_len, 7)

        tgt_m_pad_mask = Variable(tgt_m_words.data.eq(padding_idx).float())

        sc_bias = torch.unsqueeze(src_m_pad_mask * -1e9, 1)
        s_bias = torch.unsqueeze(tgt_m_pad_mask * -1e9, 1)

        tgt_pad_mask = Variable(tgt_words.data.eq(padding_idx).float().unsqueeze(1))
        tgt_pad_mask = tgt_pad_mask.repeat(1, tgt_len, 1)
        encoder_decoder_bias = torch.unsqueeze(src_pad_mask * -1e9, 1)
        decoder_local_mask = attention.get_local_mask(tgt_len)  # [1, length, length]
        decoder_local_mask = decoder_local_mask.repeat(tgt_batch, 1, 1)
        decoder_bias = torch.gt(tgt_pad_mask + decoder_local_mask, 0).float() * -1e9

        saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.layer_stack[i](output, src_memory_bank, decoder_bias, encoder_decoder_bias,
                                      previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        output, attn, B = self.memory(output, outputs, outputs_m, outputt_m, src_memory_bank, encoder_decoder_bias, sc_bias, s_bias)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        B = B.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns, B

    def init_decoder_state(self, src, src_m, tgt_m):
        return TransformerDecoderState(src, src_m, tgt_m)


class TransformerDecoderState(DecoderState):
    def __init__(self, src, src_m, tgt_m):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.src_m = src_m
        self.tgt_m = tgt_m
        self.previous_input = None
        self.previous_layer_inputs = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.previous_layer_inputs, self.src)

    def update_state(self, input, previous_layer_inputs):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src, self.src_m, self.tgt_m)
        state.previous_input = input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
        self.src_m = Variable(self.src_m.data.repeat(1, beam_size, 1),
                            volatile=True)
        self.tgt_m = Variable(self.tgt_m.data.repeat(1, beam_size, 1),
                              volatile=True)
