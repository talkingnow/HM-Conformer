#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
import sys
sys.path.append('/exp_lib/egg_exp/framework/model/deepfake_detection/frontend')

"""Encoder definition."""
from typing import Tuple, List, Optional

import torch
import math
# from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution_simple import ConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding

from wenet.transformer.encoder_layer_mp_nonlpe_hieracls22 import ConformerEncoderLayerMP_NonLPE_HieraCLS22

from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling2
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8

from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask


class BaseEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        
        downsample_layer:list = [1, 3],
        layer_cls: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
            use_cls=True,
            layer_cls=layer_cls,
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        
        # add
        self.downsample_layer = downsample_layer
        self.layer_cls = layer_cls
        

    def output_size(self) -> int:
        return self._output_size
    
    def make_cls(self, xs):
        xs = xs[ :, 1:, :]
        cls_token = torch.nn.Parameter(torch.rand(1, 1, xs.size(-1)), requires_grad=True).to(xs.device)
        xs = torch.cat((cls_token.repeat(xs.size(0), 1, 1), xs), dim=1)
        return xs

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor
    ):
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(xs, masks)  

        cls_list = []
        for i in range(len(self.encoders)):
            xs, pos_emb = self.encoders[i](xs, pos_emb)
            if i in self.downsample_layer:
                cls_list.append(xs[ :, 0, :].unsqueeze(1))
                if self.layer_cls:
                    xs = xs[ :, 1:, :]
        
        # after last layer   
        cls_list.append(xs[ :, 0, :].unsqueeze(1))
        xs = xs[ :, 1:, :]
                    
        if self.normalize_before:
            xs = self.after_norm(xs)
        
        cls = torch.cat(cls_list, dim=1)    # (B, n, H)
        
        return xs, cls

class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        
        downsample_layer:list = [1, 3],
        pooling_size:int = 4,
        input_seq_len: int = 401,
        layer_cls: bool = False,
    ):
        # assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk, downsample_layer, layer_cls)
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type == "no_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, True, True) # bias, except_CLS
        
        encoders = []
        seq_len = input_seq_len
        len_cls = len(downsample_layer) + 1
        for i in range(num_blocks):
            if i in downsample_layer: 
                downsample = torch.nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
                seq_len = math.floor( (seq_len - pooling_size) / pooling_size + 1)
            else:
                downsample = None
                seq_len = seq_len
            encoders.append(ConformerEncoderLayerMP_NonLPE_HieraCLS22(
                    output_size,
                    seq_len,
                    downsample,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(
                        *positionwise_layer_args) if macaron_style else None,
                    convolution_layer(
                        *convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    
                    len_cls=len_cls
                    )
                )
            if i in downsample_layer: 
                len_cls -= 1
        
        self.encoders = torch.nn.ModuleList(encoders)
