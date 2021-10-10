from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import time
import torch.nn as nn
import MinkowskiEngine as ME
from collections import OrderedDict
from layers import *
import pdb


class RefineNet(nn.Module):
    def __init__(self, num_ch_enc, scales=[0], num_output_channels=1, batch_size=12, use_skips=True, cat_other=False):
        super(RefineNet, self).__init__()

        self.batch_size = batch_size
        self.cat_other = cat_other
        self.in_ch = 4
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([4, 4, 6, 8, 16])

        # decoder
        self.convs = OrderedDict()

        for i in range(5):
            num_ch_in = self.in_ch if i == 0 else self.num_ch_dec[i-1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("conv", i)] = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=num_ch_in,
                    out_channels=num_ch_out,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=3),
                ME.MinkowskiBatchNorm(num_ch_out),
                ME.MinkowskiReLU(),
                ME.MinkowskiMaxPooling(2, 2, dimension=3)
            )

        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_dec[i]
            if i < 4:
                num_ch_in += self.num_ch_dec[i]
            if self.cat_other:
                num_ch_in += self.num_ch_enc[i]
            num_ch_out = self.in_ch if i == 0 else self.num_ch_dec[i-1]
            self.convs[("upconv", i)] = nn.Sequential(
                ME.MinkowskiConvolutionTranspose(
                    in_channels=num_ch_in,
                    out_channels=num_ch_out,
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    bias=False,
                    dimension=3),
                ME.MinkowskiBatchNorm(num_ch_out),
                ME.MinkowskiReLU())

        for s in self.scales:
            num_ch_in = self.in_ch if s == 0 else self.num_ch_dec[s - 1]
            self.convs[("dispconv", s)] = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=num_ch_in,
                    out_channels=self.num_output_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=True,
                    dimension=3))

        self.refine_net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_depth, batch_size, input_features=None):
        #print(input_depth.coordinates.shape)
        self.features= []
        # encoder
        x = input_depth
        for i in range(5):
            x = self.convs["conv", i](x)
            self.features.append(x)

        # decoder
        if self.cat_other:
            x = ME.cat(x, input_features[-1])
        x = self.convs[("upconv", 4)](x)
        for i in range(3, -1, -1):
            if self.cat_other:
                x = ME.cat(x, self.features[i], input_features[i])
            else:
                x = ME.cat(x, self.features[i])
            x = self.convs[("upconv", i)](x)

            if i in self.scales:
                st = self.convs[("dispconv", i)](x)
                w = int(640 / (2 ** i))
                h = int(192 / (2 ** i))
                disp = torch.zeros([batch_size, 1, h, w]).cuda()
                for batch in range(batch_size):
                    feats = st.features_at(batch_index=batch)
                    disp[batch, 0] = feats[:(w*h)].view([h, w])
                offset = torch.tanh(disp)


        return offset


class RefineNet_shallow(nn.Module):
    def __init__(self, num_output_channels=1, batch_size=12):
        super(RefineNet_shallow, self).__init__()

        self.batch_size = batch_size
        self.in_ch = 5
        self.mid_ch = 16
        self.num_output_channels = num_output_channels

        self.convs = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=self.in_ch,
                out_channels=self.mid_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.mid_ch),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=self.mid_ch,
                out_channels=self.mid_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.mid_ch),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=self.mid_ch,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=True,
                dimension=3)
        )


    def forward(self, input_depth, batch_size):

        # encoder
        #start_time = time.time()
        st = self.convs(input_depth)
        #inf_time = time.time()
        #print("### inf time: ", inf_time - start_time)

        w = int(640)
        h = int(192)
        disp = torch.zeros([batch_size, 1, 192, 640]).cuda()
        #print("### alloc time: ", time.time() - inf_time)
        for batch in range(batch_size):
            feats = st.features_at(batch_index=batch)
            disp[batch, 0] = feats[:(w*h)].view([h, w])
        offset = torch.tanh(disp)
        #print("### collect time: ", time.time() - inf_time)
        return offset


class RefineNet_deep(nn.Module):
    def __init__(self, num_output_channels=1, batch_size=12):
        super(RefineNet_deep, self).__init__()

        self.batch_size = batch_size
        self.in_ch = 4
        self.mid_ch = 16
        self.deep_ch = 64
        self.num_output_channels = num_output_channels

        self.convs = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=self.in_ch,
                out_channels=self.mid_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.mid_ch),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=self.mid_ch,
                out_channels=self.mid_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.mid_ch),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=self.mid_ch,
                out_channels=self.deep_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.deep_ch),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=self.deep_ch,
                out_channels=self.deep_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.deep_ch),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=self.deep_ch,
                out_channels=self.mid_ch,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(self.mid_ch),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=self.mid_ch,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=True,
                dimension=3)
        )


    def forward(self, input_depth, batch_size):

        # encoder
        #start_time = time.time()
        st = self.convs(input_depth)
        #inf_time = time.time()
        #print("### inf time: ", inf_time - start_time)

        w = int(640)
        h = int(192)
        disp = torch.zeros([batch_size, 1, 192, 640]).cuda()
        #print("### alloc time: ", time.time() - inf_time)
        for batch in range(batch_size):
            feats = st.features_at(batch_index=batch)
            disp[batch, 0] = feats[:(w*h)].view([h, w])
        offset = torch.tanh(disp)
        #print("### collect time: ", time.time() - inf_time)
        return offset
