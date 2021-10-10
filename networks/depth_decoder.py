from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 use_skips=True, cat2end=False, road=False, catxy=False, deep=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.cat2end = cat2end

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            if not deep:
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            else:
                self.convs[("upconv", i, 0)] = nn.Sequential(
                    ConvBlock(num_ch_in, num_ch_in),
                    ConvBlock(num_ch_in, num_ch_out)
                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if road and i in self.scales and self.use_skips:
                num_ch_in += 3
                if catxy:
                    num_ch_in += 3
            num_ch_out = self.num_ch_dec[i]
            if not deep:
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            else:
                self.convs[("upconv", i, 1)] = nn.Sequential(
                    ConvBlock(num_ch_in, num_ch_in),
                    ConvBlock(num_ch_in, num_ch_out)
                )

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        if self.cat2end:
            self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0]+2, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features, two_channel=None, beam_features=None, depth_maps=None, tanh=False):
        #assert not torch.isnan(input_features[-1]).any(), "input_features nan"
        #assert not torch.isnan(beam_features[-1]).any(), "beam_features nan"
        self.outputs = {}

        # decoder
        if beam_features is not None:
            x = input_features[-1] + beam_features[-1]
        else:
            x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                if beam_features is not None:
                    x += [input_features[i - 1] + beam_features[i - 1]]
                else:
                    x += [input_features[i - 1]]
            if depth_maps is not None and i in self.scales and self.use_skips:
                x += [depth_maps[("disp", i)]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if i==0 and self.cat2end:
                    x = torch.cat((x, two_channel), 1)
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                elif tanh:
                    self.outputs[("disp", i)] = torch.tanh(self.convs[("dispconv", i)](x))
                else:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                    # print(self.outputs[("disp", i)])
                    #assert not torch.isnan(self.outputs[("disp", i)]).any(), "output nan"

        return self.outputs
