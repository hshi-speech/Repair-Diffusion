# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
TFGridNetBlock = layerspp.GridNetV2Block
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        # parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        # parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[0])
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        centered = True,
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = conditional  # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        num_channels = 6  # x.real, x.imag, y.real, y.imag, e_real, e_imag

        # ------------> Neural network construction
        # Output layers
        self.output_layer = nn.Conv2d(num_channels, 2, 1)
        self.output_layer_deter = nn.Conv2d(nf, 2, 1)

        # Embedding layer and condition layer
        self.embedding_layer = layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale)
        embed_dim = 2 * nf

        self.condition = nn.Sequential(
            nn.Linear(embed_dim, nf * 4),
            self.act,
            nn.Linear(nf * 4, nf * 4)
        )
        # initialization for condition layer
        for layer in self.condition:
            if isinstance(layer, nn.Linear):
                layer.weight.data = default_initializer()(layer.weight.shape)
                nn.init.zeros_(layer.bias)

        # Blocks defination
        AttnBlock = functools.partial(layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)

        """
        Upsample   = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Upsample_T = functools.partial(layerspp.Upsample_T, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Upsample_F = functools.partial(layerspp.Upsample_F, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        Downsample   = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Downsample_T = functools.partial(layerspp.Downsample_T, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        Downsample_F = functools.partial(layerspp.Downsample_F, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        """

        ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act, dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)


        # Downsampling block
        channels = num_channels
        input_pyramid_ch = channels

        # Input layer
        self.input_layer_a = conv3x3(channels, nf//2)
        self.input_layer_b = conv3x3(4, nf//2)
        hs_c = [nf]
        in_ch = nf

        # Network construction

        # ================================================= #
        # ============== Shared Encoder =================== #
        # ================================================= #
        # -------> shared encoder [1]: 2x128-128 + down
        # -------> [256, t] to [128, t]
        # in_ch = nf = 128
        self.shared_encoder_a_1 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )
        self.shared_encoder_b_1 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )

        # -------> shared encoder [2]: 2x128-128 + down
        # -------> [128, t] to [64, t]
        self.shared_encoder_a_2 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )
        self.shared_encoder_b_2 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )

        # -------> shared encoder [3]: 1x128-128 + down
        # -------> [64, t] to [32, t]
        self.shared_encoder_a_3 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )
        self.shared_encoder_b_3 = nn.Sequential(
            ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
            ResnetBlock(down=True, down_f=True, in_ch=in_ch//2, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        )

        # ================================================= #
        # =============== Score Encoder =================== #
        # ================================================= #

        # -------> score-encoder [1]: 1x128-128 + down
        # -------> [32, t] to [32, t/2]
        self.score_encoder_a_1 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
            ResnetBlock(down=True, down_t=True, in_ch=in_ch),
            # nn.GLU(dim=1)
        )
        # self.score_encoder_b_1 = nn.Sequential(
        #     ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
        #     ResnetBlock(down=True, down_t=True, in_ch=in_ch, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        # )

        # -------> score-encoder [2]: 1x128-128 + down
        # -------> [32, t/2] to [16, t/4]
        self.score_encoder_a_2 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
            ResnetBlock(down=True, in_ch=in_ch),
            # nn.GLU(dim=1)
        ) 
        # self.score_encoder_b_2 = nn.Sequential(
        #     ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
        #     ResnetBlock(down=True, in_ch=in_ch, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        # )

        # -------> score-encoder [3]: 1x128-128 + attention + down
        # -------> [16, t/4] to [8, t/8]
        self.score_encoder_a_3 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
            ResnetBlock(down=True, in_ch=in_ch),
            # nn.GLU(dim=1)
        )
        # self.score_encoder_b_3 = nn.Sequential(
        #     ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
        #     ResnetBlock(down=True, in_ch=in_ch, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        # )

        # -------> score-encoder [4]: 1x128-128 + down
        # -------> [8, t/8] to [8, t/16]
        self.score_encoder_a_4 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
            AttnBlock(channels=in_ch),
            ResnetBlock(down=True, down_t=True, in_ch=in_ch),
            # nn.GLU(dim=1)
        )
        # self.score_encoder_b_4 = nn.Sequential(
        #     ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
        #     AttnBlock(channels=in_ch//2),
        #     ResnetBlock(down=True, down_t=True, in_ch=in_ch, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        # )

        # -------> score-encoder [5]: 1x128-128 + down
        # -------> [8, t/16] to [4, t/32]
        self.score_encoder_a_5 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
            ResnetBlock(down=True, in_ch=in_ch),
            # nn.GLU(dim=1)
        )
        # self.score_encoder_b_5 = nn.Sequential(
        #     ResnetBlock(in_ch=in_ch//2, out_ch=in_ch//2),
        #     ResnetBlock(down=True, in_ch=in_ch, out_ch=in_ch//2),
            # nn.GLU(dim=1)
        # )

        # ================================================= #
        # ============== Embedding Block ================== #
        # ================================================= #

        # -------> score embedding processing
        self.score_embedding = nn.Sequential(
            ResnetBlock(in_ch=in_ch),
            AttnBlock(channels=in_ch),
            ResnetBlock(in_ch=in_ch)
        )

        # -------> deterministic embedding processing
        self.deter_embedding = nn.Sequential(
             TFGridNetBlock(
                      channel_dim=in_ch//2,
                      emb_dim=48,
                      emb_ks=3,
                      emb_hs=1,
                      n_freqs=4,
                      hidden_channels=48
             ),
             TFGridNetBlock(
                      channel_dim=in_ch,
                      emb_dim=48,
                      emb_ks=3,
                      emb_hs=1,
                      n_freqs=4,
                      hidden_channels=48
             ),
             # ResnetBlock(in_ch=in_ch, out_ch=in_ch),
             # nn.GLU(dim=1),
             # ResnetBlock(in_ch=in_ch, out_ch=in_ch),
             # nn.GLU(dim=1)
        )

        # ================================================= #
        # ============ Deterministic Decoder ============== #
        # ================================================= #
        self.deter_decoder_p_3 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2),
        )

        self.deter_decoder_p_2 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2),
        )

        self.deter_decoder_p_1 = nn.Sequential(
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch//2),
            ResnetBlock(in_ch=in_ch, out_ch=in_ch),
        )

        # ================================================= #
        # =============== Score Decoder =================== #
        # ================================================= #
        self.score_decoder_p_8 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_7 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True, up_t=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_6 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_5 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_4 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True, up_t=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_3 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_2 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_1 = nn.Sequential(
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch, up=True, up_f=True),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
            ResnetBlock(in_ch=in_ch*2, out_ch=in_ch),
        )

        self.score_decoder_p_0 = nn.Sequential(
           nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6),
           conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
        )

    def forward(self, x, time_cond, w_training=0):
        # timestep/noise_level embedding; only for continuous training
        # Convert real and imaginary parts of (x,y) into four channel dimensions
        # noisy_real = x[:,[1],:,:].real
        # noisy_imag = x[:,[1],:,:].imag

        x_a = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                x[:,[1],:,:].real, x[:,[1],:,:].imag,
                x[:,[2],:,:].real, x[:,[2],:,:].imag), dim=1)

        x_b = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
            x[:,[1],:,:].real, x[:,[1],:,:].imag,), dim=1)

        # -----------> Processing begins
        used_sigmas = time_cond
        temb = self.embedding_layer(torch.log(used_sigmas))
        temb = self.condition(temb)

        # Downsampling block
        input_pyramid = x_a
        # Input layer: Conv2d: 6ch -> 128ch
        hs_a = [self.input_layer_a(x_a)]
        hs_b = [self.input_layer_b(x_b)]
        hs_deter = [] # This is for deterministic output: pop
        hs_deter.append(hs_a[-1])

        # ================================================= #
        # ============== Shared Encoder =================== #
        # ================================================= #
        # Processing of shared encoder
        # ----> first shared layer
        h_a = self.shared_encoder_a_1[0](hs_a[-1], temb)
        hs_a.append(h_a)
        hs_deter.append(h_a)
        h_a = self.shared_encoder_a_1[1](h_a, temb)
        hs_a.append(h_a)
        hs_deter.append(h_a)
        
        h_b = self.shared_encoder_b_1[0](hs_b[-1], temb)
        hs_b.append(h_b)
        h_b = self.shared_encoder_b_1[1](h_b, temb)
        hs_b.append(h_b)

        # h = torch.cat([h_a, h_b], dim=1)
        
        h_a = self.shared_encoder_a_1[2](h_a, temb)
        # h_a = self.shared_encoder_a_1[3](h_a)
        hs_a.append(h_a)
        hs_deter.append(h_a)

        h_b = self.shared_encoder_b_1[2](h_b, temb)
        # h_b = self.shared_encoder_b_1[3](h_b)
        hs_b.append(h_b)

        # ----> second shared layer
        h_a = self.shared_encoder_a_2[0](h_a, temb)
        hs_a.append(h_a)
        hs_deter.append(h_a)
        h_a = self.shared_encoder_a_2[1](h_a, temb)
        hs_a.append(h_a)
        hs_deter.append(h_a)

        h_b = self.shared_encoder_b_2[0](h_b, temb)
        hs_b.append(h_b)
        h_b = self.shared_encoder_b_2[1](h_b, temb)
        hs_b.append(h_b)

        # h = torch.cat([h_a, h_b], dim=1)

        h_a = self.shared_encoder_a_2[2](h_a, temb)
        # h_a = self.shared_encoder_a_2[3](h_a)
        hs_a.append(h_a)
        hs_deter.append(h_a)

        h_b = self.shared_encoder_b_2[2](h_b, temb)
        # h_b = self.shared_encoder_b_2[3](h_b)
        hs_b.append(h_b)

        # ----> third shared layer
        h_a = self.shared_encoder_a_3[0](h_a, temb)
        hs_a.append(h_a)
        hs_deter.append(h_a)

        h_b = self.shared_encoder_b_3[0](h_b, temb)
        hs_b.append(h_b)

        # h = torch.cat([h_a, h_b], dim=1)

        h_a = self.shared_encoder_a_3[1](h_a, temb)
        # h_a = self.shared_encoder_a_3[2](h_a)
        hs_a.append(h_a)
        hs_deter.append(h_a)
        h_b = self.shared_encoder_b_3[1](h_b, temb)
        # h_b = self.shared_encoder_b_3[2](h_b)
        hs_b.append(h_b)

        # ================================================= #
        # =========== Deterministic Embedding ============= #
        # ================================================= #
        # h = torch.cat([h_a, h_b], dim=1)
        h_deter = self.deter_embedding[0](h_b)
        h_a = torch.cat([h_a, h_deter], dim=1)
        h_a = self.deter_embedding[1](h_a)
        hs_a.append(h_a)
        # hs_deter.append(h_a)
        # hs_b.append(h_b)

        # ================================================= #
        # =============== Score Encoder =================== #
        # ================================================= #
        # ----> first *score-encoder* layer
        h_a = self.score_encoder_a_1[0](h_a, temb)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_1[0](h_b, temb)
        # hs_b.append(h_b)
        # h = torch.cat([h_a, h_b], dim=1)
        h_a = self.score_encoder_a_1[1](h_a, temb)
        # h_a = self.score_encoder_a_1[2](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_1[1](h, temb)
        # h_b = self.score_encoder_b_1[2](h_b)
        # hs_b.append(h_b)

        # ----> second *score-encoder* layer
        h_a = self.score_encoder_a_2[0](h_a, temb)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_2[0](h_b, temb)
        # hs_b.append(h_b)
        # h = torch.cat([h_a, h_b], dim=1)
        h_a = self.score_encoder_a_2[1](h_a, temb)
        # h_a = self.score_encoder_a_2[2](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_2[1](h, temb)
        # h_b = self.score_encoder_b_2[2](h_b)
        # hs_b.append(h_b)

        # ----> third *score-encoder* layer
        h_a = self.score_encoder_a_3[0](h_a, temb)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_3[0](h_b, temb)
        # hs_b.append(h_b)
        # h = torch.cat([h_a, h_b], dim=1)
        h_a = self.score_encoder_a_3[1](h_a, temb)
        # h_a = self.score_encoder_a_3[2](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_3[1](h, temb)
        # h_b = self.score_encoder_b_3[2](h_b)
        # hs_b.append(h_b)

        # ----> forth *score-encoder* layer
        h_a = self.score_encoder_a_4[0](h_a, temb)
        h_a = self.score_encoder_a_4[1](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_4[0](h_b, temb)
        # h_b = self.score_encoder_b_4[1](h_b)
        # hs_b.append(h_b)
        # h = torch.cat([h_a, h_b], dim=1)
        h_a = self.score_encoder_a_4[2](h_a, temb)
        # h_a = self.score_encoder_a_4[3](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_4[2](h, temb)
        # h_b = self.score_encoder_b_4[3](h_b)
        # hs_b.append(h_b)

        # ----> fifth *score-encoder* layer
        h_a = self.score_encoder_a_5[0](h_a, temb)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_5[0](h_b, temb)
        # hs_b.append(h_b)
        # h = torch.cat([h_a, h_b], dim=1)
        h_a = self.score_encoder_a_5[1](h_a, temb)
        # h_a = self.score_encoder_a_5[2](h_a)
        hs_a.append(h_a)
        # h_b = self.score_encoder_b_5[1](h, temb)
        # h_b = self.score_encoder_b_5[2](h_b)
        # hs_b.append(h_b)

        # ================================================= #
        # ============== Score Embedding ================== #
        # ================================================= #
        # -----> EMBEDDING PROCESSING
        # h = torch.cat([h_a, h_b], dim=1)
        h = self.score_embedding[0](h_a, temb)
        h = self.score_embedding[1](h)
        h = self.score_embedding[2](h, temb)

        # ================================================= #
        # ============ Deterministic Decoder ============== #
        # ================================================= #
        # ----> third *deter-decoder* layer
        # h_deter = hs_deter.pop()
        h_deter = self.deter_decoder_p_3[0](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_3[1](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)

        # ----> second *deter-decoder* layer
        h_deter = self.deter_decoder_p_2[0](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_2[1](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_2[2](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)

        # ----> first *deter-decoder* layer
        h_deter = self.deter_decoder_p_1[0](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_1[1](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_1[2](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)
        h_deter = self.deter_decoder_p_1[3](torch.cat([h_deter, hs_deter.pop()], dim=1), temb)

        assert not hs_deter

        # ================================================= #
        # ================ Score Decoder ================== #
        # ================================================= #
        h = self.score_decoder_p_8[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_8[1](torch.cat([h, hs_a.pop()], dim=1), temb)

        h = self.score_decoder_p_7[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_7[1](torch.cat([h, hs_a.pop()], dim=1), temb)

        h = self.score_decoder_p_6[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_6[1](torch.cat([h, hs_a.pop()], dim=1), temb)

        h = self.score_decoder_p_5[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_5[1](torch.cat([h, hs_a.pop()], dim=1), temb)

        h = self.score_decoder_p_4[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_4[1](torch.cat([h, hs_a.pop()], dim=1), temb)

        h = self.score_decoder_p_3[0](torch.cat([h, hs_a.pop()], dim=1), temb)
        h = self.score_decoder_p_3[1](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_3[2](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)


        h = self.score_decoder_p_2[0](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_2[1](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_2[2](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)

        h = self.score_decoder_p_1[0](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_1[1](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_1[2](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)
        h = self.score_decoder_p_1[3](torch.cat([h, hs_a.pop(), hs_b.pop()], dim=1), temb)

        assert not hs_a
        assert not hs_b

        h = self.act(self.score_decoder_p_0[0](h))
        h = self.score_decoder_p_0[1](h)

        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)
        h_deter = self.output_layer_deter(h_deter)
        
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]

        h_deter = torch.permute(h_deter, (0, 2, 3, 1)).contiguous()
        h_deter = torch.view_as_complex(h_deter)[:,None, :, :]

        if(w_training):
            return h, h_deter
        else:
            return h
