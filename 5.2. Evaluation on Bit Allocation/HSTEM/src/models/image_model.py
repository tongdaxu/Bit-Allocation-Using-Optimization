# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

from ..layers.layers import conv3x3

from .common_model import CompressionModel
from .video_net import LowerBound, UNet, get_enc_dec_models, get_hyper_enc_dec_models
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize, \
    get_rounded_q, get_state_dict


class IntraNoAR(CompressionModel):
    def __init__(self, N=192, anchor_num=4):
        super().__init__(y_distribution='gaussian', z_channel=N)

        self.enc, self.dec = get_enc_dec_models(3, 16, N)
        self.refine = nn.Sequential(
            UNet(16, 16),
            conv3x3(16, 3),
        )
        self.hyper_enc, self.hyper_dec = get_hyper_enc_dec_models(N, N)
        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(N * 2, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(N * 4, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N * 3, N * 2, 3, padding=1)
        )

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        # self.q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        # the exact q_step is q_basic * q_scale
        self.N = int(N)
        self.anchor_num = int(anchor_num)

        # self._initialize_weights()

    def get_curr_q(self, q_scale, q_basic):
        q_basic = LowerBound.apply(q_basic, 0.5)
        return q_basic * q_scale

    def forward(self, x, stage, mode, y_=None, z=None, delta_I=None, it=0, total_it=0, I_q_basic=None, force_sga=False):
        if stage == "test_for_first":
            I_q_basic = self.q_basic if I_q_basic is None else I_q_basic
            curr_q = self.get_curr_q(delta_I, I_q_basic)

            y_ = self.enc(x)
            y = y_ / curr_q
            z = self.hyper_enc(y)
            z_hat = self.quant(z, mode)

            params = self.hyper_dec(z_hat)
            q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
            y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
                y, means, scales, q_step, self.y_spatial_prior, mode, it, total_it, force_sga)

            y_hat_ = y_hat * curr_q
            x_hat = self.refine(self.dec(y_hat_))

            y_for_bit = y_q
            z_for_bit = z_hat
            bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
            bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
            mse = self.mse(x, x_hat)

            B, _, H, W = x.size()
            pixel_num = B * H * W
            bpp_y = torch.sum(bits_y) / pixel_num
            bpp_z = torch.sum(bits_z) / pixel_num
            bpp = bpp_y + bpp_z
            bits = (bpp_y + bpp_z) * pixel_num

            return {
                "x_hat": x_hat,
                "mse": mse,
                "bit": bits.item(),
                "bpp": bpp,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "delta_I": delta_I,
                "I_q_basic": I_q_basic,
                "y_": y_,
                "z": z,
            }

        elif stage == "finetune":
            I_q_basic = self.q_basic if I_q_basic is None else I_q_basic
            curr_q = self.get_curr_q(delta_I, I_q_basic)

            y = y_ / curr_q

            z_hat = self.SGA(z, mode, it, total_it)

            params = self.hyper_dec(z_hat)
            q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
            y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
                y, means, scales, q_step, self.y_spatial_prior, mode, it, total_it, force_sga)

            y_hat = y_hat * curr_q
            x_hat = self.refine(self.dec(y_hat))

            y_for_bit = y_q
            z_for_bit = z_hat
            bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
            bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
            mse = self.mse(x, x_hat)

            B, _, H, W = x.size()
            pixel_num = B * H * W
            bpp_y = torch.sum(bits_y) / pixel_num
            bpp_z = torch.sum(bits_z) / pixel_num
            bpp = bpp_y + bpp_z
            bits = (bpp_y + bpp_z) * pixel_num

            return {
                "x_hat": x_hat,
                "mse": mse,
                "bit": bits.item(),
                "bpp": bpp,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "delta_I": delta_I,
                "I_q_basic": I_q_basic,
            }

        else:
            print("No this stage")

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scales = ckpt["q_scale"]
        return q_scales.reshape(-1)

    def encode_decode(self, x, delta_I, output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            return self.forward(x, delta_I)

        assert pic_height is not None
        assert pic_width is not None
        q_scale, q_index = get_rounded_q(delta_I)
        compressed = self.compress(x, q_scale)
        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_index, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width, q_index / 100)
        x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result

    def compress(self, x, delta_I):
        delta_I = self.get_curr_q(delta_I, self.q_basic)

        y = self.enc(x)
        y = y / delta_I
        z = self.hyper_enc(y)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat_ = y_hat * delta_I

        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        bit_stream = self.entropy_coder.flush_encoder()

        result = {
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, bit_stream, height, width, delta_I):
        curr_q = self.get_curr_q(delta_I, self.q_basic)

        self.entropy_coder.set_stream(bit_stream)
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat = self.decompress_dual_prior(means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.refine(self.dec(y_hat)).clamp_(0, 1)
        return {"x_hat": x_hat}

    def compress_with_y(self, y_=None, z=None, delta_I=None):
        curr_q = self.get_curr_q(delta_I, self.q_basic)

        y = y_ / curr_q
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat_ = y_hat * curr_q

        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        bit_stream = self.entropy_coder.flush_encoder()

        result = {
            "bit_stream": bit_stream,
        }
        return result

    def encode_decode_with_y(self, y_=None, z=None, delta_I=None, output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            print("output path is None")
            return 0

        assert pic_height is not None
        assert pic_width is not None
        q_scale, q_index = get_rounded_q(delta_I)
        compressed = self.compress_with_y(y_=y_, z=z, delta_I=delta_I)
        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_index, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width, q_index / 100)
        x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result