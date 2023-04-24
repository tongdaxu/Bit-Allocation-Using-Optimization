# Copyright 2020 InterDigital Communications, Inc.
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

import math
import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# pylint: disable=E0611,E0401
from ..entropy_models.entropy_models import EntropyBottleneck, GaussianConditional
from ..layers.layers import GDN, MaskedConv2d

from .utils import conv, deconv, update_registered_buffers

# pylint: enable=E0611,E0401


__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
]


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            m.update(force=force)


class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def quant(self, x, mode, delta):
        if mode == "training":
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + noise
        else:
            x = torch.round(x / delta) * delta
        return x

    def forward(self, content):
        stage = content[1]
        if stage == "test_for_first":
            x = content[0]
            mode = content[2]
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode)
            scales_hat = self.h_s(z_hat)

            y_hat, y_likelihoods, delta = self.gaussian_conditional(y, scales_hat, means=None, mode=mode, iter=0,
                                                                total_iter=0, delta=1.0)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "y": y,
                "z": z,
                "delta": delta,
            }

        elif stage == "finetune":
            mode = content[2]
            iter = content[3]
            total_iter = content[4]
            y = content[5]
            z = content[6]
            delta = content[7]

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode, it=iter, total_iter=total_iter)
            scales_hat = self.h_s(z_hat)

            y_hat, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=None, mode=mode, iter=iter,
                                                                total_iter=total_iter, delta=delta)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        elif stage == "finetune_enc":
            x = content[0]
            mode = content[2]
            iter = content[3]
            total_iter = content[4]
            delta = content[7]

            y = self.g_a(x)
            z = self.h_a(abs(y))

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode, it=iter, total_iter=total_iter, finetune_enc=1)
            scales_hat = self.h_s(z_hat)

            y_hat, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=None, mode=mode, iter=iter,
                                                                total_iter=total_iter, finetune_enc=1)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        elif stage == "test_for_final":
            mode = content[2]
            y = content[5]
            z = content[6]
            delta = content[7]

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode)
            scales_hat = self.h_s(z_hat)

            y_hat, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=None, mode=mode, iter=0, total_iter=0,
                                                            delta=delta)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        else:
            pass

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def encode_decode(self, x, output_path):
        N, C, H, W = x.size()
        bits = self.encode(x, output_path) * 8
        bpp = bits / (H * W)
        x_hat = self.decode(output_path)
        result = {
            'bpp': bpp,
            'x_hat': x_hat,
        }
        return result

    def encode(self, x, output_path):
        from ..utils.stream_helper import encode_i
        N, C, H, W = x.size()
        compressed = self.compress(x)
        y_string = compressed['strings'][0][0]
        z_string = compressed['strings'][1][0]
        encode_i(H, W, y_string, z_string, output_path)
        return len(y_string) + len(z_string)

    def decode(self, input_path):
        from ..utils.stream_helper import decode_i, get_downsampled_shape
        height, width, y_string, z_string = decode_i(input_path)
        shape = get_downsampled_shape(height, width, 64)
        decompressed = self.decompress([[y_string], [z_string]], shape)
        return decompressed['x_hat']

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        y_hat = y_hat.to(z_hat.device)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class JointAutoregressiveHierarchicalPriors(CompressionModel):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.Quantizator_SGA = Quantizator_SGA()

    def quant(self, x, mode, delta):
        if mode == "training":
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + noise
        else:
            x = torch.round(x / delta) * delta
        return x

    def forward(self, content):
        stage = content[1]
        if stage == "test_for_first":
            x = content[0]
            mode = content[2]
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode)
            params = self.h_s(z_hat)

            y_hat = self.Quantizator_SGA(y, mode=mode)
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods, delta = self.gaussian_conditional(y, scales_hat, means=means_hat, mode=mode, iter=0,
                                                                total_iter=0, delta=1.0)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "y": y,
                "z": z,
                "delta": delta,
            }

        elif stage == "finetune":
            mode = content[2]
            iter = content[3]
            total_iter = content[4]
            y = content[5]
            z = content[6]
            delta = content[7]

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode, it=iter, total_iter=total_iter)
            params = self.h_s(z_hat)

            y_hat = self.Quantizator_SGA(y / delta, it=iter, mode=mode, total_it=total_iter) * delta
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=means_hat, mode=mode, iter=iter,
                                                            total_iter=total_iter, delta=delta)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        elif stage == "finetune_enc":
            x = content[0]
            mode = content[2]
            iter = content[3]
            total_iter = content[4]
            delta = content[7]

            y = self.g_a(x)
            z = self.h_a(y)

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode, it=iter, total_iter=total_iter, finetune_enc=1)
            params = self.h_s(z_hat)

            y_hat = self.quant(y, mode=mode, delta=1)
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=means_hat, mode=mode, iter=iter,
                                                            total_iter=total_iter, finetune_enc=1)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        elif stage == "test_for_final":
            mode = content[2]
            y = content[5]
            z = content[6]
            delta = content[7]

            z_hat, z_likelihoods = self.entropy_bottleneck(z, mode)
            params = self.h_s(z_hat)

            y_hat = self.Quantizator_SGA(y/delta, mode=mode)*delta
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods, _ = self.gaussian_conditional(y, scales_hat, means=means_hat, mode=mode, iter=0, total_iter=0,
                                                         delta=delta)
            x_hat = self.g_s(y_hat)

            if len(content) == 9:
                dir = content[8]
                path = "observe/Analysis_E/Seq" + str(dir) + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                bits = -torch.log(y_likelihoods) / math.log(2)
                bits_for_vis = torch.sum(bits, dim=1)
                writer = open(path + "I_y.bin", "wb")
                writer.write(bits_for_vis.cpu().detach().numpy())

            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }

        else:
            pass

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def encode_decode(self, x, output_path):
        N, C, H, W = x.size()
        bits = self.encode(x, output_path) * 8
        bpp = bits / (H * W)
        x_hat = self.decode(output_path)
        result = {
            'bpp': bpp,
            'x_hat': x_hat,
        }
        return result

    def encode(self, x, output_path):
        from ..utils.stream_helper import encode_i
        N, C, H, W = x.size()
        compressed = self.compress(x)
        y_string = compressed['strings'][0][0]
        z_string = compressed['strings'][1][0]
        encode_i(H, W, y_string, z_string, output_path)
        return len(y_string) + len(z_string)

    def decode(self, input_path):
        from ..utils.stream_helper import decode_i, get_downsampled_shape
        height, width, y_string, z_string = decode_i(input_path)
        shape = get_downsampled_shape(height, width, 64)
        decompressed = self.decompress([[y_string], [z_string]], shape)
        return decompressed['x_hat']

    def compress(self, x):
        from ..entropy_models.MLCodec_rans import BufferedRansEncoder
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access

        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[i:i + 1, :, h:h + kernel_size, w:w + kernel_size]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )

                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i:i + 1, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                        i, :, padding, padding
                    ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        from ..entropy_models.MLCodec_rans import RansDecoder
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[i:i + 1, :, h:h + kernel_size, w:w + kernel_size]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i:i + 1, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)

                    rv = decoder.decode_stream(
                        indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)

                    rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[i, :, h + padding: h + padding + 1, w + padding: w + padding + 1] = rv
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class Quantizator_SGA(nn.Module):
    """
    https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/c9b5c1354a38e0bb505fc34c6c8f27170f62a75b/sga.py#L110
    Stochastic Gumbeling Annealing
    sample() has no grad, so we choose STE to backward. We can also try other estimate func.
    """

    def __init__(self, gap=1000, c=0.001):
        super(Quantizator_SGA, self).__init__()
        self.gap = gap
        self.c = c

    def annealed_temperature(self, t, r, ub, lb=1e-8, backend=np, scheme='exp', **kwargs):
        """
        Return the temperature at time step t, based on a chosen annealing schedule.
        :param t: step/iteration number
        :param r: decay strength
        :param ub: maximum/init temperature
        :param lb: small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
        :param backend: np or tf
        :param scheme:
        :param kwargs:
        :return:
        """
        default_t0 = kwargs.get('t0')

        if scheme == 'exp':
            tau = backend.exp(-r * t)
        elif scheme == 'exp0':
            # Modified version of above that fixes temperature at ub for initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = ub * backend.exp(-r * (t - t0))
        elif scheme == 'linear':
            # Cool temperature linearly from ub after the initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = -r * (t - t0) + ub
        else:
            raise NotImplementedError

        if backend is None:
            return min(max(tau, lb), ub)
        else:
            return backend.minimum(backend.maximum(tau, lb), ub)

    def forward(self, input, it=None, mode=None, total_it=None):
        if mode == "training":
            assert it is not None
            x_floor = torch.floor(input)
            x_ceil = torch.ceil(input)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)

            eps = 1e-5

            # TDOO: input outside
            annealing_scheme = 'exp0'
            annealing_rate = 1e-3  # default annealing_rate = 1e-3
            t0 = int(total_it * 0.35)  # default t0 = 700 for 2000 iters
            T_ub = 0.5

            T = self.annealed_temperature(it, r=annealing_rate, ub=T_ub, scheme=annealing_scheme, t0=t0)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

            rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            rx = F.softmax(rx_logits, dim=-1)  # just for observation in tensorboard
            rx_dist = torch.distributions.RelaxedOneHotCategorical(T, rx)

            rx_sample = rx_dist.rsample()

            x_tilde = torch.sum(x_bds * rx_sample, dim=-1)
            return x_tilde
        else:
            return torch.round(input)