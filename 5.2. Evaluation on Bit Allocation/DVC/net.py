import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchac
import cv2

from .subnet.sga import Quantizator_SGA
from .subnet.flowlib import save_flow_image
from .subnet import *
from .subnet.vis_flow import *


def vis_feature(x, max_num=5, out_path='observe/'):
    for i in range(0, x.shape[1]):
        if i >= max_num:
            break
        feature = x[0, i, :, :].view(x.shape[-2], x.shape[-1])
        feature = feature.cpu().numpy()
        feature = np.ones_like(feature) - feature
        # feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.asarray(feature * 255).astype(np.uint8)
        feature_img = cv2.applyColorMap(feature, cv2.COLORMAP_OCEAN)
        dst_path = os.path.join(out_path, str(i) + '____ini.png')
        cv2.imwrite(dst_path, feature_img)


def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0
        # self.mxrange = 15
        self.SGA = Quantizator_SGA()

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def quant(self, x, mode, delta, tau=None, sigma=None):
        if mode == "training":
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + noise
        else:
            x = self.round(x / delta, tau, sigma) * delta
        return x

    def round(self, x, tau=None, sigma=None):
        if sigma is not None or tau is not None:
            tau = torch.mean(sigma) * 0.8 if tau is None else tau
            threshold = torch.zeros_like(sigma) + tau
            mask = torch.ceil(sigma - threshold).clamp(0, 1)
            return mask * torch.round(x)
        else:
            return torch.round(x)

    def forward(self, input_image, referframe, iter, total_iter, stage, mode, mvfeature=None, feature=None, z=None,
                delta=1.0, dir=None, calrealbits=False, channel_delta=None, delta_mv=1.0, channel_delta_mv=None,
                force_sga=False):
        if stage == "test_for_first":
            estmv = self.opticFlow(input_image, referframe)
            mvfeature = self.mvEncoder(estmv)

            quant_mv = self.quant(mvfeature, mode, delta=1.0)

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            z = self.respriorEncoder(feature)

            compressed_z = self.quant(z, mode, delta=1.0)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            compressed_feature_renorm = self.quant(feature_renorm, mode, delta=1.0, sigma=None)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = bits

                return bits, real_bits, probs, delta

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv=1.0):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                bits = torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = bits

                return bits, real_bits, prob, delta_mv

            bits_feature, real_bits_feature, _, delta = feature_probs_based_sigma(compressed_feature_renorm,
                                                                                        recon_sigma, delta=1.0)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            bits_mv, real_bits_mv, _, delta_mv = iclr18_estrate_bits_mv(quant_mv)

            im_shape = input_image.size()

            # bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            # bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            # bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])

            bpp_feature = bits_feature.sum(dim=(0, 1)) / batch_size
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = bits_mv.sum(dim=(0, 1)) / batch_size

            bpp = bpp_feature + bpp_z + bpp_mv
            #
            # print(bpp_feature.abs())
            # print(bpp_mv.abs())
            # assert 0

            real_bpp_feature = real_bits_feature.sum(dim=(0, 1)) / batch_size
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, mvfeature, \
                   feature, z, torch.tensor(delta).cuda(), torch.tensor(
                delta_mv).cuda(), real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune":
            # estmv = self.opticFlow(input_image, referframe)
            # mvfeature = self.mvEncoder(estmv)
            if force_sga:
                quant_mv = self.SGA(mvfeature / delta_mv, it=iter, mode="training", total_it=total_iter) * delta_mv
            else:
                quant_mv = torch.round(mvfeature / delta_mv) * delta_mv

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            # feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            # z = self.respriorEncoder(feature)

            # compressed_z = self.quant(z, mode, delta)
            compressed_z = self.SGA(z, it=iter, mode=mode, total_it=total_iter)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            # compressed_feature_renorm = self.quant(feature_renorm, mode, delta)
            compressed_feature_renorm = self.SGA(feature_renorm / delta, it=iter, mode=mode,
                                                 total_it=total_iter) * delta

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def feature_probs_based_sigma_per_pixel(feature, sigma, delta=1.0):

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)
                return total_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                total_bits = torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv)

            im_shape = input_image.size()

            # bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            # bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            # bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            # bpp = bpp_feature + bpp_z + bpp_mv

            bpp_feature = bits_feature.sum(dim=(0, 1)) / batch_size
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = bits_mv.sum(dim=(0, 1)) / batch_size

            bpp = bpp_feature + bpp_z + bpp_mv
            #
            # print(bpp_feature.abs())
            # print(bpp_mv.abs())
            # assert 0

            real_bpp_feature = real_bits_feature.sum(dim=(0, 1)) / batch_size
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            # '''visualize bit allocation'''
            # if mode == "test" and dir is not None:
            #     path = "observe/Analysis_E/Seq" + str(dir) + "/"
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #
            #     bits = feature_probs_based_sigma_per_pixel(compressed_feature_renorm, recon_sigma, delta=delta)
            #     bits_for_vis = torch.sum(bits, dim=1)
            #     print("latent representation shape", bits_for_vis.shape)
            #     writer = open(path + "P_y.bin", "wb")
            #     writer.write(bits_for_vis.cpu().detach().numpy())
            #
            #     vis_mv = F.interpolate(quant_mv_upsample, scale_factor=0.05, mode='bilinear', align_corners=True)
            #     vis_sparse_flow(vis_mv.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),
            #                     path=path+"arrow_flow.pdf")
            #     save_flow_image(quant_mv_upsample.squeeze(0).cpu().detach().numpy(), path+"normal_flow.png")

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune_enc":
            estmv = self.opticFlow(input_image, referframe)
            mvfeature = self.mvEncoder(estmv)

            quant_mv = self.quant(mvfeature, mode=mode, delta=1)

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            z = self.respriorEncoder(feature)

            compressed_z = self.quant(z, mode, 1)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            compressed_feature_renorm = self.quant(feature_renorm, mode, 1)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(i - 0.5).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and not self.training:
                    decodedx, real_bits = getrealbitsg(feature, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and not self.training:
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and not self.training:
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            total_bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            total_bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

            im_shape = input_image.size()

            bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            bpp = bpp_feature + bpp_z + bpp_mv

            real_bpp_feature = real_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune_flow":
            # estmv = self.opticFlow(input_image, referframe)
            # mvfeature = self.mvEncoder(estmv)

            quant_mv = self.SGA(mvfeature / delta_mv, it=iter, mode=mode, total_it=total_iter) * delta_mv

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            z = self.respriorEncoder(feature)

            # compressed_z = self.SGA(z, it=iter, mode=mode, total_it=total_iter)
            compressed_z = self.quant(z, mode, delta=1.0)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            # compressed_feature_renorm = self.SGA(feature_renorm, it=iter, mode=mode, total_it=total_iter)
            compressed_feature_renorm = self.quant(feature_renorm, mode, delta=1.0)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                total_bits = torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv)

            im_shape = input_image.size()

            # bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            # bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            # bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            # bpp = bpp_feature + bpp_z + bpp_mv

            bpp_feature = bits_feature.sum(dim=(0, 1)) / batch_size
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = bits_mv.sum(dim=(0, 1)) / batch_size

            bpp = bpp_feature + bpp_z + bpp_mv
            #
            # print(bpp_feature.abs())
            # print(bpp_mv.abs())
            # assert 0

            real_bpp_feature = real_bits_feature.sum(dim=(0, 1)) / batch_size
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune_flow_STE":

            quant_mv = self.SGA(mvfeature / delta_mv, it=iter, mode="training", total_it=total_iter) * delta_mv

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            batch_size = feature.size()[0]

            class STEFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, y_fake, z_fake, y_real, z_real):
                    return y_real, z_real

                @staticmethod
                def backward(ctx, y_real_grad, z_real_grad):
                    return y_real_grad, z_real_grad, None, None

            feature_fake = self.resEncoder(input_residual)

            z_fake = self.respriorEncoder(feature_fake)

            feature, z = STEFunction.apply(feature_fake, z_fake, feature, z)
            # feature, z = feature_fake, z_fake
            # feature = feature_fake
            # z = z_fake
            if force_sga:
                compressed_z = self.SGA(z, it=iter, mode="training", total_it=total_iter)
            else:
                compressed_z = self.quant(z, mode, delta=1.0)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            if force_sga:
                compressed_feature_renorm = self.SGA(feature_renorm, it=iter, mode="training", total_it=total_iter)
            else:
                compressed_feature_renorm = torch.round(feature_renorm)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            total_bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            total_bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv)

            im_shape = input_image.size()

            bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            bpp = bpp_feature + bpp_z + bpp_mv

            real_bpp_feature = real_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "test_for_stage1":
            # estmv = self.opticFlow(input_image, referframe)
            # mvfeature = self.mvEncoder(estmv)
            if force_sga:
                quant_mv = self.SGA(mvfeature / delta_mv, it=iter, mode="training", total_it=total_iter) * delta_mv
            else:
                quant_mv = self.quant(mvfeature, mode, delta=delta_mv)

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            z = self.respriorEncoder(feature)

            if force_sga:
                compressed_z = self.SGA(z, it=iter, mode="training", total_it=total_iter)
            else:
                compressed_z = self.quant(z, mode, delta=1.0)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            if force_sga:
                compressed_feature_renorm = self.SGA(feature_renorm, it=iter, mode="training", total_it=total_iter)
            else:
                compressed_feature_renorm = self.quant(feature_renorm, mode, delta=1.0)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs, delta

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                total_bits = torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            bits_feature, real_bits_feature, _, delta = feature_probs_based_sigma(compressed_feature_renorm,
                                                                                        recon_sigma, delta=1.0)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv)

            im_shape = input_image.size()

            # bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            # bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            # bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            # bpp = bpp_feature + bpp_z + bpp_mv

            bpp_feature = bits_feature.sum(dim=(0, 1)) / batch_size
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = bits_mv.sum(dim=(0, 1)) / batch_size

            bpp = bpp_feature + bpp_z + bpp_mv
            #
            # print(bpp_feature.abs())
            # print(bpp_mv.abs())
            # assert 0

            real_bpp_feature = real_bits_feature.sum(dim=(0, 1)) / batch_size
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, mvfeature, \
                   feature, z, torch.tensor(delta).cuda(), real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune_all":
            # estmv = self.opticFlow(input_image, referframe)
            # mvfeature = self.mvEncoder(estmv)

            quant_mv = self.SGA(mvfeature / delta_mv, it=iter, mode=mode, total_it=total_iter) * delta_mv

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            batch_size = feature.size()[0]

            compressed_z = self.SGA(z, it=iter, mode=mode, total_it=total_iter)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            compressed_feature_renorm = self.SGA(feature_renorm, it=iter, mode=mode, total_it=total_iter)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(gaussian.cdf(torch.tensor(i) * delta - delta / 2).view(n, c, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + delta / 2) - gaussian.cdf(feature - delta / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / delta, gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + delta_mv / 2) - self.bitEstimator_mv(mv - delta_mv / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            total_bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            total_bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv=delta_mv)

            im_shape = input_image.size()

            bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            bpp = bpp_feature + bpp_z + bpp_mv

            real_bpp_feature = real_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        elif stage == "finetune_delta":
            estmv = self.opticFlow(input_image, referframe)
            mvfeature = self.mvEncoder(estmv)

            quant_mv = self.SGA(mvfeature / (delta_mv * channel_delta_mv), it=iter, mode=mode,
                                total_it=total_iter) * (delta_mv * channel_delta_mv)

            quant_mv_upsample = self.mvDecoder(quant_mv)

            prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

            input_residual = input_image - prediction

            feature = self.resEncoder(input_residual)
            batch_size = feature.size()[0]

            z = self.respriorEncoder(feature)

            compressed_z = torch.round(z)

            recon_sigma = self.respriorDecoder(compressed_z)

            feature_renorm = feature

            compressed_feature_renorm = self.SGA(feature_renorm / (delta * channel_delta), it=iter, mode=mode,
                                                 total_it=total_iter) * (delta * channel_delta)

            recon_res = self.resDecoder(compressed_feature_renorm)
            recon_image = prediction + recon_res

            clipped_recon_image = recon_image.clamp(0., 1.)

            mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

            warploss = torch.mean((warpframe - input_image).pow(2))
            interloss = torch.mean((prediction - input_image).pow(2))

            def feature_probs_based_sigma(feature, sigma, delta=1.0, channel_delta=None):

                def getrealbitsg(x, gaussian):
                    # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(
                            gaussian.cdf(torch.tensor(i) * delta * channel_delta - (delta * channel_delta) / 2).view(n,
                                                                                                                     c,
                                                                                                                     h,
                                                                                                                     w,
                                                                                                                     1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()

                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits[0]

                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                probs = gaussian.cdf(feature + (delta * channel_delta) / 2) - gaussian.cdf(
                    feature - (delta * channel_delta) / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbitsg(feature / (delta * channel_delta), gaussian)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, probs

            def iclr18_estrate_bits_z(z):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(z)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            def iclr18_estrate_bits_mv(mv, delta_mv, channel_delta_mv):

                def getrealbits(x):
                    cdfs = []
                    mxrange = torch.ceil(torch.abs(x.max())) if torch.abs(x.max()) > torch.abs(x.min()) else torch.ceil(
                        torch.abs(x.min()))
                    mxrange = int(mxrange) + 2
                    x = x + mxrange
                    n, c, h, w = x.shape
                    for i in range(-mxrange, mxrange):
                        cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                    cdfs = torch.cat(cdfs, 4).cpu().detach()
                    byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
                                                           check_input_bounds=True)

                    real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                    sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                    return sym_out - mxrange, real_bits

                prob = self.bitEstimator_mv(mv + (delta_mv * channel_delta_mv) / 2) - self.bitEstimator_mv(
                    mv - (delta_mv * channel_delta_mv) / 2)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

                if calrealbits and mode == "test":
                    decodedx, real_bits = getrealbits(mv)
                else:
                    real_bits = total_bits

                return total_bits, real_bits, prob

            total_bits_feature, real_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma,
                                                                                 delta=delta,
                                                                                 channel_delta=channel_delta)
            total_bits_z, real_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
            total_bits_mv, real_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv, delta_mv=delta_mv,
                                                                    channel_delta_mv=channel_delta_mv)

            im_shape = input_image.size()

            bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
            bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            bpp = bpp_feature + bpp_z + bpp_mv

            real_bpp_feature = real_bits_feature / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_z = real_bits_z / (batch_size * im_shape[2] * im_shape[3])
            real_bpp_mv = real_bits_mv / (batch_size * im_shape[2] * im_shape[3])
            real_bpp = real_bpp_feature + real_bpp_z + real_bpp_mv

            return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
                   real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp

        else:
            print("No this stage")



