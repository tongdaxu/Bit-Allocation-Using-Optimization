import math
import os
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torchvision
import argparse
import itertools
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
from torch.nn import functional
from torch.autograd import Variable

from dataset import HEVCDataSet, UVGDataSet, crop, merge
from HSTEM.src.models.video_model_q import DMC
from HSTEM.src.models.image_model import IntraNoAR
from DCVC.src.models.DCVC_net_dcvc import DCVC_net
from DVC.net import VideoCompressor
from I_model.zoo.image import model_architectures as architectures


os.environ["CUDA_VISIBLE_DEVICES"] = " 0 "
gpu_num = torch.cuda.device_count()

lambda_quality_map = {256: 3,
                      512: 4,
                      1024: 5,
                      2048: 6}


def activate_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def close_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def cal_rd_cost(distortion: torch.Tensor, bpp: torch.Tensor, lambda_weight):
    rd_cost = lambda_weight * distortion + bpp
    return rd_cost


def cal_bpp(likelihood: torch.Tensor, num_pixels: int):
    bpp = torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
    return bpp


def cal_bits(likelihood: torch.Tensor):
    bits = torch.log(likelihood).sum() / (-math.log(2))
    return bits


def cal_distoration(A: torch.Tensor, B:torch.Tensor):
    dis = nn.MSELoss()
    return dis(A, B)


def cal_psnr(distortion: torch.Tensor):
    psnr = -10 * torch.log10(distortion)
    return psnr


def Var(x):
    return Variable(x.cuda())


def test_dvc(net, iter_res, test_dataset, optimize_range, factor, overlap, calrealbits=False):
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=1, batch_size=1, pin_memory=True)
    sumbpp, sumbpp_mv_y, sumbpp_mv_z = 0, 0, 0
    sumbpp_res_y, sumbpp_res_z, sumbpp_real = 0, 0, 0
    sumpsnr, sumpsnr_pre = 0, 0
    eval_step = 0
    gop_num = 0
    avg_loss = torch.zeros(size=[1, ])
    dir = 1
    close_grad(I_codec)
    close_grad(net)
    I_codec.eval()
    net.eval()
    ############################ iterative tuning #############################
    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("[info] testing : %d/%d" % (batch_idx, len(test_loader)))
        input_images = input[0].squeeze(0)
        seqlen = input_images.size()[0]
        cropped_len = len(crop(torch.unsqueeze(input_images[0, :, :, :], 0), factor=factor, overlap=overlap)[0])
        print("number of blocks:", cropped_len)
        B, C, H, W = torch.unsqueeze(input_images[0, :, :, :], 0).shape[0], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[1], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[2], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[3]
        cropped_blocks = []
        for m in range(seqlen):
            cur_frame = torch.unsqueeze(input_images[m, :, :, :], 0)
            cropped_images, _, _, _, _ = crop(cur_frame, factor=factor, overlap=overlap)
            cropped_blocks.append(cropped_images)
        total_bits_stack, rec_img_stack = [], []
        ############################ initial testing ##########################
        total_rd_cost = 0
        for i in range(seqlen):
            cur_frame = Var(torch.unsqueeze(input_images[i, :, :, :].cuda(), 0))
            b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
            num_pixels = b * h * w
            if i == 0:
                with torch.no_grad():
                    arr = I_codec([cur_frame, "test_for_first", "testing"])
                I_rec = arr['x_hat']
                I_likelihood_y, I_likelihood_z = arr["likelihoods"]['y'], arr["likelihoods"]['z']
                ref_image = I_rec.detach().clone()
                y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy()
                z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                bpp = y_bpp + z_bpp
                distortion = cal_distoration(cur_frame, I_rec)
                rd_cost = cal_rd_cost(distortion, bpp.mean(), lambda_for_test)
                print("\n------------------------------------ GOP {0} --------------------------------------".format(
                    batch_idx + 1))
                print("I frame:  ", "bpp:", bpp.mean(), "\t", "psnr:", psnr.mean(), "\t", "mse:", distortion.cpu().detach().numpy(), "\trd_cost:",
                      rd_cost.cpu().detach().numpy())
            else:
                with torch.no_grad():
                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                        net(referframe=ref_image, input_image=cur_frame, iter=0, total_iter=0, stage="test_for_first",
                            mode="test")
                ref_image = clipped_recon_image

                distortion = cal_distoration(cur_frame, clipped_recon_image)
                rd_cost = cal_rd_cost(distortion, bpp.sum() / h / w, lambda_for_test)
                psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                mv_bpp = bpp_mv.sum() / h / w
                res_y_bpp = bpp_feature.sum() / h / w
                res_z_bpp = bpp_z
                bpp = bpp.sum() / h / w
                print("P{0} frame: ".format(i), "mv_bpp:", mv_bpp.cpu().detach().numpy().mean(), "\t", "res_bpp:",
                      res_y_bpp.cpu().detach().numpy().mean(), "\t", "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy().mean(), "\t",
                      "bpp:", bpp.cpu().detach().numpy().mean(), "\t", "psnr", psnr, "\t", "mse:", distortion.cpu().detach().numpy(), "\t", "rd_cost",
                      rd_cost.cpu().detach().numpy())
            total_rd_cost += rd_cost
        print("total_rd_cost_initial:", total_rd_cost.cpu().detach().numpy())

        for n in range(cropped_len):

            # """ finetuning the encoder """
            # optimizer_enc = Adam(params=itertools.chain(I_codec.g_a.parameters(),
            #                                             I_codec.h_a.parameters(),
            #                                             net.opticFlow.parameters(),
            #                                             net.mvEncoder.parameters(),
            #                                             net.resEncoder.parameters(),
            #                                             net.respriorEncoder.parameters(),
            #                                             ), lr=5e-5)
            # scheduler_enc = lr_scheduler.StepLR(optimizer_enc, int(iter_enc * 0.9), gamma=0.1)
            # I_codec.train()
            # net.train()
            # activate_grad(I_codec)
            # activate_grad(net)
            # close_grad(I_codec.g_s)
            # close_grad(I_codec.h_s)
            # close_grad(net.mvDecoder)
            # close_grad(net.warpnet)
            # close_grad(net.resDecoder)
            # close_grad(net.respriorDecoder)
            # close_grad(net.bitEstimator_z)
            # close_grad(net.bitEstimator_mv)
            # for it in tqdm(range(iter_enc)):
            #     optimizer_enc.zero_grad()
            #     for i in range(seqlen):
            #         cur_frame = Var(cropped_blocks[i][n].cuda())
            #         b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
            #         num_pixels = b * h * w
            #         if i == 0:
            #             result = I_codec([cur_frame, "finetune_enc", "training", 0, 0, None,
            #                               None, 1.0])
            #             recon_image = result['x_hat']
            #             I_likelihood_y = result["likelihoods"]['y']
            #             I_likelihood_z = result["likelihoods"]['z']
            #             y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels)
            #             z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
            #             bpp = y_bpp + z_bpp
            #         else:
            #             clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, \
            #             real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp = net(
            #                 referframe=ref_image, input_image=cur_frame, mode="training",
            #                 stage="finetune_enc", iter=0, total_iter=0, delta=1.0,
            #                 mvfeature=None, feature=None, z=None)
            #             recon_image = clipped_recon_image
            #             bpp = bpp
            #
            #         ref_image = recon_image
            #         distortion = cal_distoration(cur_frame, recon_image)
            #         rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test).requires_grad_(True)
            #         rd_cost = rd_cost / seqlen
            #         rd_cost.backward(retain_graph=True)
            #         print(net.resEncoder.conv1.weight.grad)
            #     optimizer_enc.step()
            #     scheduler_enc.step()

            ############# initial testing end, tuning start #######################
            I_y_stack, I_z_stack, delta_I_stack = [], [], []
            mv_feature_stack, delta_mv_stack = [], []
            feature_stack, z_stack, delta_res_stack = [], [], []
            corr_iter_I, corr_iter_w, corr_iter_y = 1, 1, 1
            sub_iter_I, sub_iter_w, sub_iter_y, sub_lr = int(2000 / corr_iter_I), int(400 / corr_iter_w), \
                                                         int(400 / corr_iter_y), 1e-3

            '''calculate inference time'''
            torch.cuda.synchronize()
            time_start = time.time()
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                num_pixels = b * h * w
                if i == 0:
                    ###################### initialize param #######################
                    with torch.no_grad():
                        # use deterministic rounding here
                        arr = I_codec([cur_frame, "test_for_first", "testing"])
                    I_y_stack.append(arr['y'].detach().clone().requires_grad_(True))
                    I_z_stack.append(arr['z'].detach().clone().requires_grad_(True))
                    delta_I_stack.append(torch.tensor(arr["delta"]).clone().detach().requires_grad_(True))

                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_I + 2)], [[] for _ in range(corr_iter_I + 2)]
                    ##################### calculate D, R before tuning ###########################
                    frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                    for sub_i in range(frame_range):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec([cur_frame, "test_for_first", "testing"])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            ref_image = result['x_hat'].detach().clone()
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                           cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                            I_rec = result["x_hat"]
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)

                        else:
                            with torch.no_grad():
                                clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                                    net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                        stage="test_for_first", mode="test")
                            ref_image = clipped_recon_image
                            distortion = cal_distoration(sub_cur_frame, clipped_recon_image)
                            bpp = bpp.sum() / h / w
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)

                    ##################### actual tuning I frame ###########################
                    cur_params = I_y_stack + I_z_stack + delta_I_stack
                    # cur_params = I_y_stack + I_z_stack
                    optimizer_I = Adam(params=cur_params, lr=sub_lr)
                    lambda_for_test_new_I = lambda_for_test
                    for corr_iter in range(corr_iter_I):
                        for sub_it in range(sub_iter_I):
                            optimizer_I.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                                if sub_i == 0:
                                    result = I_codec(
                                        [sub_cur_frame, "finetune", "training", sub_it, sub_iter_I, I_y_stack[0], I_z_stack[0],
                                         delta_I_stack[0]])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                                   cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _, _, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_first", mode="training")
                                    recon_image = clipped_recon_image
                                    bpp = bpp.sum() / h / w
                                ref_image = recon_image
                                distortion = cal_distoration(sub_cur_frame, recon_image)
                                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new_I) / frame_range
                                rd_cost.backward(retain_graph=True)

                            optimizer_I.step()
                            optimizer_I.zero_grad()
                        if corr_iter < corr_iter_I - 1:

                            ##################### calculate D, R after sub_iter_I steps ###########################
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                                if sub_i == 0:
                                    with torch.no_grad():
                                        result = I_codec(
                                            [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0],
                                             delta_I_stack[0]])
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    ref_image = result['x_hat'].detach().clone()
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                                   cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    I_rec = result["x_hat"]
                                    bpp = y_bpp + z_bpp
                                    distortion = cal_distoration(sub_cur_frame, I_rec)
                                    print("I frame after k1 steps. mse:", distortion.cpu().detach().numpy(), "bpp:",
                                          bpp.cpu().detach().numpy())
                                else:
                                    with torch.no_grad():
                                        clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                stage="test_for_first", mode="test")
                                    ref_image = clipped_recon_image
                                    bpp = bpp_feature.sum() / h / w + bpp_mv.sum() / h / w + bpp_z
                                    distortion = cal_distoration(sub_cur_frame, clipped_recon_image)
                                    print("P {0} frame after k1 steps. mse:".format(sub_i), distortion.cpu().detach().numpy(),
                                          "bpp:", bpp.cpu().detach().numpy())
                                if sub_i in range(i, frame_range):
                                    R_k_stack[corr_iter + 1].append(bpp)
                                    D_k_stack[corr_iter + 1].append(distortion)

                    for param in cur_params:
                        param.requires_grad = False
                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_w + 2)], [[] for _ in range(corr_iter_w + 2)]
                    ###################### sub testing start and calculate R, D before mv tuning ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            ref_image = result['x_hat'].detach().clone()
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels),\
                                           cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                            I_rec = result["x_hat"]
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion.cpu(), bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {0} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t",
                                  "psnr:", psnr, "\t", "mse:", distortion.mean().cpu().detach().numpy(), "\t",
                                  "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                                    net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                        stage="test_for_first", mode="test")
                            ref_image = clipped_recon_image
                            distortion = cal_distoration(sub_cur_frame, clipped_recon_image)
                            bpp = bpp_feature.sum()/h/w + bpp_mv.sum()/h/w + bpp_z
                            rd_cost = cal_rd_cost(distortion.cpu(), bpp.cpu(), lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_bpp = bpp_mv.sum()/h/w
                            res_y_bpp = bpp_feature.sum()/h/w
                            res_z_bpp = bpp_z
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_bpp:",
                                  mv_bpp.cpu().detach().numpy(), "\t", "res_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                                  "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:", bpp.cpu().detach().numpy(), "\t",
                                  "psnr", psnr, "\t", "mse:", distortion.cpu().detach().numpy(), "\t",
                                  "rd_cost", rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                        frame_range = i + optimize_range if i + optimize_range < seqlen else seqlen
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)
                    print("total_rd_cost_middle {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                else:
                    ################## initialize motion vectors of frame i #######
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            ref_image = result['x_hat'].detach().clone()
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    clipped_recon_image, _, _, _, _, _, _, _, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="finetune", mode="test", \
                                            feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                            delta=delta_res_stack[sub_i - 1][0], \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                            calrealbits=calrealbits)
                                elif sub_i == i:
                                    clipped_recon_image, _, _, _, _, _, _, _, mvfeature, _, _, _, delta_mv, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_first", mode="test")
                                    mv_feature_stack.append([mvfeature.detach().clone().requires_grad_(True)])
                                    delta_mv_stack.append([torch.tensor(delta_mv).detach().clone().requires_grad_(False)])
                                else:
                                    assert (0)
                            ref_image = clipped_recon_image

                    ##################### actual tuning mv of frame i #############
                    cur_params = mv_feature_stack[i - 1]
                    optimizer_mv = Adam(params=cur_params, lr=sub_lr)
                    lambda_for_test_new = lambda_for_test
                    print("lambda_for_test_new:", lambda_for_test_new)
                    for corr_iter in range(corr_iter_w):
                        for sub_it in range(sub_iter_w):
                            optimizer_mv.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                if sub_i == 0:
                                    I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], delta_I_stack[
                                        0]
                                    result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_for_optim, I_z_for_optim,
                                                      delta_I_for_optim])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                                   cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    if sub_i < i:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                stage="finetune", mode="test",
                                                feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                delta=delta_res_stack[sub_i - 1][0],
                                                mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                                calrealbits=calrealbits)
                                    elif sub_i == i:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=sub_it, total_iter=sub_iter_w,
                                                stage="finetune_flow", mode="training",
                                                mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0])
                                    else:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _, _, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                stage="test_for_first", mode="training")
                                    recon_image = clipped_recon_image
                                    bpp = bpp.sum() / h / w
                                ref_image = recon_image
                                if sub_i >= i:
                                    distortion = cal_distoration(sub_cur_frame, recon_image)
                                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new) / frame_range
                                    rd_cost.backward(retain_graph=True)
                            optimizer_mv.step()
                            optimizer_mv.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_y + 2)], [[] for _ in range(corr_iter_y + 2)]
                    ###################### sub testing start and calculate R, D before residule tuning ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            I_rec = result['x_hat'].detach().clone()
                            ref_image = I_rec
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                           cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t",
                                  "psnr:", psnr, "\t", "mse:", distortion.cpu().detach().numpy(), "\t",
                                  "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="finetune", mode="test", \
                                            feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                            delta=delta_res_stack[sub_i - 1][0], \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                            calrealbits=calrealbits)
                                elif sub_i == i:
                                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="finetune_flow", mode="test", \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0])
                                else:
                                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_first", mode="test")
                            ref_image = clipped_recon_image
                            mv_bpp = bpp_mv.sum() / h / w
                            res_y_bpp = bpp_feature.sum() / h / w
                            res_z_bpp = bpp_z
                            bpp = mv_bpp + res_y_bpp + res_z_bpp
                            distortion = cal_distoration(sub_cur_frame, clipped_recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_bpp:",
                                  mv_bpp.cpu().detach().numpy(), "\t", "res_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                                  "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "mse:",
                                  distortion.cpu().detach().numpy(), "\t",
                                  "rd_cost", rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                        frame_range = i + optimize_range if i + optimize_range < seqlen else seqlen
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)
                    print("total_rd_cost_middle_mv {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                    ################## initialize residule of frame i #############
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            ref_image = result['x_hat'].detach().clone()
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    clipped_recon_image, _, _, _, _, _, _, _, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="finetune", mode="test", \
                                            feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                            delta=delta_res_stack[sub_i - 1][0], \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                            calrealbits=calrealbits)
                                elif sub_i == i:
                                    clipped_recon_image, _, _, _, _, _, _, _, _, feature, z, delta, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_stage1", mode="test", \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0])
                                    feature_stack.append([feature.detach().clone().requires_grad_(True)])
                                    z_stack.append([z.detach().clone().requires_grad_(True)])
                                    delta_res_stack.append([torch.tensor(delta).detach().clone().requires_grad_(True)])
                                else:
                                    assert (0)
                            ref_image = clipped_recon_image

                    ##################### actual tuning residule of frame i ############
                    cur_params = feature_stack[i - 1] + z_stack[i - 1] + delta_res_stack[i - 1]
                    # cur_params = feature_stack[i - 1] + z_stack[i - 1]
                    optimizer_res = Adam(params=cur_params, lr=sub_lr)
                    lambda_for_test_new = lambda_for_test
                    print("lambda_for_test_new:", lambda_for_test_new)
                    for corr_iter in range(corr_iter_y):
                        for sub_it in range(sub_iter_y):
                            optimizer_res.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                                if sub_i == 0:
                                    I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], delta_I_stack[
                                        0]
                                    result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_for_optim, I_z_for_optim,
                                                      delta_I_for_optim])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), \
                                                   cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    if sub_i < i:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                stage="finetune", mode="test", \
                                                feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                delta=delta_res_stack[sub_i - 1][0], \
                                                mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                                calrealbits=calrealbits)
                                    elif sub_i == i:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=sub_it, total_iter=sub_iter_y,
                                                stage="finetune", mode="training", \
                                                feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                delta=delta_res_stack[sub_i - 1][0], \
                                                mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                                calrealbits=calrealbits)
                                    else:
                                        clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _, _, _, _, _, _ = \
                                            net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                stage="test_for_first", mode="training")
                                    recon_image = clipped_recon_image
                                    bpp = bpp.sum() / h / w
                                ref_image = recon_image
                                if sub_i >= i:
                                    distortion = cal_distoration(sub_cur_frame, recon_image)
                                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new) / frame_range
                                    rd_cost.backward(retain_graph=True)
                            optimizer_res.step()
                            optimizer_res.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    ###################### sub testing start ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        h, w = sub_cur_frame.shape[2], sub_cur_frame.shape[3]
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            I_rec = result['x_hat'].detach().clone()
                            ref_image = I_rec
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy(),\
                                           cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp, "\t", "psnr:", psnr, "\t", "mse:",
                                  distortion, "\t", "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i <= i:
                                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="finetune", mode="test", \
                                            feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                            delta=delta_res_stack[sub_i - 1][0], \
                                            mvfeature=mv_feature_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0],
                                            calrealbits=calrealbits)
                                else:
                                    clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, _, _, _, _, _, _, _, _, _ = \
                                        net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_first", mode="test")
                            ref_image = clipped_recon_image
                            mv_bpp_scalar = bpp_mv.sum()/h/w
                            res_y_bpp_scalar = bpp_feature.sum()/h/w
                            res_z_bpp = bpp_z
                            bpp = mv_bpp_scalar + res_y_bpp_scalar + res_z_bpp
                            distortion = cal_distoration(sub_cur_frame, clipped_recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()

                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_bpp:",
                                  mv_bpp_scalar.cpu().detach().numpy(), "\t", "res_bpp:",
                                  res_y_bpp_scalar.cpu().detach().numpy(), "\t", "res_hyper_bpp:",
                                  res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy().sum()/h/w, "\t", "psnr", psnr, "\t", "mse:",
                                  distortion.cpu().detach().numpy(), "\t",
                                  "rd_cost", rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                    print("total_rd_cost_middle_res {0}:".format(i), total_rd_cost.cpu().detach().numpy())

            '''calculate inference time'''
            torch.cuda.synchronize()
            time_end = time.time()
            time_sum = time_end - time_start
            print("inference time:", time_sum)

            ################### 2nd round edit latent feature #####################
            if iter_res > 0:
                cur_params = []
                for param in I_y_stack + I_z_stack + delta_I_stack:
                    param.requires_grad = True
                    cur_params.append(param)
                for i in range(seqlen - 1):
                    for param in feature_stack[i] + z_stack[i] + delta_res_stack[i]:
                        param.requires_grad = True
                        cur_params.append(param)
                optimizer_fin = Adam(params=cur_params, lr=1e-3)
                scheduler_fin = lr_scheduler.StepLR(optimizer_fin, int(iter_res * 0.9), gamma=0.1)
                for it in range(iter_res):
                    optimizer_fin.zero_grad()
                    for i in range(seqlen):
                        cur_frame = Var(cropped_blocks[i][n].cuda())
                        b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                        num_pixels = b * h * w
                        if i == 0:
                            result = I_codec([cur_frame, "finetune", "training", it, iter_res, I_y_stack[0],
                                            I_z_stack[0], delta_I_stack[0]])
                            recon_image = result['x_hat']
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels)
                            bpp = y_bpp + z_bpp
                        else:
                            clipped_recon_image, _, _, _, _, _, _, bpp, _, _, _, _ = \
                                net(referframe=ref_image, input_image=cur_frame, iter=it, total_iter=iter_res, stage="finetune",
                                    mode="training", \
                                    feature=feature_stack[i - 1][0], z=z_stack[i - 1][0], delta=delta_res_stack[i - 1][0], \
                                    mvfeature=mv_feature_stack[i - 1][0], delta_mv=delta_mv_stack[i - 1][0])
                            recon_image = clipped_recon_image
                            bpp = bpp
                        ref_image = recon_image
                        distortion = cal_distoration(cur_frame, recon_image)
                        rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test).requires_grad_(True)
                        rd_cost = rd_cost / seqlen
                        rd_cost.backward(retain_graph=True)
                    optimizer_fin.step()
                    optimizer_fin.zero_grad()
                    scheduler_fin.step()
                for param in cur_params:
                    param.requires_grad = False
            ####################### formal test ###################################
            rec_block_stack = []
            bits_stack = []
            total_rd_cost_latent_finetune = 0
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                num_pixels = b * h * w
                if i == 0:
                    with torch.no_grad():
                        arr = I_codec(
                            [cur_frame, "test_for_final", "testing", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                    I_rec = arr['x_hat']
                    I_likelihood_y, I_likelihood_z = arr["likelihoods"]['y'], arr["likelihoods"]['z']
                    ref_image = I_rec.clone().detach()
                    y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels)
                    z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                    psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                    bpp = y_bpp + z_bpp
                    rec_block_stack.append(I_rec)

                    distortion = cal_distoration(cur_frame, I_rec)
                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                    print("\n", "I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                          psnr, "\t", "mse:", distortion.cpu().detach().numpy(), "\t", "rd_cost:", rd_cost.cpu().detach().numpy(), "\t",
                          "delta", delta_I_stack[0].cpu().detach().numpy())
                    gop_num += 1
                    sumbpp_real += bpp
                else:
                    with torch.no_grad():
                        clipped_recon_image, _, _, _, bpp_feature, bpp_z, bpp_mv, bpp, real_bpp_feature, real_bpp_z, real_bpp_mv, real_bpp = \
                            net(referframe=ref_image, input_image=cur_frame, iter=0, total_iter=0, stage="finetune",
                                mode="test", \
                                feature=feature_stack[i - 1][0], z=z_stack[i - 1][0], delta=delta_res_stack[i - 1][0],
                                mvfeature=mv_feature_stack[i - 1][0], delta_mv=delta_mv_stack[i - 1][0],
                                calrealbits=calrealbits)
                    ref_image = clipped_recon_image
                    mv_bpp_scalar = bpp_mv.sum()/h/w
                    res_y_bpp_scalar = bpp_feature.sum()/h/w
                    res_z_bpp = bpp_z
                    bpp = mv_bpp_scalar + res_y_bpp_scalar + res_z_bpp
                    distortion = cal_distoration(cur_frame, clipped_recon_image)
                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                    psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()

                    rec_block_stack.append(clipped_recon_image)
                    print("\n", "P{0} frame block {1} after: ".format(i, n), "mv_bpp:", mv_bpp_scalar.cpu().detach().numpy(), "\t",
                          "res_bpp:", res_y_bpp_scalar.cpu().detach().numpy(),
                          "\t", "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                          bpp.cpu().detach().numpy(),
                          "\t", "psnr", psnr, "\t", "mse:", distortion.cpu().detach().numpy(), "\t", "rd_cost",
                          rd_cost.cpu().detach().numpy(), "\t", "delta_res:",
                          delta_res_stack[i - 1][0].cpu().detach().numpy(),
                          "\t", "delta_mv", delta_mv_stack[i - 1][0].cpu().detach().numpy())
                    sumbpp_mv_y += mv_bpp_scalar
                    sumbpp_res_y += res_y_bpp_scalar
                    sumbpp_res_z += res_z_bpp
                    # sumbpp_real += real_bpp
                bits_stack.append(bpp.cpu().detach().numpy() * num_pixels)
                total_rd_cost_latent_finetune += rd_cost
            dir += 1
            print("total_rd_cost_after_res_finetune:", total_rd_cost_latent_finetune.cpu().detach().numpy())

            total_bits_stack.append(bits_stack)
            rec_img_stack.append(rec_block_stack)
        reversed_rec_img_stack = []
        reversed_bits_stack = []
        for x in range(seqlen):
            reversed_block_stack = []
            reversed_block_bits_stack = []
            for y in range(cropped_len):
                reversed_block_stack.append(rec_img_stack[y][x])
                reversed_block_bits_stack.append(total_bits_stack[y][x])

            reversed_rec_img_stack.append(merge(reversed_block_stack, B, C, H, W, factor=factor, overlap=overlap))
            reversed_bits_stack.append(np.sum(reversed_block_bits_stack))

        for p in range(len(reversed_rec_img_stack)):
            distortion = cal_distoration(reversed_rec_img_stack[p], torch.unsqueeze(input_images[p, :, :, :], 0))
            psnr = cal_psnr(distortion)
            bpp = reversed_bits_stack[p] / (B * H * W)
            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
            print("\n", "P{0} frame after: ".format(p), "mv_bpp:", 0, "\t", "mv_hyper_bpp:",
                  0, "\t", "res_bpp:", 0,
                  "\t", "res_hyper_bpp:", 0, "\t", "bpp:",
                  bpp, "\t", "psnr", psnr, "\t", "pre_psnr", 0, "\t", "rd_cost",
                  rd_cost, "\t")
            sumbpp += bpp
            sumpsnr += psnr
            eval_step += 1

    sumbpp /= eval_step
    sumbpp_real /= eval_step
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)
    sumpsnr /= eval_step
    sumpsnr_pre /= (eval_step - gop_num)
    print('\nEpoch {0}  Average MSE={1}  Eval Step={2}\n'.format(str(0), str(avg_loss.data), int(eval_step)))
    log = "HEVC_Class_D  : bpp : %.6lf, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, " \
          " res_y_bpp : %.6lf, res_z_bpp : %.6lf, psnr : %.6lf, psnr_pre : %.6lf\n" % (
              sumbpp, sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr, sumpsnr_pre)
    print(log)


def test_dcvc(net, iter_res, test_dataset, optimize_range, factor, overlap, calrealbits=False):
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=1, batch_size=1, pin_memory=True)
    sumbpp, sumbpp_mv_y, sumbpp_mv_z = 0, 0, 0
    sumbpp_res_y, sumbpp_res_z, sumbpp_real = 0, 0, 0
    sumpsnr, sumpsnr_pre = 0, 0
    eval_step = 0
    gop_num = 0
    avg_loss = torch.zeros(size=[1, ])
    dir = 1
    close_grad(I_codec)
    close_grad(net)
    I_codec.eval()
    net.eval()
    ############################ iterative tuning #############################
    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("[info] testing : %d/%d" % (batch_idx, len(test_loader)))
        input_images = input[0].squeeze(0)
        seqlen = input_images.size()[0]
        cropped_len = len(crop(torch.unsqueeze(input_images[0, :, :, :], 0), factor=factor, overlap=overlap)[0])
        print("number of blocks:", cropped_len)
        B, C, H, W = torch.unsqueeze(input_images[0, :, :, :], 0).shape[0], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[1], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[2], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[3]
        cropped_blocks = []
        for m in range(seqlen):
            cur_frame = torch.unsqueeze(input_images[m, :, :, :], 0)
            cropped_images, _, _, _, _ = crop(cur_frame, factor=factor, overlap=overlap)
            cropped_blocks.append(cropped_images)
        total_bits_stack, rec_img_stack = [], []
        ############################ initial testing ##########################
        total_rd_cost = 0
        for i in range(seqlen):
            cur_frame = Var(torch.unsqueeze(input_images[i, :, :, :].cuda(), 0))
            b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
            num_pixels = b * h * w
            if i == 0:
                with torch.no_grad():
                    arr = I_codec([cur_frame, "test_for_first", "testing"])
                I_rec = arr['x_hat']
                I_likelihood_y, I_likelihood_z = arr["likelihoods"]['y'], arr["likelihoods"]['z']
                ref_image = I_rec.detach().clone()
                y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy()
                z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                bpp = y_bpp + z_bpp
                distortion = cal_distoration(cur_frame, I_rec)
                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                print("\n------------------------------------ GOP {0} --------------------------------------".format(
                    batch_idx + 1))
                print("I frame:  ", "bpp:", bpp, "\t", "psnr:", psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy())
            else:
                with torch.no_grad():
                    result = net(referframe=ref_image, input_image=cur_frame, mode="test", stage="test_for_first",
                                 iter=0, total_iter=0)
                ref_image = result["recon_image"]

                distortion = cal_distoration(cur_frame, result["recon_image"])
                rd_cost = cal_rd_cost(distortion, result["bpp"], lambda_for_test)
                psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                mv_y_bpp = result["bpp_mv_y"]
                mv_z_bpp = result["bpp_mv_z"]
                res_y_bpp = result["bpp_y"]
                res_z_bpp = result["bpp_z"]
                bpp = result["bpp"]
                print("P{0} frame: ".format(i), "mv_y_bpp:", mv_y_bpp.cpu().detach().numpy(),
                      "\t", "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(), "\t", "res_bpp:",
                      res_y_bpp.cpu().detach().numpy(), "\t", "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t",
                      "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                      rd_cost.cpu().detach().numpy())
            total_rd_cost += rd_cost
        print("total_rd_cost_initial:", total_rd_cost.cpu().detach().numpy())

        for n in range(cropped_len):
            ############# initial testing end, tuning start #######################
            I_y_stack, I_z_stack, delta_I_stack = [], [], []
            mv_feature_stack, z_mv_stack, delta_mv_stack = [], [], []
            feature_stack, z_stack, delta_res_stack = [], [], []
            corr_iter_I, corr_iter_w, corr_iter_y = 1, 1, 1
            sub_iter_I, sub_iter_w, sub_iter_y, sub_lr = int(2000 / corr_iter_I), int(400 / corr_iter_w), \
                                                         int(400 / corr_iter_y), 1e-3
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                num_pixels = b * h * w
                if i == 0:
                    ###################### initialize param #######################
                    with torch.no_grad():
                        # use deterministic rounding here
                        arr = I_codec([cur_frame, "test_for_first", "testing"])
                    I_y_stack.append(arr['y'].detach().clone().requires_grad_(True))
                    I_z_stack.append(arr['z'].detach().clone().requires_grad_(True))
                    delta_I_stack.append(torch.tensor(arr["delta"]).clone().detach().requires_grad_(True))

                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_I + 2)], [[] for _ in range(corr_iter_I + 2)]
                    ##################### calculate D, R before tuning ###########################
                    frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                    for sub_i in range(frame_range):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            result = I_codec([sub_cur_frame, "test_for_first", "testing"])
                            recon_image = result['x_hat']
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels)
                            bpp = y_bpp + z_bpp
                        else:
                            result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                         stage="test_for_first", mode="testing")
                            recon_image = result["recon_image"]
                            bpp = result["bpp"]
                        ref_image = recon_image
                        distortion = cal_distoration(sub_cur_frame, recon_image)
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)

                    ##################### actual tuning  I frame ###########################
                    cur_params = I_y_stack + I_z_stack + delta_I_stack
                    optimizer_I = Adam(params=cur_params, lr=sub_lr)
                    lambda_for_test_new = lambda_for_test
                    for corr_iter in range(corr_iter_I):
                        for sub_it in range(sub_iter_I):
                            optimizer_I.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                if sub_i == 0:
                                    result = I_codec([sub_cur_frame, "finetune", "training", sub_it, sub_iter_I,
                                                      I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                        likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                            stage="test_for_first", mode="training")
                                    recon_image = result["recon_image"]
                                    bpp = result["bpp"]
                                ref_image = recon_image
                                distortion = cal_distoration(sub_cur_frame, recon_image)
                                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new) / frame_range
                                rd_cost.backward(retain_graph=True)
                            optimizer_I.step()
                            optimizer_I.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_w + 2)], [[] for _ in range(corr_iter_w + 2)]
                    ###################### sub testing start and calculate R, D before mv tuning ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0],
                                                  delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            ref_image = result['x_hat'].detach().clone()
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y,
                                                   num_pixels=num_pixels).cpu().detach().numpy(), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                            I_rec = result["x_hat"]
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {0} :  ".format(n), "bpp:", bpp, "\t", "psnr:", psnr, "\t", "rd_cost:",
                                  rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                        stage="test_for_first", mode="test")
                            ref_image = result["recon_image"]
                            distortion = cal_distoration(sub_cur_frame, result["recon_image"])
                            rd_cost = cal_rd_cost(distortion, result["bpp"], lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            bpp = result["bpp"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(),
                                  "\t", "res_bpp:", res_y_bpp.cpu().detach().numpy(), "\t", "res_hyper_bpp:",
                                  res_z_bpp.cpu().detach().numpy(), "\t", "bpp:", bpp.cpu().detach().numpy(), "\t",
                                  "psnr", psnr, "\t", "rd_cost", rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                        frame_range = i + optimize_range if i + optimize_range < seqlen else seqlen
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)
                    print("total_rd_cost_middle {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                else:
                    ################## initialize motion vectors of frame i #######
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            ref_image = result['x_hat'].detach().clone()
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="Finetuning_all", mode="test",
                                                 feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0],
                                                 mvfeature=mv_feature_stack[sub_i - 1][0],
                                                 z_mv=z_mv_stack[sub_i - 1][0], delta_mv=delta_mv_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="test_for_first", mode="test")
                                    mv_feature_stack.append([result["mvfeature"].detach().clone().requires_grad_(True)])
                                    z_mv_stack.append([result["z_mv"].detach().clone().requires_grad_(True)])
                                    delta_mv_stack.append([result["delta_mv"].detach().clone().requires_grad_(True)])
                                else:
                                    assert (0)
                            ref_image = result["recon_image"]
                    ##################### actual tuning mv of frame i #############
                    cur_params = mv_feature_stack[i - 1] + z_mv_stack[i - 1] + delta_mv_stack[i - 1]
                    optimizer_mv = Adam(params=cur_params, lr=sub_lr)
                    if i != seqlen - 1:
                        lambda_for_test_new = lambda_for_test
                    else:
                        lambda_for_test_new = lambda_for_test
                    for corr_iter in range(corr_iter_w):
                        for sub_it in range(sub_iter_w):
                            optimizer_mv.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                if sub_i == 0:
                                    I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], \
                                                                                      delta_I_stack[0]
                                    result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_for_optim, I_z_for_optim,
                                                      delta_I_for_optim])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                        likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    if sub_i < i:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                     stage="Finetuning_all", mode="test", feature=feature_stack[sub_i - 1][0],
                                                     z=z_stack[sub_i - 1][0], delta_res=delta_res_stack[sub_i - 1][0],
                                                     mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                     delta_mv=delta_mv_stack[sub_i - 1][0])
                                    elif sub_i == i:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=sub_it,
                                                     total_iter=sub_iter_w, stage="Finetuning_flow", mode="training",
                                                     mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                     delta_mv=delta_mv_stack[sub_i - 1][0])
                                    else:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                     stage="test_for_first", mode="training")
                                    recon_image = result["recon_image"]
                                    bpp = result["bpp"]
                                ref_image = recon_image
                                if sub_i >= i:
                                    distortion = cal_distoration(sub_cur_frame, recon_image)
                                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new) / frame_range
                                    rd_cost.backward(retain_graph=True)
                            optimizer_mv.step()
                            optimizer_mv.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    R_k_stack, D_k_stack = [[] for _ in range(corr_iter_y + 2)], [[] for _ in range(corr_iter_y + 2)]
                    ###################### sub testing start and calculate R, D before residule tuning ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0],
                                                  delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            I_rec = result['x_hat'].detach().clone()
                            ref_image = I_rec
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y,
                                                   num_pixels=num_pixels).cpu().detach().numpy(), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp, "\t", "psnr:", psnr, "\t", "rd_cost:",
                                  rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="Finetuning_all", mode="test", feature=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0], delta_res=delta_res_stack[sub_i - 1][0],
                                                 mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="Finetuning_flow", mode="test",
                                                 mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0])
                                else:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="test_for_first", mode="test")
                            ref_image = result["recon_image"]
                            distortion = cal_distoration(sub_cur_frame, result["recon_image"])
                            rd_cost = cal_rd_cost(distortion, result["bpp"], lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            bpp = result["bpp"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(),
                                  "\tres_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                                  "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                                  rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                        frame_range = i + optimize_range if i + optimize_range < seqlen else seqlen
                        if sub_i in range(i, frame_range):
                            R_k_stack[0].append(bpp)
                            D_k_stack[0].append(distortion)
                    print("total_rd_cost_middle_mv {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                    ################## initialize residule of frame i #############
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(
                                    [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                            ref_image = result['x_hat'].detach().clone()
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="Finetuning_all", mode="test",
                                                 feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0],
                                                 mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="test_for_stage1", mode="test",
                                                 mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0])
                                    feature_stack.append([result["feature"].detach().clone().requires_grad_(True)])
                                    z_stack.append([result["z"].detach().clone().requires_grad_(True)])
                                    delta_res_stack.append([result["delta"].detach().clone().requires_grad_(True)])
                                else:
                                    assert (0)
                            ref_image = result["recon_image"]
                    ##################### actual tuning residule of frame i ############
                    cur_params = feature_stack[i - 1] + z_stack[i - 1] + delta_res_stack[i - 1]
                    optimizer_res = Adam(params=cur_params, lr=sub_lr)
                    if i != seqlen - 1:
                        lambda_for_test_new = lambda_for_test
                    else:
                        lambda_for_test_new = lambda_for_test
                    for corr_iter in range(corr_iter_y):
                        for sub_it in range(sub_iter_y):
                            optimizer_res.zero_grad()
                            frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                            for sub_i in range(frame_range):
                                sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                                if sub_i == 0:
                                    I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], delta_I_stack[
                                        0]
                                    result = I_codec([sub_cur_frame, "finetune", "test", 0, 0, I_y_for_optim, I_z_for_optim,
                                                      delta_I_for_optim])
                                    recon_image = result['x_hat']
                                    I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                                    y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                        likelihood=I_likelihood_z, num_pixels=num_pixels)
                                    bpp = y_bpp + z_bpp
                                else:
                                    if sub_i < i:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                     stage="Finetuning_all", mode="test", feature=feature_stack[sub_i - 1][0],
                                                     z=z_stack[sub_i - 1][0], delta_res=delta_res_stack[sub_i - 1][0],
                                                     mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                     delta_mv=delta_mv_stack[sub_i - 1][0])
                                    elif sub_i == i:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=sub_it,
                                                     total_iter=sub_iter_y, stage="Finetuning_all", mode="training",
                                                     feature=feature_stack[sub_i - 1][0], z=z_stack[sub_i - 1][0],
                                                     delta_res=delta_res_stack[sub_i - 1][0],
                                                     mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                     delta_mv=delta_mv_stack[sub_i - 1][0])
                                    else:
                                        result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                     stage="test_for_first", mode="training")
                                    recon_image = result["recon_image"]
                                    bpp = result["bpp"]
                                ref_image = recon_image
                                if sub_i >= i:
                                    distortion = cal_distoration(sub_cur_frame, recon_image)
                                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test_new) / frame_range
                                    rd_cost.backward(retain_graph=True)
                            optimizer_res.step()
                            optimizer_res.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    ###################### sub testing start ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec( [sub_cur_frame, "finetune", "test", 0, 0, I_y_stack[0], I_z_stack[0],
                                                   delta_I_stack[0]])
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            I_rec = result['x_hat'].detach().clone()
                            ref_image = I_rec
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y,
                                                   num_pixels=num_pixels).cpu().detach().numpy(), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = y_bpp + z_bpp
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp, "\t", "psnr:", psnr, "\t", "rd_cost:",
                                  rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i <= i:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="Finetuning_all", mode="test", feature=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0], delta_res=delta_res_stack[sub_i - 1][0],
                                                 mvfeature=mv_feature_stack[sub_i - 1][0], z_mv=z_mv_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0])
                                else:
                                    result = net(referframe=ref_image, input_image=sub_cur_frame, iter=0, total_iter=0,
                                                 stage="test_for_first", mode="test")
                            ref_image = result["recon_image"]
                            distortion = cal_distoration(sub_cur_frame, result["recon_image"])
                            rd_cost = cal_rd_cost(distortion, result["bpp"], lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            bpp = result["bpp"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:",
                                  mv_z_bpp.cpu().detach().numpy(), "\t", "res_bpp:", res_y_bpp.cpu().detach().numpy(),
                                  "\t", "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                                  rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                    print("total_rd_cost_middle_res {0}:".format(i), total_rd_cost.cpu().detach().numpy())
            ################### 2nd round edit latent feature #####################
            if iter_res > 0:
                cur_params = []
                for param in I_y_stack + I_z_stack + delta_I_stack:
                    param.requires_grad = True
                    cur_params.append(param)
                for i in range(seqlen - 1):
                    for param in feature_stack[i] + z_stack[i] + delta_res_stack[i]:
                        param.requires_grad = True
                        cur_params.append(param)
                optimizer_fin = Adam(params=cur_params, lr=1e-3)
                scheduler_fin = lr_scheduler.StepLR(optimizer_fin, int(iter_res * 0.9), gamma=0.1)
                for it in range(iter_res):
                    optimizer_fin.zero_grad()
                    for i in range(seqlen):
                        cur_frame = Var(cropped_blocks[i][n].cuda())
                        b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                        num_pixels = b * h * w
                        if i == 0:
                            result = I_codec([cur_frame, "finetune", "training", it, iter_res, I_y_stack[0],
                                              I_z_stack[0], delta_I_stack[0]])
                            recon_image = result['x_hat']
                            I_likelihood_y, I_likelihood_z = result["likelihoods"]['y'], result["likelihoods"]['z']
                            y_bpp, z_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels), cal_bpp(
                                likelihood=I_likelihood_z, num_pixels=num_pixels)
                            bpp = y_bpp + z_bpp
                        else:
                            result = net(referframe=ref_image, input_image=cur_frame, iter=it, total_iter=iter_res,
                                         stage="Finetuning_res", mode="training", feature=feature_stack[i - 1][0],
                                         z=z_stack[i - 1][0], delta_res=delta_res_stack[i - 1][0],
                                         mvfeature=mv_feature_stack[i - 1][0], z_mv=z_mv_stack[i - 1][0],
                                         delta_mv=delta_mv_stack[i - 1][0])
                            recon_image = result["recon_image"]
                            bpp = result["bpp"]
                        ref_image = recon_image
                        distortion = cal_distoration(cur_frame, recon_image)
                        rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test).requires_grad_(True)
                        rd_cost = rd_cost / seqlen
                        rd_cost.backward(retain_graph=True)
                    optimizer_fin.step()
                    optimizer_fin.zero_grad()
                    scheduler_fin.step()
                for param in cur_params:
                    param.requires_grad = False
            ####################### formal test ###################################
            rec_block_stack = []
            bits_stack = []
            total_rd_cost_latent_finetune = 0
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                num_pixels = b * h * w
                if i == 0:
                    with torch.no_grad():
                        arr = I_codec(
                            [cur_frame, "test_for_final", "testing", 0, 0, I_y_stack[0], I_z_stack[0], delta_I_stack[0]])
                    I_rec = arr['x_hat']
                    I_likelihood_y, I_likelihood_z = arr["likelihoods"]['y'], arr["likelihoods"]['z']
                    ref_image = I_rec.clone().detach()
                    y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels)
                    z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels)
                    psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                    bpp = y_bpp + z_bpp
                    rec_block_stack.append(I_rec)

                    distortion = cal_distoration(cur_frame, I_rec)
                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                    print("\n", "I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                          psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy(), "\t", "delta",
                          delta_I_stack[0].cpu().detach().numpy())
                    gop_num += 1
                    sumbpp_real += bpp
                else:
                    with torch.no_grad():
                        result = net(referframe=ref_image, input_image=cur_frame, iter=0, total_iter=0,
                                     stage="Finetuning_all", mode="test", feature=feature_stack[i - 1][0],
                                     z=z_stack[i - 1][0], delta_res=delta_res_stack[i - 1][0],
                                     mvfeature=mv_feature_stack[i - 1][0], z_mv=z_mv_stack[i - 1][0],
                                     delta_mv=delta_mv_stack[i - 1][0])
                    ref_image = result["recon_image"]
                    distortion = cal_distoration(cur_frame, result["recon_image"])
                    rd_cost = cal_rd_cost(distortion, result["bpp"], lambda_for_test)
                    psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                    mv_y_bpp = result["bpp_mv_y"]
                    mv_z_bpp = result["bpp_mv_z"]
                    res_y_bpp = result["bpp_y"]
                    res_z_bpp = result["bpp_z"]
                    bpp = result["bpp"]
                    rec_block_stack.append(result["recon_image"])
                    print("\nP{0} frame block {1} after: ".format(i, n),
                          "mv_y_bpp:", mv_y_bpp.cpu().detach().numpy(), "\t",
                          "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(), "\t",
                          "res_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                          "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t",
                          "bpp:", bpp.cpu().detach().numpy(), "\t",
                          "psnr", psnr, "\t",
                          "rd_cost", rd_cost.cpu().detach().numpy(), "\t",
                          "delta_res:", delta_res_stack[i - 1][0].cpu().detach().numpy(), "\t",
                          "delta_mv", delta_mv_stack[i - 1][0].cpu().detach().numpy())
                    sumbpp_mv_y += mv_y_bpp
                    sumbpp_mv_z += mv_z_bpp
                    sumbpp_res_y += res_y_bpp
                    sumbpp_res_z += res_z_bpp
                bits_stack.append(bpp.cpu().detach().numpy() * num_pixels)
                total_rd_cost_latent_finetune += rd_cost
            dir += 1
            print("total_rd_cost_after_res_finetune:", total_rd_cost_latent_finetune.cpu().detach().numpy())

            total_bits_stack.append(bits_stack)
            rec_img_stack.append(rec_block_stack)
        reversed_rec_img_stack = []
        reversed_bits_stack = []
        for x in range(seqlen):
            reversed_block_stack = []
            reversed_block_bits_stack = []
            for y in range(cropped_len):
                reversed_block_stack.append(rec_img_stack[y][x])
                reversed_block_bits_stack.append(total_bits_stack[y][x])

            reversed_rec_img_stack.append(merge(reversed_block_stack, B, C, H, W, factor=factor, overlap=overlap))
            reversed_bits_stack.append(np.sum(reversed_block_bits_stack))

        for p in range(len(reversed_rec_img_stack)):
            distortion = cal_distoration(reversed_rec_img_stack[p], torch.unsqueeze(input_images[p, :, :, :], 0))
            psnr = cal_psnr(distortion)
            bpp = reversed_bits_stack[p] / (B * H * W)
            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
            print("\n", "P{0} frame after: ".format(p), "mv_bpp:", 0, "\t", "mv_hyper_bpp:",
                  0, "\t", "res_bpp:", 0,
                  "\t", "res_hyper_bpp:", 0, "\t", "bpp:",
                  bpp, "\t", "psnr", psnr, "\t", "pre_psnr", 0, "\t", "rd_cost",
                  rd_cost, "\t")
            sumbpp += bpp
            sumpsnr += psnr
            eval_step += 1

    sumbpp /= eval_step
    sumbpp_real /= eval_step
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)
    sumpsnr /= eval_step
    sumpsnr_pre /= (eval_step - gop_num)
    print('\nEpoch {0}  Average MSE={1}  Eval Step={2}\n'.format(str(0), str(avg_loss.data), int(eval_step)))
    log = "HEVC_Class_D  : bpp : %.6lf, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, " \
          " res_y_bpp : %.6lf, res_z_bpp : %.6lf, psnr : %.6lf, psnr_pre : %.6lf\n" % (
              sumbpp, sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr, sumpsnr_pre)
    print(log)


def test_hstem(net, test_dataset, factor, overlap, optimize_range, iter_res):
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=1, batch_size=1, pin_memory=True)
    sumbpp, sumbpp_mv_y, sumbpp_mv_z = 0, 0, 0
    sumbpp_res_y, sumbpp_res_z, sumbpp_real = 0, 0, 0
    sumpsnr, sumpsnr_pre = 0, 0
    eval_step = 0
    gop_num = 0
    avg_loss = torch.zeros(size=[1, ])
    dir = 1
    close_grad(I_codec)
    close_grad(net)
    I_codec.eval()
    net.eval()
    ############################ iterative tuning #############################
    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("[info] testing : %d/%d" % (batch_idx, len(test_loader)))
        input_images = input[0].squeeze(0)
        seqlen = input_images.size()[0]
        cropped_len = len(crop(torch.unsqueeze(input_images[0, :, :, :], 0), factor=factor, overlap=overlap)[0])
        print("number of blocks:", cropped_len)
        B, C, H, W = torch.unsqueeze(input_images[0, :, :, :], 0).shape[0], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[1], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[2], \
                     torch.unsqueeze(input_images[0, :, :, :], 0).shape[3]
        cropped_blocks = []
        for m in range(seqlen):
            cur_frame = torch.unsqueeze(input_images[m, :, :, :], 0)
            cropped_images, _, _, _, _ = crop(cur_frame, factor=factor, overlap=overlap)
            cropped_blocks.append(cropped_images)
        total_bits_stack, rec_img_stack = [], []
        ############################ initial testing ##########################
        total_rd_cost = 0
        for i in range(seqlen):
            cur_frame = Var(torch.unsqueeze(input_images[i, :, :, :].cuda(), 0))
            if i == 0:
                with torch.no_grad():
                    result = I_codec(x=cur_frame, stage="test_for_first", mode="test", delta_I=i_q_scale)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                I_rec = result["x_hat"]
                psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                bpp = result["bpp"]
                distortion = cal_distoration(cur_frame, I_rec)
                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)

                print("\nI frame initial: ", "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:", psnr,
                      "\trd_cost:", rd_cost.cpu().detach().numpy(), "\ty_q_scale:",
                      result["delta_I"].cpu().detach().numpy())

            else:
                with torch.no_grad():
                    result = net(x=cur_frame, dpb=dpb, stage="test_for_first", mode="test", delta_mv=mv_y_q_scale,
                                 delta_res=y_q_scale)
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]

                distortion = cal_distoration(cur_frame, recon_frame)
                psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                mv_y_bpp = result["bpp_mv_y"]
                mv_z_bpp = result["bpp_mv_z"]
                res_y_bpp = result["bpp_y"]
                res_z_bpp = result["bpp_z"]
                bpp = result["bpp"]
                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)

                print("P{0} frame initial: ".format(i), "mv_y_bpp:",
                      mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:",
                      mv_z_bpp.cpu().detach().numpy(), "\t", "res_y_bpp:", res_y_bpp.cpu().detach().numpy(),
                      "\t", "res_z_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                      bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t",
                      "rd_cost", rd_cost.cpu().detach().numpy())
            total_rd_cost += rd_cost
        print("total_rd_cost_initial:", total_rd_cost.cpu().detach().numpy())

        for n in range(cropped_len):
            ############# initial testing end, tuning start #######################
            I_y_stack, I_z_stack, delta_I_stack = [], [], []
            mv_feature_stack, z_mv_feature_stack, delta_mv_stack = [], [], []
            feature_stack, z_stack, delta_res_stack = [], [], []
            sub_iter_I, sub_iter_w, sub_iter_y, sub_lr = 2000, 400, 400, 1e-3
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                if i == 0:
                    ###################### initialize param #######################
                    with torch.no_grad():
                        # use deterministic rounding here
                        arr = I_codec(x=cur_frame, stage="test_for_first", mode="test", delta_I=i_q_scale)
                    I_y_stack.append(arr['y_'].detach().clone().requires_grad_(True))
                    I_z_stack.append(arr['z'].detach().clone().requires_grad_(True))
                    delta_I_stack.append(torch.tensor(arr["delta_I"]).clone().detach().requires_grad_(True))
                    ##################### actual tuning ###########################
                    cur_params = I_y_stack + I_z_stack + delta_I_stack
                    optimizer_I = Adam(params=cur_params, lr=sub_lr)
                    for sub_it in tqdm(range(sub_iter_I)):
                        optimizer_I.zero_grad()
                        frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                        for sub_i in range(frame_range):
                            sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                            if sub_i == 0:
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="training", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0], it=sub_it,
                                                 total_it=sub_iter_I, force_sga=True)
                                recon_image = result['x_hat']
                                bpp = result["bpp"]
                                dpb = {
                                    "ref_frame": result["x_hat"],
                                    "ref_feature": None,
                                    "ref_y": None,
                                    "ref_mv_y": None,
                                }
                            else:
                                result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="training",
                                             delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                                recon_image = result["dpb"]["ref_frame"]
                                bpp = result["bpp"]
                                dpb = result["dpb"]
                            distortion = cal_distoration(sub_cur_frame, recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test) / frame_range
                            rd_cost.backward(retain_graph=True)
                        optimizer_I.step()
                        optimizer_I.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    ###################### sub testing start ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0])
                            bpp = result["bpp"]
                            dpb = {
                                "ref_frame": result["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                            I_rec = result["x_hat"]
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {0} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                                  psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="test",
                                             delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                            recon_image = result["dpb"]["ref_frame"]
                            dpb = result["dpb"]
                            bpp = result["bpp"]
                            distortion = cal_distoration(sub_cur_frame, recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(),
                                  "\t", "res_y_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                                  "res_z_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                                  rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                    print("total_rd_cost_middle {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                else:
                    ################## initialize motion vectors of frame i #######
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0])
                            dpb = {
                                "ref_frame": result["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="test",
                                                 delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                                    mv_feature_stack.append([result["mv_y_"].detach().clone().requires_grad_(True)])
                                    z_mv_feature_stack.append([result["mv_z"].detach().clone().requires_grad_(True)])
                                    delta_mv_stack.append([torch.tensor(result["delta_mv"]).detach().clone().requires_grad_(True)])
                                else:
                                    assert (0)
                            dpb = result["dpb"]
                    ##################### actual tuning mv of frame i #############
                    cur_params = mv_feature_stack[i - 1] + z_mv_feature_stack[i - 1] + delta_mv_stack[i - 1]
                    optimizer_mv = Adam(params=cur_params, lr=sub_lr)
                    for sub_it in tqdm(range(sub_iter_w)):
                        optimizer_mv.zero_grad()
                        frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                        for sub_i in range(frame_range):
                            sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                            if sub_i == 0:
                                I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], \
                                                                                  delta_I_stack[0]
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_for_optim,
                                                 z=I_z_for_optim, delta_I=delta_I_for_optim)
                                dpb = {
                                    "ref_frame": result["x_hat"],
                                    "ref_feature": None,
                                    "ref_y": None,
                                    "ref_mv_y": None,
                                }
                            else:
                                if sub_i < i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune_flow", mode="training",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=y_q_scale,
                                                 it=sub_it, total_it=sub_iter_w, force_sga=True)
                                else:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="training",
                                                 delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                                recon_image = result["dpb"]["ref_frame"]
                                dpb = result["dpb"]
                                bpp = result["bpp"]
                            if sub_i >= i:
                                distortion = cal_distoration(sub_cur_frame, recon_image)
                                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test) / frame_range
                                rd_cost.backward(retain_graph=True)
                        optimizer_mv.step()
                        optimizer_mv.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    ###################### sub testing start ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0])
                            dpb = {
                                "ref_frame": result["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                            I_rec = result['x_hat'].detach().clone()
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = result["bpp"]
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                                  psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune_flow", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=y_q_scale)
                                else:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="test",
                                                 delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                            recon_image = result["dpb"]["ref_frame"]
                            bpp = result["bpp"]
                            dpb = result["dpb"]
                            distortion = cal_distoration(sub_cur_frame, recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:", mv_z_bpp.cpu().detach().numpy(),
                                  "\t", "res_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                                  "res_hyper_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                                  rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                    print("total_rd_cost_middle_mv {0}:".format(i), total_rd_cost.cpu().detach().numpy())
                    ################## initialize residule of frame i #############
                    for sub_i in range(i + 1):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0])
                            dpb = {
                                "ref_frame": result["x_hat"].detach().clone(),
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                        else:
                            with torch.no_grad():
                                if sub_i < i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune_flow", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=y_q_scale)
                                    feature_stack.append([result["y_"].detach().clone().requires_grad_(True)])
                                    z_stack.append([result["z"].detach().clone().requires_grad_(True)])
                                    delta_res_stack.append([torch.tensor(result["delta_res"]).detach().clone().requires_grad_(True)])
                                else:
                                    assert (0)
                                dpb = result["dpb"]
                    ##################### actual tuning res of frame i ############
                    cur_params = feature_stack[i - 1] + z_stack[i - 1] + delta_res_stack[i - 1]
                    # cur_params = delta_res_stack[i - 1]
                    optimizer_res = Adam(params=cur_params, lr=sub_lr)
                    for sub_it in tqdm(range(sub_iter_y)):
                        optimizer_res.zero_grad()
                        frame_range = (i + optimize_range) if (i + optimize_range) < seqlen else seqlen
                        for sub_i in range(frame_range):
                            sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                            if sub_i == 0:
                                I_y_for_optim, I_z_for_optim, delta_I_for_optim = I_y_stack[0], I_z_stack[0], \
                                                                                  delta_I_stack[0]
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_for_optim,
                                                 z=I_z_for_optim, delta_I=delta_I_for_optim)
                                recon_image = result['x_hat']
                                bpp = result["bpp"]
                                dpb = {
                                    "ref_frame": result["x_hat"],
                                    "ref_feature": None,
                                    "ref_y": None,
                                    "ref_mv_y": None,
                                }
                            else:
                                if sub_i < i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                elif sub_i == i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="training",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0],
                                                 it=sub_it, total_it=sub_iter_y, force_sga=True)
                                else:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="training",
                                                 delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                                recon_image = result["dpb"]["ref_frame"]
                                dpb = result["dpb"]
                                bpp = result["bpp"]
                            if sub_i >= i:
                                distortion = cal_distoration(sub_cur_frame, recon_image)
                                rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test) / frame_range
                                rd_cost.backward(retain_graph=True)
                        optimizer_res.step()
                        optimizer_res.zero_grad()

                    for param in cur_params:
                        param.requires_grad = False
                    ###################### sub testing start ######################
                    total_rd_cost = 0
                    for sub_i in range(seqlen):
                        sub_cur_frame = Var(cropped_blocks[sub_i][n].cuda())
                        if sub_i == 0:
                            with torch.no_grad():
                                result = I_codec(x=sub_cur_frame, stage="finetune", mode="test", y_=I_y_stack[0],
                                                 z=I_z_stack[0], delta_I=delta_I_stack[0])
                            I_rec = result['x_hat']
                            psnr = cal_psnr(distortion=cal_distoration(I_rec, sub_cur_frame)).cpu().detach().numpy()
                            bpp = result["bpp"]
                            dpb = {
                                "ref_frame": result["x_hat"].detach().clone(),
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                            distortion = cal_distoration(sub_cur_frame, I_rec)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            print(
                                "\n------------------------------------ GOP {0} --------------------------------------".format(
                                    batch_idx + 1))
                            print("I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                                  psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy())
                        else:
                            with torch.no_grad():
                                if sub_i <= i:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="finetune", mode="test",
                                                 mv_y_=mv_feature_stack[sub_i - 1][0],
                                                 mv_z=z_mv_feature_stack[sub_i - 1][0],
                                                 y_=feature_stack[sub_i - 1][0],
                                                 z=z_stack[sub_i - 1][0],
                                                 delta_mv=delta_mv_stack[sub_i - 1][0],
                                                 delta_res=delta_res_stack[sub_i - 1][0])
                                else:
                                    result = net(x=sub_cur_frame, dpb=dpb, stage="test_for_first", mode="test",
                                                 delta_mv=mv_y_q_scale, delta_res=y_q_scale)
                            recon_image = result["dpb"]["ref_frame"]
                            bpp = result["bpp"]
                            dpb = result["dpb"]
                            distortion = cal_distoration(sub_cur_frame, recon_image)
                            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                            psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                            mv_y_bpp = result["bpp_mv_y"]
                            mv_z_bpp = result["bpp_mv_z"]
                            res_y_bpp = result["bpp_y"]
                            res_z_bpp = result["bpp_z"]
                            print("P{0} frame block {1} : ".format(sub_i, n), "mv_y_bpp:",
                                  mv_y_bpp.cpu().detach().numpy(), "\t", "mv_z_bpp:",
                                  mv_z_bpp.cpu().detach().numpy(), "\t", "res_y_bpp:", res_y_bpp.cpu().detach().numpy(),
                                  "\t", "res_z_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:",
                                  bpp.cpu().detach().numpy(), "\t", "psnr", psnr, "\t", "\t", "rd_cost",
                                  rd_cost.cpu().detach().numpy())
                        total_rd_cost += rd_cost
                    print("total_rd_cost_middle_res {0}:".format(i), total_rd_cost.cpu().detach().numpy())
            ################### 2nd round edit latent feature #####################
            if iter_res > 0:
                cur_params = []
                for param in I_y_stack + I_z_stack + delta_I_stack:
                    param.requires_grad = True
                    cur_params.append(param)
                for i in range(seqlen - 1):
                    for param in feature_stack[i] + z_stack[i] + delta_res_stack[i]:
                        param.requires_grad = True
                        cur_params.append(param)
                optimizer_fin = Adam(params=cur_params, lr=1e-3)
                scheduler_fin = lr_scheduler.StepLR(optimizer_fin, int(iter_res * 0.9), gamma=0.1)
                for it in range(iter_res):
                    optimizer_fin.zero_grad()
                    for i in range(seqlen):
                        cur_frame = Var(cropped_blocks[i][n].cuda())
                        if i == 0:
                            result = I_codec(x=cur_frame, stage="finetune", mode="training", y=I_y_stack[0],
                                             z=I_z_stack[0], delta_I=delta_I_stack[0], it=it, total_it=iter_res,
                                             force_sga=True)
                            recon_image = result['x_hat']
                            bpp = result["bpp"]
                            dpb = {
                                "ref_frame": result["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                        else:
                            result = net(x=cur_frame, dpb=dpb, stage="finetune", mode="training",
                                         mv_y=mv_feature_stack[i - 1][0],
                                         mv_z=z_mv_feature_stack[i - 1][0],
                                         y=feature_stack[i - 1][0],
                                         z=z_stack[i - 1][0],
                                         delta_mv=delta_mv_stack[i - 1][0],
                                         delta_res=delta_res_stack[i - 1][0],
                                         it=it, total_it=iter_res)
                            recon_image = result["dpb"]["ref_frame"]
                            bpp = result["bpp"]
                            dpb = result["dpb"]
                        distortion = cal_distoration(cur_frame, recon_image)
                        rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test).requires_grad_(True)
                        rd_cost = rd_cost / seqlen
                        rd_cost.backward(retain_graph=True)
                    optimizer_fin.step()
                    optimizer_fin.zero_grad()
                    scheduler_fin.step()
                for param in cur_params:
                    param.requires_grad = False
            ###################### formal test ###################################
            rec_block_stack = []
            bits_stack = []
            total_rd_cost_latent_finetune = 0
            for i in range(seqlen):
                cur_frame = Var(cropped_blocks[i][n].cuda())
                b, h, w = cur_frame.shape[0], cur_frame.shape[2], cur_frame.shape[3]
                num_pixels = b * h * w
                if i == 0:
                    with torch.no_grad():
                        arr = I_codec(x=cur_frame, stage="finetune", mode="test", y_=I_y_stack[0], z=I_z_stack[0],
                                      delta_I=delta_I_stack[0])
                    I_rec = arr['x_hat']
                    dpb = {
                        "ref_frame": arr["x_hat"].clone().detach(),
                        "ref_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                    }
                    psnr = cal_psnr(distortion=cal_distoration(I_rec, cur_frame)).cpu().detach().numpy()
                    bpp = arr["bpp"]
                    rec_block_stack.append(I_rec)

                    distortion = cal_distoration(cur_frame, I_rec)
                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                    print("\n", "I frame block {} :  ".format(n), "bpp:", bpp.cpu().detach().numpy(), "\t", "psnr:",
                          psnr, "\t", "rd_cost:", rd_cost.cpu().detach().numpy(), "\t", "delta",
                          delta_I_stack[0].cpu().detach().numpy())
                    gop_num += 1
                    sumbpp_real += bpp
                else:
                    with torch.no_grad():
                        result = net(x=cur_frame, dpb=dpb, stage="finetune", mode="test",
                                     mv_y_=mv_feature_stack[i - 1][0],
                                     mv_z=z_mv_feature_stack[i - 1][0],
                                     y_=feature_stack[i - 1][0],
                                     z=z_stack[i - 1][0],
                                     delta_mv=delta_mv_stack[i - 1][0],
                                     delta_res=delta_res_stack[i - 1][0])
                    recon_image = result["dpb"]["ref_frame"]
                    dpb = result["dpb"]
                    bpp = result["bpp"]
                    distortion = cal_distoration(cur_frame, recon_image)
                    rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
                    psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                    mv_y_bpp = result["bpp_mv_y"]
                    mv_z_bpp = result["bpp_mv_z"]
                    res_y_bpp = result["bpp_y"]
                    res_z_bpp = result["bpp_z"]
                    rec_block_stack.append(recon_image)
                    print("\n", "P{0} frame block {1} after: ".format(i, n), "mv_y_bpp:",
                          mv_y_bpp.cpu().detach().numpy(), "\t", "res_y_bpp:", res_y_bpp.cpu().detach().numpy(), "\t",
                          "res_z_bpp:", res_z_bpp.cpu().detach().numpy(), "\t", "bpp:", bpp.cpu().detach().numpy(),
                          "\t", "psnr", psnr, "\t", "\t", "rd_cost", rd_cost.cpu().detach().numpy(), "\t", "delta_res:",
                          delta_res_stack[i - 1][0].cpu().detach().numpy(), "\t", "delta_mv",
                          delta_mv_stack[i - 1][0].cpu().detach().numpy())
                    sumbpp_mv_y += mv_y_bpp
                    sumbpp_mv_z += mv_z_bpp
                    sumbpp_res_y += res_y_bpp
                    sumbpp_res_z += res_z_bpp
                bits_stack.append(bpp.cpu().detach().numpy() * num_pixels)
                total_rd_cost_latent_finetune += rd_cost

            dir += 1
            print("total_rd_cost_after_res_finetune:", total_rd_cost_latent_finetune.cpu().detach().numpy())

            total_bits_stack.append(bits_stack)
            rec_img_stack.append(rec_block_stack)
        reversed_rec_img_stack = []
        reversed_bits_stack = []
        for x in range(seqlen):
            reversed_block_stack = []
            reversed_block_bits_stack = []
            for y in range(cropped_len):
                reversed_block_stack.append(rec_img_stack[y][x])
                reversed_block_bits_stack.append(total_bits_stack[y][x])

            reversed_rec_img_stack.append(merge(reversed_block_stack, B, C, H, W, factor=factor, overlap=overlap))
            reversed_bits_stack.append(np.sum(reversed_block_bits_stack))

        for p in range(len(reversed_rec_img_stack)):
            distortion = cal_distoration(reversed_rec_img_stack[p], torch.unsqueeze(input_images[p, :, :, :], 0))
            psnr = cal_psnr(distortion)
            bpp = reversed_bits_stack[p] / (B * H * W)
            rd_cost = cal_rd_cost(distortion, bpp, lambda_for_test)
            print("\n", "P{0} frame after: ".format(p), "mv_bpp:", 0, "\t", "mv_hyper_bpp:",
                  0, "\t", "res_bpp:", 0,
                  "\t", "res_hyper_bpp:", 0, "\t", "bpp:",
                  bpp, "\t", "psnr", psnr, "\t", "pre_psnr", 0, "\t", "rd_cost",
                  rd_cost, "\t")
            sumbpp += bpp
            sumpsnr += psnr
            eval_step += 1

    sumbpp /= eval_step
    sumbpp_real /= eval_step
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)
    sumpsnr /= eval_step
    sumpsnr_pre /= (eval_step - gop_num)
    print('\nEpoch {0}  Average MSE={1}  Eval Step={2}\n'.format(str(0), str(avg_loss.data), int(eval_step)))
    log = "HEVC_Class_D  : bpp : %.6lf, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, " \
          " res_y_bpp : %.6lf, res_z_bpp : %.6lf, psnr : %.6lf, psnr_pre : %.6lf\n" % (
              sumbpp, sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr, sumpsnr_pre)
    print(log)


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--test_model', type=str, default="DCVC", help="choose one from: DVC, DCVC, HSTEM")
    parser.add_argument('--test_lambdas', type=int, nargs="+", default=(2048, 1024, 512, 256))
    parser.add_argument("--iter_res", type=int, default=0, help="2nd round edit latent feature")
    parser.add_argument("--factor", type=int, default=100, help="block size = (factor * 64, factor * 64)")
    parser.add_argument('--overlap',  type=int, default=0, help="overlap area between cropped blocks")
    parser.add_argument('--test_class', type=str, default="HEVC_D", help="dataset to be tested")
    parser.add_argument('--gop_size', type=int, default=10, help="GoP size, 10 for HEVC, 12 for UVG")
    parser.add_argument('--test_gop_num', type=int, default=1, help="GoP number to be tested in each sequence")
    parser.add_argument('--optimize_range', type=int, default=1, help="likelihood range (for spatial complexity reduction)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    global test_dataset
    if args.test_class[:3] == "UVG":
        test_dataset = UVGDataSet(gop_size=args.gop_size, test_gop_num=args.test_gop_num)
    elif args.test_class[:4] == "HEVC":
        test_dataset = HEVCDataSet(class_=args.test_class[-1], gop_size=args.gop_size, test_gop_num=args.test_gop_num)
    else:
        print("\nThere is no dataset named", args.test_class, "!")
        exit(-1)
    print("Now the tested dataset is:", args.test_class)

    for i in range(len(args.test_lambdas)):
        lambda_for_test = args.test_lambdas[i]
        print("================================ testing lambda: {} =====================================".format(lambda_for_test))

        I_codec_checkpoint = torch.load(
            './Checkpoints/cheng2020-anchor-{0}.pth.tar'.format(lambda_quality_map[lambda_for_test]),
            map_location=torch.device('cpu'))
        I_codec = architectures['cheng2020-anchor'].from_state_dict(I_codec_checkpoint).cuda()

        if args.test_model == "DVC":
            model = VideoCompressor()
            pretrained_dict = torch.load("./Checkpoints/DVC_{0}.model".format(lambda_for_test))
            model.load_state_dict(pretrained_dict, strict=False)
            model = model.cuda()
            print("Number of Total Parameters:", sum(x.numel() for x in model.parameters()))
            test_dvc(model, args.iter_res, test_dataset, args.optimize_range, args.factor, args.overlap)
        elif args.test_model == "DCVC":
            model = DCVC_net()
            pretrained_dict = torch.load("./Checkpoints/DCVC_{0}.pth".format(lambda_for_test))
            model.load_state_dict(pretrained_dict, strict=False)
            model = model.cuda()
            print("Number of Total Parameters:", sum(x.numel() for x in model.parameters()))
            test_dcvc(model, args.iter_res, test_dataset, args.optimize_range, args.factor, args.overlap)
        elif args.test_model == "HSTEM":
            pretrained_i = torch.load("./Checkpoints/acmmm2022_image_psnr.pth.tar")
            i_q_scales = pretrained_i["q_scale"].reshape(-1)
            pretrained_p = torch.load("./Checkpoints/acmmm2022_video_psnr.pth.tar")
            y_q_scales = pretrained_p["y_q_scale"].reshape(-1)
            mv_y_q_scales = pretrained_p["mv_y_q_scale"].reshape(-1)
            i_q_scale = i_q_scales[i]
            y_q_scale = y_q_scales[i]
            mv_y_q_scale = mv_y_q_scales[i]

            I_codec = IntraNoAR()
            pretrained_I_dict = torch.load("./Checkpoints/acmmm2022_image_psnr.pth.tar")
            I_codec.load_state_dict(pretrained_I_dict, strict=False)
            I_codec = I_codec.cuda().eval()
            model = DMC()
            pretrained_dict = torch.load("./Checkpoints/acmmm2022_video_psnr.pth.tar")
            model.load_state_dict(pretrained_dict, strict=False)
            model = model.cuda()
            print("Number of Total Parameters:", sum(x.numel() for x in model.parameters()))
            test_hstem(model, test_dataset, args.factor, args.overlap, args.optimize_range, args.iter_res)
        else:
            print("\nThere is no model named", args.test_model, "!")
            exit(-1)
    exit(0)
