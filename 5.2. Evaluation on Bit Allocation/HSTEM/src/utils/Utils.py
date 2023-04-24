import argparse
import datetime
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # for loading data
    parser.add_argument("--train_h5", type=str, default='/home/zhaoyu/FVC_TU/Dataset/Train_Raw/', help="train dataset h5 file")
    parser.add_argument("--eval_h5", type=str, default='/home/zhaoyu/FVC_TU/Dataset/Eval_Raw/', help="eval dataset h5 file")
    # parser.add_argument("--pretrained_model", type=str, help="checkpoint file path")
    # parser.add_argument("--pretrained_model", type=str, default='Experiments/2022-08-13_16-20/Checkpoints/DVC_003.pth',
    #                     help="checkpoint file path")
    parser.add_argument("--pretrained_model", type=str, default='./checkpoints/acmmm2022_video_psnr.pth.tar',
                        help="checkpoint file path")
    parser.add_argument("--pretrained_stage", type=str, default='ME', help="checkpoint optimizer stage")

    # for training
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--gpu", action='store_true', default=True, help="use gpu or cpu")
    parser.add_argument("--lambda_weight", type=float, nargs="+", default=(85, 170, 380, 840), help="weights for distortion")  # 85, 170, 380, 840
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")  # 1e-4 for ACMMM 2022 (MSRA)
    # parser.add_argument("--mv_y_q_scale", type=float, default=0.9189, help="mv_y_q_scale")
    # parser.add_argument("--y_q_scale", type=float, default=0.5319, help="y_q_scale")

    parser.add_argument("--ME_D_epoch", type=int, default=0, help="training epochs for ME&MC stage")
    parser.add_argument("--ME_R_v_D_epoch", type=int, default=0, help="training epochs for ME&MC stage")
    parser.add_argument("--REC_D_epoch", type=int, default=0, help="warming up residual parts with only mse")
    parser.add_argument("--REC_R_f_D_epoch", type=int, default=0, help="training residual parts with mse and bpp")
    parser.add_argument("--Whole_epoch", type=int, default=6, help="training whole network for one_frame stage")
    parser.add_argument("--Rolling_epoch", type=int, default=9, help="training whole network for 4_frames stage")
    parser.add_argument("--Cascaded_epoch", type=int, default=5, help="STE Quant for optical flow")
    parser.add_argument("--Max_epoch", type=int, default=30, help="max epochs")
    # set network types
    parser.add_argument("--learning_rate_down_epoch", type=int, default=24, help="lower learning rate")

    # for recording
    parser.add_argument("--save_epochs", type=int, default=1, help="frequency for recording state")
    parser.add_argument("--save_dir", type=str, default="./Experiments", help="directory for recording")
    parser.add_argument("--summary_frequency", type=int, default=1000, help="frequency for recording state")
    args = parser.parse_args()
    return args


def init():
    # parse arguments
    args = parse_args()

    # create directory for recording
    experiment_dir = Path(args.save_dir)
    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath("Checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('Log/')
    log_dir.mkdir(exist_ok=True)

    tb_dir = experiment_dir.joinpath('Tensorboard/')
    tb_dir.mkdir(exist_ok=True)

    # initialize tensorboard
    print("Save tensorboard logger to ", str(tb_dir))
    tensorboard = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)

    # initialize logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/Log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('PARAMETER ...')
    logger.info(args)

    return args, logger, checkpoints_dir, tensorboard


def warp(mv: torch.Tensor, ref: torch.Tensor):
    """
    Args:
        mv : motion map with shape (B, 2, H, W)
        ref: reference with shape (B, 3, H, W)
    Returns:
        warped_ref: warped reference with shape (B, 3, H, W)
    """
    _, _, img_height, img_width = ref.shape

    grid_ver, grid_hor = torch.meshgrid([torch.arange(start=0, end=img_height, dtype=torch.float32),
                                         torch.arange(start=0, end=img_width, dtype=torch.float32)])
    grid = torch.stack((grid_hor, grid_ver), dim=0).unsqueeze(dim=0).to(mv.device)  # shape (B, 2, H, W)
    warped_grid = grid + mv
    warped_grid = torch.cat([warped_grid[:, 0: 1, :, :] / (img_width - 1.0) * 2 - 1,
                             warped_grid[:, 1: 2, :, :] / (img_height - 1.0) * 2 - 1], dim=1)

    warped_ref = F.grid_sample(input=ref, grid=warped_grid.permute(0, 2, 3, 1).contiguous(),
                               mode="bilinear", padding_mode="border", align_corners=False)

    return warped_ref


def conv_init(conv: nn.Conv2d, gain: float, val: float):
    torch.nn.init.xavier_normal_(tensor=conv.weight.data, gain=gain)
    torch.nn.init.constant_(tensor=conv.bias.data, val=val)
