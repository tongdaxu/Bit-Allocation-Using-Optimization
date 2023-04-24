import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from pathlib import Path
from model import BinarizeMnist, ScaleMnist, VAE_SingleLayer, VAE_TwoLayer, VAE_TwoLayer_Alt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def main(args):

    model_name = args.model
    checkpoint_dir = Path(model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "VAE_TwoLayer_Alt":
        PostTransform = ScaleMnist()
    else:
        PostTransform = BinarizeMnist()


    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(28, scale=(0.8,1.0)),
        transforms.ToTensor(),
        PostTransform
    ])
    transform_test= transforms.Compose([
        transforms.ToTensor(),
        PostTransform
    ])
    if args.model == "VAE_SingleLayer":
        model = VAE_SingleLayer()
    elif args.model == "VAE_TwoLayer":
        model = VAE_TwoLayer()
    elif args.model == "VAE_TwoLayer_Alt":
        model = VAE_TwoLayer_Alt()
    else:
        raise NotImplementedError
    model.to(device)
    critical_epochs = [1,3,9,27,81,243,729,2187]
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-04)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.71968)
    training_dataset = datasets.MNIST("./MNIST", train=True, download=True,
                                        transform=transform_train)
    test_dataset = datasets.MNIST("./MNIST", train=False, download=True,
                                    transform=transform_test)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=True, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True)
    num_epochs = args.num_epochs
    N = 1 * 28 * 28
    for epoch in range(0, num_epochs + 1):
        ############### training ###############
        model.train()
        average_elbo = average_bpd = 0
        for i, (images, _) in enumerate(training_dataloader, 1):
            images = images.to(device)
            images = torch.round(images)
            elbo = model(images)
            elbo = torch.mean(elbo / N, dim=0)
            loss = -elbo
            bpd = - (elbo / np.log(2))
            average_elbo += -loss.item()
            average_bpd += bpd.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_elbo /= i
        average_bpd /= i
        print("[train] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
              .format(epoch, average_elbo * N, average_bpd))
        ############### testing ###############
        model.eval()
        average_elbo = average_bpd = 0
        for i, (images, _) in enumerate(test_dataloader, 1):
            images = images.to(device)
            with torch.no_grad():
                elbo = model(images)
                elbo = torch.mean(elbo / N, dim=0)
                bpd = - (elbo / np.log(2))
                average_elbo += elbo.item()
                average_bpd += bpd.item()
        average_elbo /= i
        average_bpd /= i
        print("[test] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
            .format(epoch, average_elbo * N, average_bpd))
        ########### critical points ############
        if epoch in critical_epochs:
            scheduler.step()
            model.eval()
            average_elbo = average_bpd = 0
            for i, (images, _) in enumerate(test_dataloader, 1):
                images = images.to(device)
                with torch.no_grad():
                    elbo = model.nll_iwae(images, 500)
                    elbo = torch.mean(elbo / N, dim=0)
                    bpd = - (elbo / np.log(2))
                    average_elbo += elbo.item()
                    average_bpd += bpd.item()
            average_elbo /= i
            average_bpd /= i
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)
            print("[iwtest] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
                .format(epoch, average_elbo * N, average_bpd))
    model.eval()
    average_elbo = average_bpd = 0
    for i, (images, _) in enumerate(test_dataloader, 1):
        images = images.to(device)
        with torch.no_grad():
            elbo = model.nll_iwae(images, 5000)
            elbo = torch.mean(elbo / N, dim=0)
            bpd = - (elbo / np.log(2))
            average_elbo += elbo.item()
            average_bpd += bpd.item()
    average_elbo /= i
    average_bpd /= i
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)
    print("[end test] elbo:{:.3f}, bpd:{:.3f}"
        .format(epoch, average_elbo * N, average_bpd))


if __name__ == "__main__":

    SEED=3470
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--batch-size-test", type=int, default=100, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=3280, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model", type=str, default="VAE_SingleLayer", help="model type.")

    args = parser.parse_args()
    main(args)
