import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
# from Models.transunet import transunet

# from Models.Ms_RED.DeepLabV3Plus.network import modeling
# from Models import resunetplusplus_pytorch
# from Models.BCDUNet import BCDUNet
# from Models.Ms_RED.networks import ms_red_models
# from Models.Ms_RED.networks import ca_network
# from Models import unetplusplus
# from Models import unet_family
# from Models import atten_unet
# from Models.pytorch_dcsaunet import DCSAU_Net
# from Models import Doubleunet
# from Models.SANet.model import Model
# from Models.MSNet import miccai_msnet
# from Models.SmaAtUNet import SmaAt_UNet
# from Models import unet14
# from Models.DenseASPP import denseaspp
# from Models import cfpnet
# from Models.M2SNet import miccai_msnet
# from Models.DilatedSegNet import DilatedSegNet
# from Models.META_Unet import META_Unet
from Models import unet9_16
# from Models import unet21
# from Models.WNet import WNet
from Metrics import performance_metrics
from Metrics import losses



def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):


    t = time.time() #记录当前时间
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)


        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)




@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)



def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "ISIC2018":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "ISIC2017":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "ColonDB":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "ETIS":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "PH2":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "TrainDataset":
        img_path = args.root + "image/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "mask/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "Serratedadenoma":
        img_path = args.root + "image/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "label/*"
        target_paths = sorted(glob.glob(depth_path))
    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()


    # model = atten_unet.AttU_Net(in_channels=3,num_classes=1)
    # model = modeling.deeplabv3_resnet50(num_classes=1)
    # model = resunetplusplus_pytorch.build_resunetplusplus()
    # model = res_unet.Resnet34_Unet(in_channel=3, out_channel=1, pretrained=True)
    # model = BCDUNet.BCDUNet(input_dim=3, output_dim=1, num_filter=64, frame_size=(352,352), bidirectional=True)
    # model = ms_red_models.Ms_red_v1(classes=1, channels=3)
    # args = parser.parse_args()
    # model = ca_network.Comprehensive_Atten_Unet(args,in_ch=3, n_classes=2)
    # model = unetplusplus.UnetPlusPlus(num_classes=1)
    # model = unet_family.NestedUNet(in_ch=3, out_ch=1)
    # model = atten_unet.AttU_Net(in_channels=3,num_classes=1)
    # model = DCSAU_Net.Model(img_channels = 3, n_classes = 1)
    # model = Doubleunet.build_doubleunet()
    # model = miccai_msnet.MSNet()
    # model = SmaAt_UNet.SmaAt_UNet(n_channels=3, n_classes=1)

    # model = denseaspp.DenseASPP(nclass=1)
    # model = unet14.UNet(in_channels=3,num_classes=1)
    # model = cfpnet.CFPNet(classes=1)
    # model = miccai_msnet.M2SNet()
    # model = DilatedSegNet.RUPNet()
    # model = META_Unet.META_Unet()
    # model = WNet.WNet(3,1)
    model = unet9_16.UNet(in_channels=3, num_classes=1)
    # model = unet21.UNet(in_channels=3, num_classes=1)
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/MCGFFNet_" + args.dataset + ".pt",
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC", "ISIC2018", "ColonDB","ETIS","PH2", "ISIC2017","TrainDataset","Serratedadenoma"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()



def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
