import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
# from Models.PraNet import PraNet_Res2Net
# from Models.Ms_RED.DeepLabV3Plus.network import modeling
# from Models import resunetplusplus_pytorch
# from Models.BCDUNet import BCDUNet
# from Models import unet_family
# from Models import atten_unet
# from Models.pytorch_dcsaunet import DCSAU_Net
# from Models.MSNet import miccai_msnet
# from Models.SmaAtUNet import SmaAt_UNet
# from Models.DenseASPP import denseaspp
# from Models import cfpnet
from Models import unet9_16
# from Models.M2SNet import miccai_msnet
# from Models.DilatedSegNet import DilatedSegNet
# from Models.WNet import WNet
from Metrics import performance_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.test_dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "ISIC2018":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "ISIC2017":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "ColonDB":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "ETIS":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "PH2":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "labels/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC-300":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "Serratedadenoma":
        img_path = args.root + "image/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "label/*"
        target_paths = sorted(glob.glob(depth_path))
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1
    )

    _, test_indices, _ = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()
    # model = modeling.deeplabv3_resnet50(num_classes=1)
    # model = PraNet_Res2Net.PraNet()
    # model = FAT_Net(n_classes=1)
    # model = resunetplusplus_pytorch.build_resunetplusplus()
    # model = BCDUNet.BCDUNet(input_dim=3, output_dim=1, num_filter=64, frame_size=(352, 352), bidirectional=True)
    # model = unet_family.NestedUNet(in_ch=3, out_ch=1)
    # model = atten_unet.AttU_Net(in_channels=3,num_classes=1)
    # model = DCSAU_Net.Model(img_channels=3, n_classes=1)
    # model = miccai_msnet.MSNet()
    # model = SmaAt_UNet.SmaAt_UNet(n_channels=3, n_classes=1)
    # model = denseaspp.DenseASPP(nclass=1)
    # model = cfpnet.CFPNet(classes=1)
    # model = unet9_7.UNet(in_channels=3,num_classes=1)
    model = unet9_16.UNet(in_channels= 3,num_classes=1)
    # model = miccai_msnet.M2SNet()
    # model = DilatedSegNet.RUPNet()
    # model = WNet.WNet(3,1)
    state_dict = torch.load(
        "./Trained models/MCGFFNet_{}.pt".format(args.train_dataset)
    )

    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
        os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
    if not os.path.exists(
        "./Predictions/Trained on {}/Tested on {}".format(
            args.train_dataset, args.test_dataset
        )
    ):
        os.makedirs(
            "./Predictions/Trained on {}/Tested on {}".format(
                args.train_dataset, args.test_dataset
            )
        )

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite(
            "./Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
            ),
            predicted_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC", "ISIC2018", "ColonDB","ETIS","PH2", "ISIC2017","TrainDataset","Serratedadenoma"]
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC", "ISIC2018", "ColonDB","ETIS","PH2", "ISIC2017","CVC-300","Serratedadenoma"]
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

