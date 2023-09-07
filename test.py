import glob
import os
import cv2 as cv
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torchvision.io import read_image
from utils import ImageTransform, make_datapath_list, Dataset, train_model, prepare_input
import torch.optim as optim
from tqdm import tqdm

debug = False


def main():

    # オリジナル画像と生成画像へのファイルパスのリストを作成
    train_file_list, valid_file_list = make_datapath_list('./data/train_1')

    # datasetを作成
    size = 512  # discussion pageより引用　
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = Dataset(train_file_list, transform=ImageTransform(size, mean, std), phase="train")
    valid_dataset = Dataset(valid_file_list, transform=ImageTransform(size, mean, std), phase="val")

    # dataloaderを作成
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {
        'train': train_dataloader, 'val': val_dataloader
    }

    # ネットワークモデルを作成
    # resnet34モデルのインスタンスを生成
    use_pretrained = True  # 学習済みのパラメータを使用
    net = models.resnet34(pretrained=use_pretrained)
    # resnet34の最後の出力層の出力ユニットを0,1の2つに付け加える
    net.fc = nn.Linear(in_features=512, out_features=2)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の設定
    params_to_update = []  # to store parameters that will train
    update_param_names = ['fc.weight', 'fc.bias']
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    # train
    num_epochs = 15
    if not debug:
        train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

    # test image
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_file_list = []
    data_dir = './data/evaluation/'
    testdata_path_list = glob.glob(os.path.join(data_dir + '/*.png'))
    net.eval()
    inference_result = []
    data_file_name = []
    pred_fun = nn.Softmax(dim=1)
    for data_path in tqdm(testdata_path_list):
        # preprocess
        image = prepare_input(data_path, size, mean, std)
        image = image.to(device)  # on GPU

        # inference
        output = pred_fun(net(image))

        # postprocess
        output = output.to('cpu').detach().numpy().copy()
        output = output[0][1] # 0: original, 1: generated
        inference_result.append(output)

        # extract data_file_name from data_path
        data_file_name.append(data_path.split('/')[3])

    # data_file_name = (data_path.split('/')[3] for data_path in testdata_path_list)

    result_list = list(zip(data_file_name, inference_result))
    df = pd.DataFrame(result_list)
    df.to_csv('./data/upload.csv', index=False, header=False)


if __name__ == "__main__":
    main()
