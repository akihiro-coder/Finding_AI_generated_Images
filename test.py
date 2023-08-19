import torch
import torch.nn as nn
from torchvision import models
from utils import ImageTransform, make_datapath_list, Dataset, train_model
import torch.optim as optim
from tqdm import tqdm


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

    net.fc = nn.Sequential(
        net.fc,
        nn.Softmax(),
    )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の設定
    params_to_update = []  # to store parameters that will train
    update_param_names = ['fc.0.weight', 'fc.0.bias']

    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)



    x = torch.randn([1, 3, size, size]).to('cuda:0')
    outputs = net(x)
    print(outputs)
    max_value, idx = torch.max(input=outputs, dim=1)
    print(max_value)
    print(idx)

if __name__ == "__main__":
    main()
