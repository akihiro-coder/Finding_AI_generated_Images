import torch
import torch.nn as nn
from torchvision import models
from utils import ImageTransform, make_datapath_list, Dataset, train_model
import torch.optim as optim
from tqdm import tqdm


def main():

    # MAX_SPLIT_SIZE_MB = 21
    # メモリアロケーターの最大分割サイズを小さめに設定（vramの断片化を防ぐ）
    # torch.cuda.memory._set_allocator_settings(f"max_split_size_mb:{MAX_SPLIT_SIZE_MB}")

    # オリジナル画像と生成画像へのファイルパスのリストを作成
    train_file_list, valid_file_list = make_datapath_list('./data/train_1')

    # datasetを作成
    size = 512  # discussionより引用　
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
    # 学習させるパラメータとそうでないパラメータを分ける
    # 学習させるパラメータを格納するリスト
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    # 学習させる層のパラメータ名を追加
    update_params_names_1 = ["features"]
    update_params_names_2 = ["classifier.0.weight", "classifier.0.bias",
                             "classifier.3.weight", "classifier.3.bias"]
    update_params_names_3 = ["classifier.6.weight", "classifier.6.bias"]
    for name, param in net.named_parameters():
        if update_params_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_params_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_params_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        else:
            param.requires_grad = False

    # 各パラメータに最適化手法を設定する
    optimizer = optim.SGD([
        {"params": params_to_update_1, "lr": 1e-4},
        {"params": params_to_update_2, "lr": 5e-4},
        {"params": params_to_update_3, "lr": 1e-3},
    ], momentum=0.9)

    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)


if __name__ == "__main__":
    main()
