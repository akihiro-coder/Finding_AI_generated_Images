import os
import glob

import torch
import torch.utils.data as data
from torchvision import models, transforms
from albumentations import RandomBrightnessContrast
from albumentations import ShiftScaleRotate
from PIL import Image
import csv
import pandas as pd


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # to tensor and scale the values[0, 1]
                transforms.Normalize(mean, std),
                # RandomBrightnessContrast(),
                # ShiftScaleRotate(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        img : PIL image
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


def preprocess_test(img_path, save_path='img_transformed.png', resize=500, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), phase='train'):
    """
    Parameters
    ----------
    img_path : str
    """
    img = Image.open(img_path)
    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, phase)
    img = transforms.functional.to_pil_image(img_transformed)
    img.save(save_path)


def make_datapath_list(data_dir, split_ratio=0.8):
    """
    データのパスを格納したリストを作成
    Parameters
    ----------
    data_dir : str
        データが格納されたディレクトリパス
    split_ratio : float
        データ全体のうちの訓練データの割合 default: 0.8

    Returns
    -------
    train_file_list : list
        学習データファイルへのパスを格納したリスト
    valid_file_list : list
        検証データファイルへのパスを格納したリスト
    """

    train_file_list = []
    valid_file_list = []

    data_path_list = glob.glob(os.path.join(data_dir + '/*.png'))

    num_data = len(data_path_list)
    num_split = int(num_data * split_ratio)

    train_file_list = data_path_list[:num_split]
    valid_file_list = data_path_list[num_split:]

    return train_file_list, valid_file_list


def extract_label_from_csv(csv_path, img_path):
    """
    test_0.png, 1
    test_1.png, 0
    test_2.png, 1
    .......
    のようなcsvファイルからラベルを画像ファイル名から抜き出す関数

    Parameters
    ----------
    csv_path: str
        csvファイルのパス
    img_path: str
        画像ファイルパス　

    Returns
    -------
    label : str
    """
    df = pd.read_csv(csv_path, names=('file', 'label'))
    labels = df['label']
    img_filename = os.path.split(img_path)[1]
    img_file_num = int(img_filename.split("_")[1].split(".")[0])
    label = labels[img_file_num]
    return label


class Dataset(data.Dataset):
    """
    オリジナル画像と生成画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : list
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : str
        'train' or 'test' 学習か訓練かを設定
    """

    def __init__(self, file_list, transform=None, phase='train', csv_path='./data/train_1.csv'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定
        self.csv_path = csv_path

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # h, w, c

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        label = extract_label_from_csv(self.csv_path, img_path)

        return img_transformed, label
