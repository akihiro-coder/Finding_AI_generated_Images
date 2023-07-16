import os
import glob

import torch
import torch.utils.data as data
from torchvision import models, transforms
from albumentations import RandomBrightnessContrast
from albumentations import ShiftScaleRotate
from PIL import Image
import cv2 as cv


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
                RandomBrightnessContrast(),
                ShiftScaleRotate(),
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
        img : 
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)




# img = cv.imread('./data/train_1/train_0.png')
img = Image.open('./data/train_1/train_0.png')
resize = 500
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = ImageTransform(resize, mean, std)
img_transformed = transform(img, 'train')





def make_datapath_list(data_dir, split_ratio):
    """
    データのパスを格納したリストを作成
    Parameters
    ----------
    data_dir : str
        データが格納されたディレクトリパス
    split_ratio : float
        データ全体のうちの訓練データの割合

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


class Dataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label
