import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class BikeRentalDataset(Dataset):
    def __init__(self, csv_file, seq_length=96, output_time=240, json_file=None):
        self.seq_length = seq_length
        self.output_time = output_time
        # 读取 CSV 数据
        self.cycling_datas = pd.read_csv(csv_file)

        # dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        # for each in dummy_fields:
        #     dummies = pd.get_dummies(self.cycling_datas[each], prefix=each, drop_first=False)
        #     self.cycling_datas = pd.concat([self.cycling_datas, dummies], axis=1)
        # 删除无关数据字段
        # fields_to_drop = ['instant', 'season', 'weathersit', 'weekday',
        #                 'atemp', 'mnth', 'workingday', 'hr', 'dteday']
        fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday']
        # 删除无关数据以及经过扩维的列axis=0表示删除指定行，axis=1表示删除指定列
        self.cycling_datas = self.cycling_datas.drop(fields_to_drop, axis=1)

        # 缩放特征
        scaled_features = {}
        # 量化特征
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        if json_file:
            scaled_features = json.load(open(json_file, 'r'))
            for each in quant_features:
                mean, std = scaled_features[each]
                self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std
        else:
            for each in quant_features:
                # 计算这几个字段的 均值 和 标准差
                mean, std = self.cycling_datas[each].mean(), self.cycling_datas[each].std()
                scaled_features[each] = [mean, std]
                # 将均值和标准差保存到文件中
                json.dump(scaled_features, open(f'scaled_features.json', 'w'))
                # 数据减去均值除以标准差等于标准分值（Standard Score），这样处理是为了符合标准正态分布
                self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std

        # 分开特征features和目标target
        target_cols = ['cnt', 'casual', 'registered']
        target_col = ['cnt']
        self.nomalize(targets=target_cols)
        self.targets = self.cycling_datas[target_col]
        self.features = self.cycling_datas.drop(target_cols, axis=1)

    def __len__(self):
        """返回数据集的样本数"""
        return len(self.targets) - self.output_time

    def __getitem__(self, idx):
        # 获取从 idx 开始，长度为 seq_length 的序列数据
        # 使用 iloc 获取特定行的数据，并转换为 numpy 数组，然后转为张量
        feature = []
        target = []
        for i in range(self.seq_length):
            feature.append(torch.tensor(self.features.iloc[idx + i].values.astype(float), dtype=torch.float32))

        # 将特征序列转换为一个 2D 张量，形状为 (seq_length, feature_size)
        feature = torch.stack(feature)

        # 获取目标值，假设目标值与序列的最后一个时间步相关
        for i in range(self.seq_length, self.output_time):
            target.append(torch.tensor(self.targets.iloc[idx + i - 1].values.astype(float), dtype=torch.float32))

        target = torch.stack(target)
        return feature, target

    def nomalize(self, targets=None):
        for each in targets:
            mean, std = self.cycling_datas[each].mean(), self.cycling_datas[each].std()
            self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std


class BikeDataset(Dataset):
    def __init__(self, csv_file, json_file=None):
        super().__init__()

        self.cycling_datas = pd.read_csv(csv_file)

        fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday']
        # 删除无关数据以及经过扩维的列axis=0表示删除指定行，axis=1表示删除指定列
        self.cycling_datas = self.cycling_datas.drop(fields_to_drop, axis=1)

        # 缩放特征
        scaled_features = {}
        # 量化特征
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        if json_file:
            scaled_features = json.load(open(json_file, 'r'))
            for each in quant_features:
                mean, std = scaled_features[each]
                self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std
        else:
            for each in quant_features:
                # 计算这几个字段的 均值 和 标准差
                mean, std = self.cycling_datas[each].mean(), self.cycling_datas[each].std()
                scaled_features[each] = [mean, std]
                # 将均值和标准差保存到文件中
                json.dump(scaled_features, open(f'scaled_features.json', 'w'))
                # 数据减去均值除以标准差等于标准分值（Standard Score），这样处理是为了符合标准正态分布
                self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std

        # 分开特征features和目标target
        target_cols = ['cnt', 'casual', 'registered']
        self.nomalize(targets=target_cols)
        self.targets = self.cycling_datas[target_cols]
        self.features = self.cycling_datas.drop(target_cols, axis=1)

    def __len__(self):
        return len(self.cycling_datas)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx].values.astype(float), dtype=torch.float32), torch.tensor(
            self.targets.iloc[idx].values.astype(float), dtype=torch.float32)

    def nomalize(self, targets=None):
        for each in targets:
            mean, std = self.cycling_datas[each].mean(), self.cycling_datas[each].std()
            self.cycling_datas[each] = (self.cycling_datas[each] - mean) / std

if __name__ == '__main__':
    dataset = BikeRentalDataset('train_data.csv', seq_length=96, output_time=240)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)