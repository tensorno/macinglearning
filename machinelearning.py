import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mydataset import BikeDataset, BikeRentalDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

columns_to_extract = [
    'instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 
    'holiday', 'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt'
]

input_size = 10
hidden_size_1 = 256
hidden_size_2 = 256
hidden_size_3 = 128
hidden_size_4 = 32
output_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, input_features, output_features, num_heads=4, num_layers=2, hidden_dim=128, dropout=0.1):
        """
        初始化 Transformer 模型
        :param input_features: 输入特征的维度
        :param output_features: 输出特征的维度
        :param num_heads: 多头注意力的头数
        :param num_layers: Transformer 层的数量
        :param hidden_dim: FFN 隐藏层的维度
        :param dropout: Dropout 概率
        """
        super().__init__()

        # 输入特征编码器（线性映射）
        self.input_projection = nn.Linear(input_features, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True  # 支持 batch 维度优先
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出特征解码器（线性映射到目标维度）
        self.output_projection = nn.Linear(hidden_dim, output_features)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, input_features)
        :return: 输出张量 (batch_size, output_features)
        """
        # 添加额外的序列维度，适配 Transformer (batch_size, seq_len=1, input_features)
        x = x.unsqueeze(1)

        # 输入特征映射到隐藏维度
        x = self.input_projection(x)

        # Transformer 编码器处理
        x = self.encoder(x)

        # 去掉序列维度，映射到输出特征维度 (batch_size, output_features)
        x = self.output_projection(x.squeeze(1))
        
        return x


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, num_layers=2, dropout=0.4, batch_first=True)  # num_layers=2
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, num_layers=2, dropout=0.3, batch_first=True)  # num_layers=2
        self.lstm3 = nn.LSTM(hidden_size_2, hidden_size_3, num_layers=1, dropout=0.0, batch_first=True)  # num_layers=1
        self.lstm4 = nn.LSTM(hidden_size_3, hidden_size_4, num_layers=1, dropout=0.0, batch_first=True)  # num_layers=1
        self.fc = nn.Linear(hidden_size_4, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

def transformer_train():
    print(f"Training on {device}.")

    model = TransformerModel(input_size, output_size).to(device)

    train_dataset = BikeDataset('train_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print("Training started.")
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            # 送入设备（GPU 或 CPU）
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        # 每个 epoch 打印一次损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 每个 epoch 可以添加模型的保存等操作（例如保存最佳模型等）
        if (epoch + 1) % 5 == 0:  # 例如每5个 epoch 保存一次
            torch.save(model.state_dict(), f"transfomer_model_epoch_{epoch+1}.pth")

    print("Training complete.")


def lstm_train():
    print(f"Training on {device}.")

    # 定义模型
    model = LSTMModel(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size)

    # 数据集
    train_dataset = BikeRentalDataset('train_data.csv', seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print("Training started.")
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            # 送入设备（GPU 或 CPU）
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        # 每个 epoch 打印一次损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 每个 epoch 可以添加模型的保存等操作（例如保存最佳模型等）
        if (epoch + 1) % 5 == 0:  # 例如每5个 epoch 保存一次
            torch.save(model.state_dict(), f"lstm_model_epoch_{epoch+1}.pth")

    print("Training complete.")
        
def predict_transfomer():
    # 加载训练好的模型
    model = TransformerModel(input_size, output_size).to(device)
    model.load_state_dict(torch.load('transfomer_model_epoch_20.pth'))
    model.eval()  # 设置模型为评估模式
    
    # 加载测试数据集
    test_dataset = BikeDataset('test_data.csv')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 读取缩放特征
    filename = 'scaled_features.json'
    scaled_features = json.load(open(filename, 'r'))
    std = scaled_features['cnt'][1]
    mean = scaled_features['cnt'][0]
    
    predictions = []
    targets = []

    # 对测试集进行预测
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for features, target in test_loader:
            # 特征输入到模型
            features = features.to(torch.float32).to(device)  # 将特征数据转换为float32类型
            
            # 获取模型预测值
            predicted = model(features)  # 得到原始的预测值
            
            # 对预测结果进行反标准化，恢复到原始的尺度
            # print(f"predicted:{predicted}")
            # print(f"target:{target}")
            predicted = predicted * std + mean  # 反标准化公式
            target = target * std + mean  # 反标准化公式
            targets.append(target[0].cpu().numpy())  # 保存目标值
            predictions.append(predicted[0].cpu().numpy())  # 保存预测值
            
    # 将预测结果和目标值转换为数组
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)


    # 计算均方误差（MSE）
    mse = mean_squared_error(targets, predictions)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(targets, predictions)

    # 计算预测值的标准差（STD）
    std = np.std(predictions)

      # 打印结果
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"绝对误差 (MAE): {mae:.2f}")
    print(f"预测值的标准差 (STD): {std:.2f}")

    # 将预测结果保存到 CSV 文件
    prediction_df = pd.DataFrame({
        'predictions': predictions.flatten(),  # 展平以确保是 1D 数组
        'targets': targets.flatten()  # 展平目标值
    })
    # 将 DataFrame 保存到 CSV 文件
    prediction_df.to_csv('predictions_with_targets_transfomer.csv', index=False)
    # 绘制实际值与预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
    plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
    plt.legend()
    plt.show()

def predict_lstm():
    # 加载训练好的模型
    model = LSTMModel(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size)
    model.load_state_dict(torch.load('lstm_model_epoch_20.pth'))
    model.eval()  # 设置模型为评估模式
    
    # 加载测试数据集
    test_dataset = BikeRentalDataset('test_data.csv', seq_length=24, json_file='scaled_features.json')  # 根据需要调整seq_length
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 读取缩放特征
    filename = 'scaled_features.json'
    scaled_features = json.load(open(filename, 'r'))
    std = scaled_features['cnt'][1]
    mean = scaled_features['cnt'][0]
    
    predictions = []
    targets = []

    # 对测试集进行预测
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for features, target in test_loader:
            # 特征输入到模型
            features = features.to(torch.float32)  # 将特征数据转换为float32类型
            
            # 获取模型预测值
            predicted = model(features)  # 得到原始的预测值
            
            # 对预测结果进行反标准化，恢复到原始的尺度
            # print(f"predicted:{predicted}")
            # print(f"target:{target}")
            predicted = predicted * std + mean  # 反标准化公式
            target = target * std + mean  # 反标准化公式
            targets.append(target[0].cpu().numpy())  # 保存目标值
            predictions.append(predicted[0].cpu().numpy())  # 保存预测值
            
    # 将预测结果和目标值转换为数组
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    

    # 计算均方误差（MSE）
    mse = mean_squared_error(targets, predictions)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(targets, predictions)

    # 计算预测值的标准差（STD）
    std = np.std(predictions)

    # 打印结果
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"绝对误差 (MAE): {mae:.2f}")
    print(f"预测值的标准差 (STD): {std:.2f}")

    # 将预测结果保存到 CSV 文件
    prediction_df = pd.DataFrame({
        'predictions': predictions.flatten(),  # 展平以确保是 1D 数组
        'targets': targets.flatten()  # 展平目标值
    })
    # 将 DataFrame 保存到 CSV 文件
    prediction_df.to_csv('predictions_with_targets_lstm.csv', index=False)
    print("预测完成，结果保存在 'predictions.csv' 文件中")
    # 绘制实际值与预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
    plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # lstm_train()
    # predict()
    # transformer_train()
    predict_transfomer()
    # train_dataset = BikeDataset(csv_file='train_data.csv')
    # test_dataset = BikeDataset(csv_file='test_data.csv', json_file='scaled_features.json')
    # train_dataset = BikeRentalDataset(csv_file='train_data.csv')
    # test_dataset = BikeRentalDataset(csv_file='test_data.csv', json_file='scaled_features.json')
    # print(train_dataset[0][0].shape)
    # print(test_dataset[0][0].shape)
    # print(train_dataset[0][1].shape)
    # print(f'Feature shape: {train_dataset.features.shape}')
    # print(f'Target shape: {train_dataset.targets.shape}')
    # print(train_dataset[0][0])