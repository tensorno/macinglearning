import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mydataset import BikeRentalDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, seq_len, conv_channels, kernel_size, lstm_hidden_dim, lstm_layers, output_dim, dropout=0.5):
        super(CNN_LSTM, self).__init__()
        
        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_dim, 
                               out_channels=conv_channels, 
                               kernel_size=kernel_size, 
                               padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # LSTM 部分
        self.lstm = nn.LSTM(input_size=conv_channels, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True, 
                            dropout=dropout)
        
        # 全连接层 (FC)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # CNN 部分
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_dim, seq_len) 适配 Conv1d
        x = self.conv1(x)       # (batch_size, conv_channels, seq_len)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_len, conv_channels) 适配 LSTM
        
        # LSTM 部分
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # 仅取最后一个时间步的输出 (batch_size, lstm_hidden_dim)
        
        # 全连接层
        output = self.fc(lstm_out)  # (batch_size, output_dim)
        return output


class HybridTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, dim_val, n_heads, seq_len, output_size, dropout=0.1):
        super(HybridTimeSeriesModel, self).__init__()

        # LSTM 部分
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.lstm_fc = nn.Linear(hidden_size, dim_val)  # 将 LSTM 输出转换为 Transformer 的输入维度

        # Transformer 部分
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=2)

        # 输出层
        self.fc = nn.Linear(seq_len * dim_val, output_size)  # 线性层，用于最终输出预测

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        lstm_out = self.lstm_fc(lstm_out)  # (batch_size, seq_len, dim_val)

        # Transformer
        transformer_input = lstm_out.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, dim_val)
        transformer_out = self.transformer_encoder(transformer_input)  # (seq_len, batch_size, dim_val)
        transformer_out = transformer_out.permute(1, 0, 2).reshape(batch_size, -1)  # (batch_size, seq_len * dim_val)

        # 输出层
        output = self.fc(transformer_out)  # (batch_size, output_size)

        return output
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    model_params = {
        "input_dim": 10,              # 每个时间步的输入特征维度
        "seq_len": 24,               # 输入序列长度
        "conv_channels": 16,         # CNN 的卷积通道数
        "kernel_size": 3,            # CNN 的卷积核大小
        "lstm_hidden_dim": 32,       # LSTM 的隐藏层维度
        "lstm_layers": 2,            # LSTM 堆叠的层数
        "output_dim": 3,             # 模型输出维度 (如单步预测)
        "dropout": 0.3               # Dropout 比例
    }
    model = CNN_LSTM(**model_params).to(device)

    print(f"Training on {device}.")
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
            torch.save(model.state_dict(), f"hybid_model_epoch_{epoch+1}.pth")

    print("Training complete.")

def evaluate():
    model_params = {
        "input_dim": 10,              # 每个时间步的输入特征维度
        "seq_len": 24,               # 输入序列长度
        "conv_channels": 16,         # CNN 的卷积通道数
        "kernel_size": 3,            # CNN 的卷积核大小
        "lstm_hidden_dim": 32,       # LSTM 的隐藏层维度
        "lstm_layers": 2,            # LSTM 堆叠的层数
        "output_dim": 3,             # 模型输出维度 (如单步预测)
        "dropout": 0.3               # Dropout 比例
    }
    model = CNN_LSTM(**model_params).to(device)
    model.load_state_dict(torch.load('hybid_model_epoch_20.pth'))
    model.eval()  # 设置模型为评估模式
    # 加载测试数据集
    test_dataset = BikeRentalDataset('test_data.csv')
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
            predicted = predicted * std + mean  # 反标准化公式
            print(predicted)
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
    prediction_df.to_csv('predictions_with_targets_hybid.csv', index=False)
    # 绘制实际值与预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
    plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
    plt.legend()
    plt.show()

# def train():
#     model_params = {
#         "input_size": 10,        
#         "hidden_size": 128,     
#         "num_lstm_layers": 2,   
#         "dim_val": 128,         
#         "n_heads": 4,           
#         "seq_len": 24,          
#         "output_size": 3,        # 输出的维度
#         "dropout": 0.1         
#     }
#     model = HybridTimeSeriesModel(**model_params).to(device)

#     print(f"Training on {device}.")
#     # 数据集
#     train_dataset = BikeRentalDataset('train_data.csv', seq_length=24)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    
#     # 定义损失函数和优化器
#     loss_fn = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     epochs = 20
#     print("Training started.")
#     for epoch in range(epochs):
#         model.train()  # 设置模型为训练模式
#         running_loss = 0.0
#         for inputs, targets in train_loader:
#             # 送入设备（GPU 或 CPU）
#             inputs, targets = inputs.to(device), targets.to(device)

#             # 梯度清零
#             optimizer.zero_grad()

#             # 前向传播
#             outputs = model(inputs)

#             # 计算损失
#             loss = loss_fn(outputs, targets)

#             # 反向传播
#             loss.backward()

#             # 更新参数
#             optimizer.step()

#             # 累加损失
#             running_loss += loss.item()

#         # 每个 epoch 打印一次损失
#         avg_loss = running_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

#         # 每个 epoch 可以添加模型的保存等操作（例如保存最佳模型等）
#         if (epoch + 1) % 5 == 0:  # 例如每5个 epoch 保存一次
#             torch.save(model.state_dict(), f"hybid_model_epoch_{epoch+1}.pth")

#     print("Training complete.")

# def evaluate():
#     model_params = {
#         "input_size": 10,        
#         "hidden_size": 128,     
#         "num_lstm_layers": 2,   
#         "dim_val": 128,         
#         "n_heads": 4,           
#         "seq_len": 24,          
#         "output_size": 3,        # 输出的维度
#         "dropout": 0.1         
#     }
#     model = HybridTimeSeriesModel(**model_params).to(device)
#     model.load_state_dict(torch.load('hybid_model_epoch_20.pth'))
#     model.eval()  # 设置模型为评估模式
#     # 加载测试数据集
#     test_dataset = BikeRentalDataset('test_data.csv')
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     # 读取缩放特征
#     filename = 'scaled_features.json'
#     scaled_features = json.load(open(filename, 'r'))
#     std = scaled_features['cnt'][1]
#     mean = scaled_features['cnt'][0]
    
#     predictions = []
#     targets = []

#     # 对测试集进行预测
#     with torch.no_grad():  # 禁用梯度计算，节省内存
#         for features, target in test_loader:
#             # 特征输入到模型
#             features = features.to(torch.float32).to(device)  # 将特征数据转换为float32类型
            
#             # 获取模型预测值
#             predicted = model(features)  # 得到原始的预测值
            
#             # 对预测结果进行反标准化，恢复到原始的尺度
#             predicted = predicted * std + mean  # 反标准化公式
#             print(predicted)
#             target = target * std + mean  # 反标准化公式
#             targets.append(target[0].cpu().numpy())  # 保存目标值
#             predictions.append(predicted[0].cpu().numpy())  # 保存预测值
            
#     # 将预测结果和目标值转换为数组
#     predictions = np.concatenate(predictions, axis=0)
#     targets = np.concatenate(targets, axis=0)

#     # 计算均方误差（MSE）
#     mse = mean_squared_error(targets, predictions)

#     # 计算平均绝对误差（MAE）
#     mae = mean_absolute_error(targets, predictions)

#     # 计算预测值的标准差（STD）
#     std = np.std(predictions)

#       # 打印结果
#     print(f"均方误差 (MSE): {mse:.2f}")
#     print(f"绝对误差 (MAE): {mae:.2f}")
#     print(f"预测值的标准差 (STD): {std:.2f}")

#     # 将预测结果保存到 CSV 文件
#     prediction_df = pd.DataFrame({
#         'predictions': predictions.flatten(),  # 展平以确保是 1D 数组
#         'targets': targets.flatten()  # 展平目标值
#     })
#     # 将 DataFrame 保存到 CSV 文件
#     prediction_df.to_csv('predictions_with_targets_hybid.csv', index=False)
#     # 绘制实际值与预测值的对比图
#     plt.figure(figsize=(12, 6))
#     plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
#     plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    # train()
    evaluate()