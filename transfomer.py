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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, dim_val, n_heads, enc_seq_len, dec_seq_len, out_seq_len, 
                 n_encoder_layers=1, n_decoder_layers=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.dec_seq_len = dec_seq_len

        # 输入嵌入层
        self.input_fc = nn.Linear(input_size, dim_val)

        # 位置编码
        self.pos_encoder = PositionalEncoding(dim_val, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads, 
            dim_feedforward=dim_val * 4,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val, 
            nhead=n_heads, 
            dim_feedforward=dim_val * 4,
            dropout=dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # 输出层
        self.out_fc = nn.Linear(dim_val, out_seq_len)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # Encoder 输入处理
        enc_input = self.input_fc(x)  # (batch_size, seq_len, dim_val)
        enc_input = self.pos_encoder(enc_input.permute(1, 0, 2))  # (seq_len, batch_size, dim_val)

        # 编码器输出
        memory = self.encoder(enc_input)  # (seq_len, batch_size, dim_val)

        # Decoder 输入
        dec_input = x[:, -self.dec_seq_len:, :]  # (batch_size, dec_seq_len, input_size)
        dec_input = self.input_fc(dec_input)  # (batch_size, dec_seq_len, dim_val)
        dec_input = self.pos_encoder(dec_input.permute(1, 0, 2))  # (dec_seq_len, batch_size, dim_val)

        # 解码器输出
        output = self.decoder(dec_input, memory)  # (dec_seq_len, batch_size, dim_val)

        # 使用最后一个时间步的解码器输出预测
        output = self.out_fc(output[-1])  # (batch_size, out_seq_len)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Training on {device}.")
    # 数据集
    train_dataset = BikeRentalDataset('train_data.csv', seq_length=24)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(input_size=10, dim_val=64, n_heads=4, enc_seq_len=24, dec_seq_len=2, out_seq_len=3).to(device)
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

def evaluate():
    model = model = TimeSeriesTransformer(input_size=10, dim_val=64, n_heads=4, enc_seq_len=24, dec_seq_len=2, out_seq_len=3).to(device)
    model.load_state_dict(torch.load('transfomer_model_epoch_20.pth'))
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
    prediction_df.to_csv('predictions_with_targets_transfomer.csv', index=False)
    # 绘制实际值与预测值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
    plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # train()
    evaluate()