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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        # (batch_size, seq_length, input_size)->(batch_size, seq_length, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        outputs, (hidden, cell) = self.lstm(x)  # outputs: [batch_size, seq_length, hidden_size]
        return hidden, cell  # 返回隐藏状态和细胞状态


# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # 输出output(batch_size 1 output_size) hiddle cell
        # 
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
         # 输出映射层，用于生成目标序列
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.input_fc = nn.Linear(hidden_size, input_size)
    def forward(self, x, hidden, cell):
        # x: [batch_size, 1, input_size] (单步输入)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # 生成最终任务需要的输出
        predictions = self.output_fc(outputs)  # [batch_size, 1, output_size]
        
        # 生成解码器下一时间步的输入
        next_input = self.input_fc(outputs)  # [batch_size, 1, input_size]
        return predictions, next_input, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, bidirectional=False):
        super(Seq2Seq, self).__init__()

        # 动态创建 Encoder
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        # 动态创建 Decoder
        self.decoder = Decoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
        )

        # 线性层将隐藏状态映射到输出维度
        self.fc = nn.Linear(input_size, output_size)

        # 存储其他参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size

    def forward(self, src, target_len):
        """
        Args:
            src: 输入序列，形状为 (batch_size, seq_len, input_size)
            target_len: 解码器需要生成的时间步长度

        Returns:
            outputs: 形状为 (batch_size, target_len, output_size)
        """
        # 编码阶段
        batch_size, _, _ = src.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(src.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(src.device)
        hidden, cell = self.encoder(src)

        # 解码阶段
        outputs = torch.zeros(batch_size, target_len, self.output_size).to(src.device)
        decoder_input = src[:, -1, :]  # 取编码器的最后一步作为解码器的初始输入
        for t in range(target_len):
            decoder_input = decoder_input.unsqueeze(1)  # 调整形状为 (batch_size, 1, input_size)
            decoder_output, next_input,hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            outputs[:, t, :] = decoder_output.squeeze(1)  # 保存预测结果
            
            # 使用当前预测作为下一时间步的输入
            decoder_input = next_input.squeeze(1)

        return outputs


# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#         self.num_directions = 1
#         self.n_outputs = n_outputs
#         self.batch_size = batch_size
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
#         # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
#         self.fc1 = nn.Linear(self.hidden_size, self.output_size)
#         # self.fc2 = nn.Linear(self.hidden_size, self.output_size)
#         # self.fc3 = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input_seq):
#         batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
#         h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
#         c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
#         # print(input_seq.size())
#         # input(batch_size, seq_len, input_size)
#         # output(batch_size, seq_len, num_directions * hidden_size)
#         output, _ = self.lstm(input_seq, (h_0, c_0))
#         pred1 = self.fc1(output)
#         # pred1, pred2, pred3 = self.fc1(output), self.fc2(output), self.fc3(output)
#         # pred1, pred2, pred3 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :]

#         # pred = torch.cat([pred1, pred2], dim=0)
#         # pred = torch.stack([pred1, pred2, pred3], dim=0)
#         # print(pred1.shape)

#         return pred1



# 定义 LSTM 模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_size, hidden_size_1, num_layers=2, dropout=0.4, batch_first=True)  # num_layers=2
#         self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, num_layers=2, dropout=0.3, batch_first=True)  # num_layers=2
#         self.lstm3 = nn.LSTM(hidden_size_2, hidden_size_3, num_layers=1, dropout=0.0, batch_first=True)  # num_layers=1
#         self.lstm4 = nn.LSTM(hidden_size_3, hidden_size_4, num_layers=1, dropout=0.0, batch_first=True)  # num_layers=1
#         self.fc = nn.Linear(hidden_size_4, output_size)

#     def forward(self, x):
#         out, _ = self.lstm1(x)
#         out, _ = self.lstm2(out)
#         out, _ = self.lstm3(out)
#         out, _ = self.lstm4(out)
#         # out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         return out


def lstm_train():
    print(f"Training on {device}.")

    args ={
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
        "batch_size": 32,
    }
    # 定义模型
    model = Seq2Seq(**args).to(device)
    # 数据集
    train_dataset = BikeRentalDataset('train_data.csv', seq_length=96, output_time=97)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    print("Training started.")
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            # 送入设备（GPU 或 CPU）
            inputs, targets = inputs.to(device), targets.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs, target_len=1) 
            # 计算损失
            total_loss = loss_fn(outputs, targets)
            # for k in range(3):
            #     total_loss += loss_fn(outputs[k, :, :], targets[:, :, k])
            # total_loss /= outputs.shape[0]
            # 反向传播
            total_loss.backward()

            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += total_loss.item()

        # 每个 epoch 打印一次损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 每个 epoch 可以添加模型的保存等操作（例如保存最佳模型等）
        if (epoch + 1) % 5 == 0:  # 例如每5个 epoch 保存一次
            torch.save(model.state_dict(), f"result/lstm_model_epoch_{epoch+1}_short.pth")

    print("Training complete.")
        

def predict_lstm():
    # 加载训练好的模型
    args ={
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
        "batch_size": 32,
    }
    # 定义模型
    model = Seq2Seq(**args).to(device)
    model.load_state_dict(torch.load('result/lstm_model_epoch_20_short.pth'))
    model.eval()  # 设置模型为评估模式
    
    # 加载测试数据集
    test_dataset = BikeRentalDataset(csv_file='test_data.csv', seq_length=96, output_time=97, json_file='scaled_features.json')  # 根据需要调整seq_length

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
            predicted = model(features, target_len=1)  # 得到原始的预测值
            predicted = predicted[:, -1, :]  # 取最后一个时间步的预测值
            target = target[:, -1, :]
            # print(predicted)
            # 对预测结果进行反标准化，恢复到原始的尺度
            predicted = predicted * std + mean  # 反标准化公式
            target = target * std + mean  # 反标准化公式
            # break
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
    predict_lstm()
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