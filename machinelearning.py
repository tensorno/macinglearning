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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        # self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred1 = self.fc1(output)[:, -1, :]
        # pred1, pred2, pred3 = self.fc1(output), self.fc2(output), self.fc3(output)
        # pred1, pred2, pred3 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        # pred = torch.stack([pred1, pred2, pred3], dim=0)
        # print(pred1.shape)

        return pred1



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
        print(out.shape)
        return out


def lstm_train():
    print(f"Training on {device}.")

    # 定义模型
    # model = LSTMModel(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size)
    model = LSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1, batch_size=32, n_outputs=1)
    # 数据集
    train_dataset = BikeRentalDataset('train_data.csv', seq_length=96, output_time=240)
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
            outputs = model(inputs)

            # 计算损失
            total_loss = loss_fn(outputs, targets[:,-1,:])
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
            torch.save(model.state_dict(), f"result/lstm_model_epoch_{epoch+1}.pth")

    print("Training complete.")
        

def predict_lstm():
    # 加载训练好的模型
    model = LSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1, batch_size=32, n_outputs=1)
    model.load_state_dict(torch.load('result/lstm_model_epoch_20.pth'))
    model.eval()  # 设置模型为评估模式
    
    # 加载测试数据集
    test_dataset = BikeRentalDataset(csv_file='test_data.csv', seq_length=96, output_time=240, json_file='scaled_features.json')  # 根据需要调整seq_length

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
    lstm_train()
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