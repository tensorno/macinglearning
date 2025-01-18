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
        print(x.shape)
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

class TransformerModel(nn.Module):
    def __init__(self, input_shape, forecast_horizon, num_heads=8, d_model=64, dropout_rate=0.1):
        """
        Args:
            input_shape (tuple): 输入的形状 (seq_len, input_dim)
            forecast_horizon (int): 预测的时间步数
            num_heads (int): 多头注意力的头数
            d_model (int): 模型的嵌入维度
            dropout_rate (float): Dropout 比例
        """
        super(TransformerModel, self).__init__()
        self.seq_len, self.input_dim = input_shape
        self.forecast_horizon = forecast_horizon

        # 线性层将输入维度映射到模型的 d_model 维度
        self.fc1 = nn.Linear(self.input_dim, d_model)

        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

        # 第二个线性层用于特征提取
        self.fc2 = nn.Linear(d_model, 64)

        # 最终线性层将序列长度和时间步映射到预测时间步数
        self.fc3 = nn.Linear(self.seq_len, self.forecast_horizon)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: 输出张量，形状为 (batch_size, forecast_horizon)
        """
        # Step 1: 线性变换，将输入维度映射到 d_model
        x = self.fc1(x)  # (batch_size, seq_len, d_model)

        # Step 2: 多头注意力
        attn_output, _ = self.multihead_attention(x, x, x)  # (batch_size, seq_len, d_model)
        x = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)  # 残差连接

        # Step 3: 特征提取
        x = F.relu(self.fc2(x))  # (batch_size, seq_len, 64)

        # Step 4: 调整时间步到 forecast_horizon
        x = x.permute(0, 2, 1)  # (batch_size, 64, seq_len)
        x = self.fc3(x)  # (batch_size, 64, forecast_horizon)
        x = x.mean(dim=1)  # 平均池化，输出 (batch_size, forecast_horizon)
        
        x = x[:, -1].unsqueeze(-1)
        return x    

# class TransformerModel(nn.Module):
#     def __init__(self, input_size=10, output_size=1, d_model=64, enc_seq_len=96, output_seq=144):
#         """
#         input_size: 输入特征的维度
#         output_size: 输出特征的维度
#         d_model: Transformer 模型的嵌入维度
#         enc_seq_len: 编码器输入序列长度
#         output_seq: 期望的输出序列长度
#         """
#         super(TransformerModel, self).__init__()
#         self.input_fc = nn.Linear(input_size, d_model)  # 输入特征映射到 d_model
#         self.output_fc = nn.Linear(d_model, output_size)  # 将 d_model 映射到输出特征维度
#         self.pos_emb = PositionalEncoding(d_model)  # 位置编码

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=4,
#             dim_feedforward=4 * d_model,
#             batch_first=True,
#             dropout=0.2
#         )
#         self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)

#         # Transformer Decoder
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=4,
#             dim_feedforward=4 * d_model,
#             batch_first=True,
#             dropout=0.2
#         )
#         self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=3)

#         # 生成输出序列
#         self.output_seq = output_seq
#         self.d_model = d_model

#     def forward(self, x):
#         """
#         x: (batch_size, enc_seq_len, input_size)
#         输出: (batch_size, output_seq, output_size)
#         """
#         batch_size, seq_len, _ = x.size()

#         # 输入嵌入
#         x = self.input_fc(x)  # (batch_size, enc_seq_len, d_model)

#         # 添加位置编码
#         x = self.pos_emb(x)  # (batch_size, enc_seq_len, d_model)

#         # 编码器输出
#         memory = self.encoder(x)  # (batch_size, enc_seq_len, d_model)

#         # 构造解码器输入
#         # 解码器输入可以初始化为全零，或使用先前输出作为输入
#         dec_input = torch.zeros(batch_size, self.output_seq, self.d_model, device=x.device)
#         dec_input = self.pos_emb(dec_input)  # 添加位置编码

#         # 解码器输出
#         decoder_output = self.decoder(dec_input, memory)  # (batch_size, output_seq, d_model)

#         # 输出映射到目标维度
#         output = self.output_fc(decoder_output)  # (batch_size, output_seq, output_size)
        
#         out = output[:, -1, :]
#         return out



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

def train_eval(seq_length=96, output_time=97):
    print(f"Training on {device}.")

    train_dataset = BikeRentalDataset('train_data.csv', seq_length=seq_length, output_time=output_time)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)

    test_dataset = BikeRentalDataset('test_data.csv', seq_length=seq_length, output_time=output_time)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TransformerModel(input_shape=(96, 10), forecast_horizon=output_time-seq_length, num_heads=8, d_model=64, dropout_rate=0.1).to(device)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    maes = []
    mses = []
    for i in range(5):
        model.train()  # 设置模型为训练模式
        best_loss = float('inf')  # 初始化最小损失为正无穷
        best_model_path = f"best_transfomer_session_{i+1}_short.pth"  # 定义保存路径
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                # 送入设备（GPU 或 CPU）
                inputs, targets = inputs.to(device), targets.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = loss_fn(outputs, targets[:,-1,:])

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 累加损失
                running_loss += loss.item()

            # 计算当前 epoch 的平均损失
            avg_loss = running_loss / len(train_loader)
            print(f"Session {i+1}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at {best_model_path} with loss {best_loss:.4f}")
                mae, mse = evaluate(model, best_model_path, test_loader)
        maes.append(mae)
        mses.append(mse)
        model = TransformerModel(input_shape=(96, 10), forecast_horizon=output_time-seq_length, num_heads=8, d_model=64, dropout_rate=0.1).to(device) # 重新初始化模型
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 重新初始化优化器

    mae_std = np.std(maes)
    print(f"MAE: {np.mean(maes):.2f}")
    print(f"MAE std: {mae_std:.2f}")
    mse_std = np.std(mses)
    print(f"MSE: {np.mean(mses):.2f}")
    print(f"MSE std: {mse_std:.2f}")

def evaluate(model, model_path,test_loader):
    # model = TransformerModel(input_size=10, output_size=1, d_model=64, enc_seq_len=96, output_seq=144).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

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
            target = target[:, -1, :]
            # 获取模型预测值
            predicted = model(features)  # 得到原始的预测值
            # 对预测结果进行反标准化，恢复到原始的尺度
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

    return mse, mae
    #   # 打印结果
    # print(f"均方误差 (MSE): {mse:.2f}")
    # print(f"绝对误差 (MAE): {mae:.2f}")
    # print(f"预测值的标准差 (STD): {std:.2f}")

    # # 将预测结果保存到 CSV 文件
    # prediction_df = pd.DataFrame({
    #     'predictions': predictions.flatten(),  # 展平以确保是 1D 数组
    #     'targets': targets.flatten()  # 展平目标值
    # })
    # # 将 DataFrame 保存到 CSV 文件
    # prediction_df.to_csv('predictions_with_targets_transfomer.csv', index=False)
    # # 绘制实际值与预测值的对比图
    # plt.figure(figsize=(12, 6))
    # plt.plot(targets, label='(Actual)', color='blue', alpha=0.6)
    # plt.plot(predictions, label='(Predicted)', color='red', alpha=0.6)
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    train_eval(seq_length=96, output_time=97)