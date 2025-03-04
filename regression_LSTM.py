import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import plotly.graph_objs as go
import plotly.offline as py
from tqdm import tqdm
import time

# 读取数据
file_path = './data/ping_an_data_with_pandas_ta_features.csv'
data = pd.read_csv(file_path)

# 填充缺失值
data = data.bfill().ffill()

# 列名示例
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'MA5', 'MA10', 'MA50', 'MA200', 
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'STOCHk_14_3_3', 
            'STOCHd_14_3_3', 'ATR', 'VWAP', 'CMO', 'RSI', 'MACD_12_26_9', 'MACDh_12_26_9', 
            'MACDs_12_26_9', 'Volatility']

# 数据集划分
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

class StockDataset(Dataset):
    def __init__(self, data, mean=None, std=None):
        self.data = data
        self.features = data[features].values.astype(np.float32)
        self.targets = data['Close'].values.astype(np.float32)

        if mean is None or std is None:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0)
        else:
            self.mean = mean
            self.std = std

        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return torch.tensor(feature).unsqueeze(0), torch.tensor(target)  # 在这里增加时间步维度

# 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向LSTM的输出是2倍隐藏层大小

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 双向LSTM需要2倍层数的初始化
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 创建日志文件夹
logs_dir = "LSTM_logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 获取下一个训练编号
def get_next_log_index(logs_dir):
    existing_logs = [int(name.split('_')[1]) for name in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, name))]
    if not existing_logs:
        return 0
    return max(existing_logs) + 1

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item() * features.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, r2_score(all_targets, all_predictions), mean_absolute_percentage_error(all_targets, all_predictions)

# 训练和验证过程
def train_and_validate(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    val_r2s = []
    val_mapes = []
    best_val_r2 = float('-inf')
    best_model_path = None

    # 创建新的子文件夹
    log_index = get_next_log_index(logs_dir)
    logs_dir_current = os.path.join(logs_dir, f'logs_{log_index}')
    os.makedirs(logs_dir_current)

    start_time = time.time()  # 记录训练开始时间
    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_r2, val_mape = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        val_mapes.append(val_mape)

        # Save the model if validation R2 is the best we've seen so far
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_path = os.path.join(logs_dir_current, f'best_model_epoch_{epoch+1}_r2_{val_r2:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, Val MAPE: {val_mape:.4f}')

    end_time = time.time()  # 记录训练结束时间
    training_time = end_time - start_time
    print(f'Training time: {training_time:.2f} seconds')

    # Load the best model before returning
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded best model from {best_model_path}')
    return train_losses, val_losses, val_r2s, val_mapes, best_model_path, logs_dir_current, training_time

# 滚动预测函数（带再训练）
def rolling_predict_with_retraining(model, train_data, val_data, test_data, window_size, device, num_epochs=5):
    model.eval()
    predictions = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    updated_train_data = pd.concat([train_data, val_data])

    for i in tqdm(range(len(test_data)), desc="Rolling Prediction"):
        # 获取滚动窗口数据
        window_data = pd.concat([updated_train_data, test_data[:i]], axis=0).tail(window_size)
        window_dataset = StockDataset(window_data, mean=train_dataset.mean, std=train_dataset.std)
        window_loader = DataLoader(window_dataset, batch_size=len(window_dataset), shuffle=False)

        # 进行预测
        with torch.no_grad():
            for features, _ in window_loader:
                features = features.to(device)
                prediction = model(features)
                predictions.append(prediction[-1].item())

        # 将预测值加入训练数据（模拟实际情况）
        new_row = test_data.iloc[i].copy()
        new_row['Close'] = predictions[-1]
        updated_train_data = pd.concat([updated_train_data, new_row.to_frame().T])

        # 重新训练模型
        train_loader = DataLoader(window_dataset, batch_size=batch_size, shuffle=False)
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)

    return predictions

# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
# 加载最佳模型并进行训练和验证
input_size = len(features)
hidden_size = 64
num_layers = 2
model = BiLSTM(input_size, hidden_size, num_layers).to(device)
num_epochs = 100

train_dataset = StockDataset(train_data)
val_dataset = StockDataset(val_data, train_dataset.mean, train_dataset.std)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 计算模型参数量
total_params = count_parameters(model)
print(f'Model parameters: {total_params}')

train_losses, val_losses, val_r2s, val_mapes, best_model_path, logs_dir_current, training_time = train_and_validate(model, train_loader, val_loader, num_epochs, device)

# 在测试集上进行滚动预测
window_size = len(pd.concat([train_data, val_data]))  # 使用训练和验证数据作为初始窗口
model.load_state_dict(torch.load(best_model_path))  # 确保使用验证集上表现最好的模型
print(f'Using model from {best_model_path} for rolling prediction.')
test_predictions = rolling_predict_with_retraining(model, train_data, val_data, test_data, window_size, device)

# 计算测试集 R2 和 MAPE
test_actuals = test_data['Close'].values
test_r2 = r2_score(test_actuals, test_predictions)
test_mape = mean_absolute_percentage_error(test_actuals, test_predictions)

print(f'Test R2: {test_r2:.4f}')
print(f'Test MAPE: {test_mape:.4f}')

# 保存预测数据到CSV
predictions_df = pd.DataFrame({
    'Actual': test_actuals,
    'Predicted': test_predictions
})
predictions_csv_path = os.path.join(logs_dir_current, 'predictions.csv')
predictions_df.to_csv(predictions_csv_path, index=False)
print(f'Saved predictions to {predictions_csv_path}')

# 使用 Plotly 可视化训练和验证过程中的损失、R2 和 MAPE
epochs = list(range(1, num_epochs + 1))

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Loss'))
fig_loss.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines', name='Validation Loss'))
fig_loss.update_layout(title='Train and Validation Loss', xaxis=dict(title='Epoch'), yaxis=dict(title='Loss'))
fig_loss.show()

fig_r2 = go.Figure()
fig_r2.add_trace(go.Scatter(x=epochs, y=val_r2s, mode='lines', name='Validation R2'))
fig_r2.update_layout(title='Validation R2 Over Epochs', xaxis=dict(title='Epoch'), yaxis=dict(title='R2'))
fig_r2.show()

fig_mape = go.Figure()
fig_mape.add_trace(go.Scatter(x=epochs, y=val_mapes, mode='lines', name='Validation MAPE'))
fig_mape.update_layout(title='Validation MAPE Over Epochs', xaxis=dict(title='Epoch'), yaxis=dict(title='MAPE'))
fig_mape.show()

# 使用 Plotly 可视化滚动预测结果与实际结果
test_trace_actual = go.Scatter(x=test_data.index, y=test_actuals, mode='lines', name='Test Actual Close Price')
test_trace_predicted = go.Scatter(x=test_data.index, y=test_predictions, mode='lines', name='Test Predicted Close Price')

layout = go.Layout(title='Rolling Prediction vs Actual Close Price', xaxis=dict(title='Time'), yaxis=dict(title='Close Price'))

fig = go.Figure(data=[test_trace_actual, test_trace_predicted], layout=layout)
fig.show()