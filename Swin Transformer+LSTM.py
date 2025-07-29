import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import timm

class BatteryDataset(Dataset):
    def __init__(self, image_dir, cycle_numbers, soh_values, transform=None):
        self.image_dir = image_dir
        self.cycle_numbers = cycle_numbers
        self.soh_values = soh_values
        self.transform = transform

    def __len__(self):
        return len(self.soh_values)

    def __getitem__(self, idx):
        cycle_number = int(self.cycle_numbers[idx])
        img_name = f"cycle{cycle_number}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        soh = torch.tensor(self.soh_values[idx], dtype=torch.float32)
        return image, soh

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# 加载 CSV 文件
df = pd.read_csv("file")
df = df.sort_values(by=df.columns[0])  # 保证顺序一致
cycle_numbers = df.iloc[:, 0].astype(int).tolist()
soh_values = df.iloc[:, 1].astype(float).tolist()

# 划分训练和测试集（保持顺序，不打乱）
split_idx = int(0.8 * len(soh_values))
train_cycle_numbers = cycle_numbers[:split_idx]
test_cycle_numbers = cycle_numbers[split_idx:]
train_soh = soh_values[:split_idx]
test_soh = soh_values[split_idx:]

# 路径设置
image_dir = "image"

# 构建数据集和 DataLoader
train_dataset = BatteryDataset(image_dir, train_cycle_numbers, train_soh, transform)
test_dataset = BatteryDataset(image_dir, test_cycle_numbers, test_soh, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型定义
class SwinLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2):
        super(SwinLSTMModel, self).__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.swin(x)            # shape: (B, 768)
        features = features.unsqueeze(1)   # shape: (B, 1, 768)
        lstm_out, _ = self.lstm(features)  # shape: (B, 1, hidden_dim)
        output = self.fc(lstm_out[:, -1, :])
        return output.squeeze(1)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinLSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练函数
def train(model, loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.6f}")

# 预测函数
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().numpy()
            preds.extend(outputs)
            labels.extend(labs.numpy())
    return np.array(preds), np.array(labels)

# 开始训练
train(model, train_loader, criterion, optimizer, epochs=20)

# 测试集评估
preds, labels = predict(model, test_loader)
mae = np.mean(np.abs(preds - labels))
rmse = np.sqrt(np.mean((preds - labels) ** 2))
mape = np.mean(np.abs((preds - labels) / labels)) * 100
print(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# 保存预测结果
result_df = pd.DataFrame({
    "cycle_number": test_cycle_numbers,
    "true_soh": labels,
    "predicted_soh": preds
})
result_df.to_csv("soh_prediction_results.csv", index=False)
print("📁 已保存预测结果：soh_prediction_results.csv")
 
