import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# 读取数据
filename = 'file.csv'
df = pd.read_csv(filename)
processed_data = df['data'].values

# 缺失处理函数
def introduce_missing_data(data, missing_ratio=0.1):
    min_val, max_val = np.min(data), np.max(data)
    missing_index = int(len(data) * (1 - missing_ratio))
    missing_data = data.copy()
    missing_data[missing_index:] = 0
    return missing_data, data, min_val, max_val

# 缺失数据准备
missing_data, true_data, min_val, max_val = introduce_missing_data(processed_data)
assert len(missing_data) == 760
assert len(true_data) == 760

# 自定义数据集类
class BatteryDataset(Dataset):
    def __init__(self, missing_data, true_data):
        # 归一化至 [-1, 1]
        self.missing_data = 2 * (missing_data - min_val) / (max_val - min_val) - 1
        self.true_data = 2 * (true_data - min_val) / (max_val - min_val) - 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(self.missing_data, dtype=torch.float32), torch.tensor(self.true_data, dtype=torch.float32)

dataloader = DataLoader(BatteryDataset(missing_data, true_data), batch_size=1, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, output_dim=760):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        # 输入为 760 × 1 → 输出为 760
        x = x.view(-1, 1)  # 每个点单独输入
        x = self.model(x)
        return x.view(1, -1)  # 转换回 1 × 760 序列

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim=760):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# LSGAN 训练函数
def train_LSGAN(generator, discriminator, dataloader, epochs=100, lr=0.0002, lambda_rec=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.1, 0.999))

    criterion_lsgan = nn.MSELoss()
    criterion_reconstruction = nn.L1Loss()

    for epoch in range(epochs):
        for missing_data, true_data in dataloader:
            missing_data, true_data = missing_data.to(device), true_data.to(device)

            fake_data = generator(missing_data)

            # 判别器训练
            optimizer_D.zero_grad()
            real_label = torch.ones(true_data.size(0), 1).to(device)
            fake_label = torch.zeros(true_data.size(0), 1).to(device)

            output_real = discriminator(true_data)
            output_fake = discriminator(fake_data.detach())

            loss_D_real = criterion_lsgan(output_real, real_label)
            loss_D_fake = criterion_lsgan(output_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # 生成器训练
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_data)
            loss_G_LSGAN = criterion_lsgan(output_fake, real_label)
            loss_G_reconstruction = criterion_reconstruction(fake_data, true_data)
            loss_G = loss_G_LSGAN + lambda_rec * loss_G_reconstruction
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch + 1}/100], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    return generator

# 模型初始化并训练
generator = Generator(output_dim=760)
discriminator = Discriminator(input_dim=760)
trained_generator = train_LSGAN(generator, discriminator, dataloader)

# 评估与导出函数
def evaluate_and_export(generator, missing_data, true_data, min_val, max_val, filename="LSGAN-output.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    missing_data_tensor = torch.tensor(2 * (missing_data - min_val) / (max_val - min_val) - 1, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_data = generator(missing_data_tensor).cpu().squeeze().numpy()

    # 反归一化
    reconstructed_data = (reconstructed_data + 1) / 2 * (max_val - min_val) + min_val

    r2 = r2_score(true_data, reconstructed_data)
    mape = np.mean(np.abs((true_data - reconstructed_data) / true_data)) * 100

    df = pd.DataFrame({
        "Index": range(len(true_data)),
        "True Data": true_data,
        "Reconstructed Data": reconstructed_data,
        "Missing Data": missing_data,
        "Label": ["True" if x != 0 else "Missing" for x in missing_data]
    })
    df.to_csv(filename, index=False)
    print(f"保存成功: {filename}")
    print(f"R²: {r2:.4f}, MAPE: {mape:.2f}%")

    return {"R2": r2, "MAPE": mape}
