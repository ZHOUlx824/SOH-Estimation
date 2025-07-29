import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

filename = 'file'

df = pd.read_csv(filename)

processed_data = df[‘data’]

def introduce_missing_data(data, missing_ratio=0.1):
    min_val, max_val = np.min(data), np.max(data) 
    missing_index = int(len(data) * (1 - missing_ratio))
    missing_data = data.copy()
    missing_data[missing_index:] = 0  
    return missing_data, data, min_val, max_val 

# 处理数据
missing_data, true_data, min_val, max_val = introduce_missing_data(processed_data)

# 检查数据长度
assert len(missing_data) == 760
assert len(true_data) == 760

# 创建自定义数据集
class BatteryDataset(Dataset):
    def __init__(self, missing_data, true_data):
        self.missing_data = (missing_data - min_val) / (max_val - min_val)  # 归一化到 [0,1]
        self.true_data = (true_data - min_val) / (max_val - min_val)  # 归一化到 [0,1]

    def __len__(self):
        return 1  # 只用 1 组数据

    def __getitem__(self, idx):
        return torch.tensor(self.missing_data, dtype=torch.float32), torch.tensor(self.true_data, dtype=torch.float32)

# 创建 DataLoader
dataset = BatteryDataset(missing_data, true_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义 LSGAN 生成器
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 修改为 Sigmoid 以输出 [0,1]，方便反归一化
        )

    def forward(self, x):
        return self.model(x)

# 定义 LSGAN 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练 LSGAN
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

            # 生成器生成数据
            fake_data = generator(missing_data)

            # 训练判别器
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

            # 训练生成器
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_data)
            loss_G_LSGAN = criterion_lsgan(output_fake, real_label)
            loss_G_reconstruction = criterion_reconstruction(fake_data, true_data)

            loss_G = loss_G_LSGAN + lambda_rec * loss_G_reconstruction
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    return generator

# 运行 LSGAN 训练
input_dim = 760
generator = Generator(input_dim)
discriminator = Discriminator(input_dim)

trained_generator = train_LSGAN(generator, discriminator, dataloader)

# 生成重构数据并计算 R2 和 MAPE，并导出 CSV
def evaluate_and_export(generator, missing_data, true_data, min_val, max_val, filename="LSGAN-FEO-1500-10%.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    missing_data_tensor = torch.tensor((missing_data - min_val) / (max_val - min_val), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_data = generator(missing_data_tensor).cpu().squeeze().numpy()

    # 反归一化
    reconstructed_data = reconstructed_data * (max_val - min_val) + min_val

    # 计算 R2 和 MAPE
    r2 = r2_score(true_data, reconstructed_data)
    mape = np.mean(np.abs((true_data - reconstructed_data) / true_data)) * 100

    print(f"Cycle 345: R2={r2:.4f}, MAPE={mape:.4f}%")

    # 创建 DataFrame
    df = pd.DataFrame({
        "Index": range(len(true_data)),
        "True Data": true_data,
        "Reconstructed Data": reconstructed_data,
        "Missing Data": missing_data,
        "Label": ["True" if x != 0 else "Missing" for x in missing_data]  # 标记缺失数据
    })

    # 保存为 CSV
    df.to_csv(filename, index=False)
    print(f"数据已保存为 {filename}")

    return {"R2": r2, "MAPE": mape}
