import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler


file_path = 'file' 
data = pd.read_csv(file_path)

filtered_data = data[data["data"]

# 获取独特的循环编号
cycle_numbers = filtered_data["cycle number"].unique()

# 创建 GASF 和 GADF 的 GramianAngularField 对象
gasf = GramianAngularField(method='summation')  # GASF
gadf = GramianAngularField(method='difference')  # GADF

output_dir = 'GAF'
gasf_dir = os.path.join(output_dir, 'GASF')
gadf_dir = os.path.join(output_dir, 'GADF')
os.makedirs(gasf_dir, exist_ok=True)
os.makedirs(gadf_dir, exist_ok=True)

# 遍历每个循环，生成对应的 GASF 和 GADF 图像
for cycle in cycle_numbers:
    cycle_data = filtered_data[filtered_data["cycle number"] == cycle]

    # 提取 "Ecell/V" 数据
    ecell = cycle_data["Ecell/V"].values

    # 归一化数据到 [-1, 1] 之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ecell_normalized = scaler.fit_transform(ecell.reshape(-1, 1)).flatten()

    # 使用 GASF 方法转换归一化后的数据
    X_gasf = gasf.transform(ecell_normalized.reshape(1, -1))  # GASF 转换
    X_gadf = gadf.transform(ecell_normalized.reshape(1, -1))  # GADF 转换

    # 保存 GASF 图像
    plt.figure(figsize=(6, 6))
    plt.imshow(X_gasf[0], origin='lower', cmap='viridis')
    plt.title(f'GASF for Cycle Number: {int(cycle)}')
    plt.colorbar(label='Magnitude')
    plt.savefig(os.path.join(gasf_dir, f'cycle{int(cycle)}.png'), dpi=300)
    plt.close()

    # 保存 GADF 图像
    plt.figure(figsize=(6, 6))
    plt.imshow(X_gadf[0], origin='lower', cmap='viridis')
    plt.title(f'GADF for Cycle Number: {int(cycle)}')
    plt.colorbar(label='Magnitude')
    plt.savefig(os.path.join(gadf_dir, f'cycle{int(cycle)}.png'), dpi=300)
    plt.close()

print("完成所有循环的 GASF 和 GADF 图像生成！")
