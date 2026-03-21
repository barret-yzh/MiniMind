import numpy as np
import matplotlib.pyplot as plt
import os

def sinusoidal_position_encoding(seq_len, d_model):
    """
    计算正余弦位置编码（Sinusoidal PE）。
    参数：
        seq_len -- 序列长度
        d_model -- 模型的维度
    返回：
        一个形状为 (seq_len, d_model) 的位置编码矩阵
    """
    # 创建位置编码矩阵
    position = np.arange(seq_len)[:, np.newaxis]  # shape为 (seq_len, 1)
    div_term = np.power(10000, (2 * (np.arange(d_model // 2)) / np.float32(d_model)))  # 频率缩放因子

    # 计算正弦和余弦位置编码
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position / div_term)  # 偶数维度用正弦
    pe[:, 1::2] = np.cos(position / div_term)  # 奇数维度用余弦

    return pe

# 参数设置
seq_len = 512  # 序列长度
d_model = 128  # 模型的维度

# 获取位置编码
pe = sinusoidal_position_encoding(seq_len, d_model)

# 绘图可视化
plt.figure(figsize=(10, 6))
for i in [0, 32, 64]:  # 选择几个维度进行展示
    plt.plot(np.arange(seq_len), pe[:, i], label=f'i={i}')

plt.xlabel("Position (pos)")
plt.ylabel("Position Encoding Value")
plt.title(f"Sinusoidal Position Encoding (Frequency Decrease with dimension index i)")
plt.legend(loc='upper right')
plt.tight_layout()

# 保存图像供文档使用
output_path = os.path.join(os.path.dirname(__file__), '..', 'image', 'position_encoding.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"可视化结果已保存至: {output_path}")
print(f"PE Matrix Shape: {pe.shape}")