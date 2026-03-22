import numpy as np
import matplotlib.pyplot as plt

def rotate_2d(x, y, theta_rad):
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    vec = np.array([x, y])
    return R @ vec

# 原始向量
x0, y0 = 1, 0

# 旋转角度（单位：弧度）
theta_deg = 90
theta_rad = np.deg2rad(theta_deg)

# 旋转后向量
x1, y1 = rotate_2d(x0, y0, theta_rad)

# 可视化
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, x0, y0, angles='xy', scale_units='xy', scale=1, color='blue', label='original')
plt.quiver(0, 0, x1, y1, angles='xy', scale_units='xy', scale=1, color='red', label=f'rotate{theta_deg}° ')

# 坐标轴设置
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.legend()

import os

# 保存图像供文档使用
output_path = os.path.join(os.path.dirname(__file__), '..', 'image', 'rotate_2d.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"可视化结果已保存至: {output_path}")

plt.show()

