import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ
        self.bias = nn.Parameter(torch.zeros(dim))   # β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # 计算每个 token 的均值
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # 方差,unbiased=False表示除以d,True表示除以d-1
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 标准化
        return self.weight * x_norm + self.bias


# 实例化测试
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]]])  # shape = (1, 2, 4)
norm = LayerNorm(dim=4)
output = norm(x)
print(output)
print(output.shape)