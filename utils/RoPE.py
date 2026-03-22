import torch

def precompute_freqs_cis(d_model: int, end: int = int(32 * 1024), omiga: float = 1e6):
    freqs = 1.0 / (omiga ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
    t = torch.arange(end, device=freqs.device)# end是最长预计算freqs的长度，可任意扩增
    freqs = torch.outer(t, freqs).float()# 外积×：[end x 1, 1 x d_model//2)-->end x d_model//2 , 得到 "每个位置 × 每个频率" 的角度 θ
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)# end x d_model//2--> end x d_model
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)# end x d_model//2--> end x d_model
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

d_model=4
q=torch.tensor([1,2,3,4])
k=torch.tensor([5,6,7,8])

freqs_cos, freqs_sin = precompute_freqs_cis(d_model)
q_embed, k_embed=apply_rotary_pos_emb(q,k,freqs_cos, freqs_sin)
print(q_embed.shape, k_embed.shape)# torch.Size([32768, 1, 4]) torch.Size([32768, 1, 4])   1是维度扩展得到的，4是d_model，32768是当前设置的最长序列长度