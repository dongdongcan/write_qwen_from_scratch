# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import os
import sys

# 将项目的上一层目录添加到 Python 模块搜索路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# 从 part1/my_model_params.py 中导入模型有关的参数
from part1.my_model_params import *

import torch


# 生成 sin/cos 旋转矩阵
def generate_rope_matrix(hidden_size, max_position_embeddings):
    """
    生成用于旋转位置编码的 sin/cos 旋转矩阵。

    参数:
    hidden_size: 词嵌入向量的特征维度（即每个嵌入向量的长度）
    max_position_embeddings: 允许的最大位置嵌入数量，即模型可以表示的最大序列长度

    返回值:
    sin_val: sin 旋转矩阵，用于位置编码
    cos_val: cos 旋转矩阵，用于位置编码
    """

    # 生成 [0, 2, 4, ..., hidden_size-2] 的序列，表示嵌入向量的偶数维度
    # 这些偶数维度将用于构造 sin/cos 位置编码
    seq_list = torch.arange(0, hidden_size, 2, dtype=torch.int64).float()

    # 计算 `2i / hidden_size`，其中 i 为偶数维度索引
    # 该操作决定了不同维度上的位置编码频率（频率与维度有关）
    seq_list = seq_list / hidden_size

    # 计算 `10000^(2i/hidden_size)`，这是位置编码公式中的一部分
    # 使用 10000 作为基数，是为了让不同维度有不同的频率
    seq_list = 10000 ** seq_list

    # 计算 `1/(10000^(2i/hidden_size))`，即旋转角度 θ 序列
    # θ 是用于计算 sin/cos 的角度参数，定义了位置编码的频率
    theta = 1.0 / seq_list

    # 生成位置序列 [0, 1, 2, 3, ..., max_position_embeddings - 1]
    # 这个序列代表了每个位置的索引，用于位置编码
    t = torch.arange(max_position_embeddings, dtype=torch.int64).type_as(theta)

    # 计算位置序列与旋转角度序列的外积，得到一个位置-频率矩阵
    # freqs 矩阵的每一行对应一个位置，每一列对应一个频率
    freqs = torch.outer(t, theta)

    # 将 freqs 矩阵沿最后一维度复制，以便可以同时生成 sin 和 cos 编码
    emb = torch.cat((freqs, freqs), dim=-1)

    # 计算 sin 和 cos 旋转位置矩阵
    # 这些矩阵将与输入的嵌入向量结合，用于位置编码
    cos_val = emb.cos().to(torch.bfloat16)
    sin_val = emb.sin().to(torch.bfloat16)

    # 返回 sin 和 cos 位置矩阵给函数调用者
    return sin_val, cos_val


# 该函数用于对 query 向量进行旋转变换
def rotate_half(x):
    """
    将输入向量的后半部分旋转到前半部分，并将前半部分旋转到后半部分。

    参数:
    x: torch.Tensor 输入的向量张量，形状为 [..., dim]，dim 是特征维度。

    返回值:
    torch.Tensor 旋转后的向量张量，形状不变。
    """
    # 将输入向量沿最后一个维度一分为二
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2 :]  # 取后半部分

    # 将后半部分旋转到前面，将前半部分旋转到后面，并沿最后一个维度拼接
    # 具体来说，后半部分的符号取反，实现旋转的效果
    return torch.cat((-x2, x1), dim=-1)


# 该函数应用旋转位置编码（RoPE）到 Query 和 Key 上
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    将旋转位置编码（RoPE）应用到 Query 和 Key 上，以增强模型的位置信息。

    参数:
    q: torch.Tensor 输入的 Query 张量，形状为 [batch_size, num_heads, seq_length, head_dim]
    k: torch.Tensor 输入的 Key 张量，形状同上
    cos: torch.Tensor 预计算好的 cos 位置编码矩阵
    sin: torch.Tensor 预计算好的 sin 位置编码矩阵
    position_ids: torch.Tensor 表示每个 token 在序列中的实际位置，形状为 [batch_size, seq_length]

    返回值:
    q_embed, k_embed: 经过旋转位置编码变换后的 Query 和 Key 张量
    """
    # 根据序列中的实际位置，从预计算的 cos 和 sin 矩阵中提取出相应位置的编码
    # cos 和 sin 的形状会变为 [batch_size, seq_length, head_dim]
    cos = cos[position_ids]
    sin = sin[position_ids]

    # 应用 LLaMa 模型中的旋转位置编码公式，进行位置编码变换
    # 对 Query 向量进行旋转变换，然后结合 cos 和 sin 进行加权，得到最终的 Query 编码
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # 对 Key 向量进行同样的旋转位置编码变换
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 返回经过位置编码后的 Query 和 Key 向量
    return q_embed, k_embed


# 代码运行之初调用一次即可
global_sin_matrix, global_cos_matrix = generate_rope_matrix(feature_per_head, max_position_embeddings)


if __name__ == "__main__":
    # 定义句子长度为 20
    seq_length = 20

    # 定义 query/key 为 [seq_length, feature_per_head]
    query = torch.randn(1, seq_length, feature_per_head)
    key = torch.randn(1, seq_length, feature_per_head)

    # 获得本次推理时query/key的位置id，用于从全部的 sin/cos 变换矩阵中切出来需要的位置的数据
    position_id = torch.arange(seq_length).reshape(1, seq_length)

    # 将本次推理时的 query 应用到旋转位置编码上
    new_query, new_key = apply_rotary_pos_emb(query, key, global_sin_matrix, global_cos_matrix, position_id)

    print(f"query after RoPE: {new_query.shape}")
    print(f"key after RoPE: {new_key.shape}")
