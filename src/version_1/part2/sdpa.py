# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import math
import os
import sys

# 添加项目上一层目录到 Python 模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# 从 part1 模块中导入模型的配置参数，如 hidden_size
from part1.my_model_params import *

# 定义输入句子的长度，这里假设句子包含 20 个 token
seq_length = 20

# 在 qwen2 模型中，hidden_size 是 896，表示每个 token 的特征维度
# 创建 Query (Q), Key (K), Value (V) 矩阵，假设 batch_size 为 1
# 形状为 [1, seq_length, hidden_size]，即 [1, 20, 896]
query_states = torch.randn(1, seq_length, hidden_size)
key_states = torch.randn(1, seq_length, hidden_size)
value_states = torch.randn(1, seq_length, hidden_size)

# 对 Key 矩阵进行转置，交换 seq_length 和 hidden_size 两个维度
# 转置后的 Key 矩阵代表论文中的 K^T（Key 的转置）
# 转置后 Key 的形状变为 [1, hidden_size, seq_length] 即 [1, 896, 20]
key_states = key_states.transpose(1, 2)

# 计算 Query 和 Key^T 的点积，得到注意力分数矩阵
# 矩阵乘法后，得到的矩阵形状为 [1, seq_length, seq_length] 即 [1, 20, 20]
q_k_mul = torch.matmul(query_states, key_states)

# 对计算出的注意力分数矩阵进行缩放
# 缩放因子是 sqrt(hidden_size)，用于平衡点积值的大小，避免梯度消失或爆炸
scaling_factor = math.sqrt(hidden_size)
# 对 Q 和 K^T 的点积结果除以缩放因子
scaled_q_k_mul = q_k_mul / scaling_factor

# 对缩放后的分数矩阵应用 softmax 函数
# softmax 将分数转换为概率分布，以便用于加权求和
# `dim=-1` 表示在最后一个维度上进行 softmax 计算，即对每个 token 的注意力分数进行归一化
softmax_out = torch.nn.functional.softmax(scaled_q_k_mul, dim=-1)

# 使用 softmax 的输出作为权重，对 Value 矩阵进行加权求和，得到最终的注意力输出
# 结果矩阵的形状为 [1, seq_length, hidden_size] 即 [1, 20, 896]
attention_out = torch.matmul(softmax_out, value_states)

# 打印最终的注意力输出的形状，验证结果是否正确
print(attention_out.shape)  # 输出应为 [1, 20, 896]
