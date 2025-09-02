# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import math
import os
import sys

# 将项目的上一层目录添加到 Python 的模块搜索路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# 从 part1.my_model_params 模块中导入配置参数，如 hidden_size 和 num_attention_heads
from part1.my_model_params import *

# 定义输入句子的长度，这里假设句子包含 20 个 token
seq_length = 20

# 创建 Query (Q), Key (K), Value (V) 矩阵
# 每个矩阵的形状为 [1, seq_length, hidden_size]，即 [1, 20, 896]
query_states = torch.randn(1, seq_length, hidden_size)
key_states = torch.randn(1, seq_length, hidden_size)
value_states = torch.randn(1, seq_length, hidden_size)

# 定义一个用于线性映射的权重矩阵，形状为 [896, 896]
# 实际模型中，Q/K/V 的映射矩阵是不同的，但这里为了简化，使用相同的矩阵
project_weight = torch.randn(hidden_size, hidden_size)

# 对 Query、Key 和 Value 进行线性映射，得到形状为 [1, seq_length, hidden_size] 的结果
query_states = torch.nn.functional.linear(query_states, project_weight)  # [1, 20, 896]
key_states = torch.nn.functional.linear(key_states, project_weight)  # [1, 20, 896]
value_states = torch.nn.functional.linear(value_states, project_weight)  # [1, 20, 896]

# 对特征维度按照注意力头的数量（num_attention_heads = 14）进行拆分
# 将 896 个特征拆分为 [14, 64] 的特征，得到形状为 [1, seq_length, num_attention_heads, feature_per_head]
# 即 [1, 20, 14, 64]
query_states = query_states.view(1, seq_length, num_attention_heads, feature_per_head)
key_states = key_states.view(1, seq_length, num_attention_heads, feature_per_head)
value_states = value_states.view(1, seq_length, num_attention_heads, feature_per_head)

# 将 "头" 这一维度放在第二维度，即形状变为 [1, num_attention_heads, seq_length, feature_per_head]
# 即 [1, 14, 20, 64]，表示每个头独立计算
query_states = query_states.transpose(1, 2)
key_states = key_states.transpose(1, 2)
value_states = value_states.transpose(1, 2)

# 对 Key 矩阵进行转置，使其形状变为 [1, num_attention_heads, feature_per_head, seq_length]
# 这样可以与 Query 进行点积运算
key_states = key_states.transpose(2, 3)  # [1, 14, 64, 20]

# 计算 Query 和 Key^T 的点积，得到注意力分数矩阵
# 形状为 [1, num_attention_heads, seq_length, seq_length] 即 [1, 14, 20, 20]
q_k_mul = torch.matmul(query_states, key_states)

# 对点积结果进行缩放，缩放因子为 sqrt(hidden_size)
scaling_factor = math.sqrt(hidden_size)
scaled_q_k_mul = q_k_mul / scaling_factor

# 对缩放后的分数矩阵应用 softmax 函数，得到注意力权重
# softmax 的输出形状仍为 [1, num_attention_heads, seq_length, seq_length] 即 [1, 14, 20, 20]
softmax_out = torch.nn.functional.softmax(scaled_q_k_mul, dim=-1)

# 使用注意力权重对 Value 矩阵进行加权求和，得到注意力输出
# 形状为 [1, num_attention_heads, seq_length, feature_per_head] 即 [1, 14, 20, 64]
attention_out = torch.matmul(softmax_out, value_states)

# 将 "头" 这一维度再转置回原来的位置，使形状变为 [1, seq_length, num_attention_heads, feature_per_head]
attention_out = attention_out.transpose(1, 2)  # [1, 20, 14, 64]

# 将多个头的特征拼接回原来的 896 个特征，即形状变为 [1, seq_length, hidden_size] 即 [1, 20, 896]
attention_out = attention_out.reshape(1, seq_length, hidden_size)

# 使用线性映射层将特征再映射回去，得到最终的输出
attention_out = torch.nn.functional.linear(attention_out, project_weight)  # [1, 20, 896]

# 输出最终的注意力输出的形状，验证结果是否正确
print(attention_out.shape)  # 输出应为 [1, 20, 896]
