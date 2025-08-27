# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import math
import os
import sys

# 将项目的上一层目录添加到 Python 模块搜索路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# 从 part1.my_model_params 模块中导入模型的相关参数，如 hidden_size 和 num_attention_heads
from part1.my_model_params import *

# 定义输入句子的长度，这里假设句子包含 20 个 token
seq_length = 20

# 创建 Query (Q), Key (K), Value (V) 矩阵
# 每个矩阵的形状为 [1, seq_length, hidden_size]，即 [1, 20, 896]
query_states = torch.randn(1, seq_length, hidden_size)
key_states = torch.randn(1, seq_length, hidden_size)
value_states = torch.randn(1, seq_length, hidden_size)

# 定义对 Query 进行线性映射的矩阵，映射后的特征维度仍然为 896
weight_for_query = torch.randn(896, hidden_size)

# 定义对 Key 和 Value 进行线性映射的矩阵，映射后的特征维度为 128
# 这是因为在 GQA（分组查询注意力）机制中，Key 和 Value 的头的数量减少，因此总特征维度也相应减少。
# 在 qwen2 模型中，Query 的 14 个头被分成了 7 组，因此 Key 和 Value 的特征维度为 896 / 7 = 128。
weight_for_key_value = torch.randn(128, hidden_size)

# 对 Query、Key、Value 进行线性映射，得到映射后的矩阵，注意三者的输出矩阵维度不再相同
query_states = torch.nn.functional.linear(query_states, weight_for_query)  # [1, 20, 896]
key_states = torch.nn.functional.linear(key_states, weight_for_key_value)  # [1, 20, 128]
value_states = torch.nn.functional.linear(value_states, weight_for_key_value)  # [1, 20, 128]

# 对特征维度进行拆分，将 Query 的 896 个特征拆分为 [14, 64] 的特征
query_states = query_states.view(1, seq_length, num_attention_heads, feature_per_head)  # [1, 20, 14, 64]

# 对 Key 和 Value 的 128 个特征拆分为 [2, 64] 的特征
key_states = key_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, 20, 2, 64]
value_states = value_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, 20, 2, 64]

# 将头（Head）的维度放在第二维，以便每个头独立进行计算
# Query 的维度变为 [1, 14, 20, 64]
# Key 和 Value 的维度变为 [1, 2, 20, 64]
query_states = query_states.transpose(1, 2)  # [1, 14, 20, 64]
key_states = key_states.transpose(1, 2)  # [1, 2, 20, 64]
value_states = value_states.transpose(1, 2)  # [1, 2, 20, 64]

# 对 Key 矩阵进行转置，以便与 Query 进行点积运算
key_states = key_states.transpose(2, 3)  # [1, 2, 64, 20]

# 由于 Key 的头的数量为 2，而 Query 的头的数量为 14，两者维度不匹配，不能直接进行点积计算。
# 为了实现 GQA（分组查询注意力）的功能，需要将 Key/Value 的头的数量复制 7 份，使其与 Query 的头数量一致。
# 这体现了 Query 共享同一组 Key/Value 的概念。
# 使用 torch.repeat_interleave 函数在头这一维度上重复 groups（7）次
key_states = torch.repeat_interleave(key_states, repeats=groups, dim=1)  # [1, 14, 64, 20]
value_states = torch.repeat_interleave(value_states, repeats=groups, dim=1)  # [1, 14, 20, 64]

# 计算 Query 和 Key^T 的点积，得到注意力分数矩阵
# 矩阵的形状为 [1, 14, 20, 20]
q_k_mul = torch.matmul(query_states, key_states)

# 对点积结果进行缩放，缩放因子为 sqrt(hidden_size)
scaling_factor = math.sqrt(hidden_size)
scaled_q_k_mul = q_k_mul / scaling_factor

# 对缩放后的分数矩阵应用 softmax 函数，得到注意力权重
softmax_out = torch.nn.functional.softmax(scaled_q_k_mul, dim=-1)  # [1, 14, 20, 20]

# 使用注意力权重对 Value 矩阵进行加权求和，得到注意力输出
attention_out = torch.matmul(softmax_out, value_states)  # [1, 14, 20, 64]

# 将 "头" 这一维度再转置回原来的位置，使维度变为 [1, 20, 14, 64]
attention_out = attention_out.transpose(1, 2)  # [1, 20, 14, 64]

# 将多个头的输出拼接回原来的 896 个特征
attention_out = attention_out.reshape(1, seq_length, hidden_size)  # [1, 20, 896]

# 使用线性映射将拼接后的输出再次映射回去，得到最终的注意力输出
attention_out = torch.nn.functional.linear(attention_out, weight_for_query)  # [1, 20, 896]

# 打印最终的注意力输出的形状，验证结果是否正确
print(attention_out.shape)  # 输出应为 [1, 20, 896]
