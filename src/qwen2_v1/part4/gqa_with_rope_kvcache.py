# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import os
import sys

# 将项目的上级目录添加到 Python 模块搜索路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# 从其他模块中导入必要的参数和函数
from part1.my_model_params import *
from part3.my_pos_embed import apply_rotary_pos_emb, global_cos_matrix, global_sin_matrix
from part4.my_kvcache import KVCache

# 处理的 token 个数，表示当前句子中包含的 token 数量
seq_length = 20

# 实例化 KVCache，用于存储和管理注意力层的 Key 和 Value 缓存
kv_cache = KVCache()

# 创建 Query, Key, Value 矩阵，形状为 [1, 20, 896]
# 表示一个包含 20 个 token 的句子，且每个 token 的特征维度为 896
query_states = torch.randn(1, seq_length, hidden_size)
key_states = torch.randn(1, seq_length, hidden_size)
value_states = torch.randn(1, seq_length, hidden_size)

# 对 Query 进行线性映射，映射后的特征维度仍然为 896
weight_for_query = torch.randn(896, hidden_size)
# 对 Key 和 Value 进行线性映射，映射后的特征维度为 128
# 这是因为在 GQA 机制中，Key 和 Value 的头的数量减少，因此总特征维度也减少
weight_for_key_value = torch.randn(128, hidden_size)

# 分别对 Query、Key 和 Value 进行线性映射，得到映射后的矩阵
# 注意三者的输出矩阵维度不再相同
query_states = torch.nn.functional.linear(query_states, weight_for_query)  # [1, 20, 896]
key_states = torch.nn.functional.linear(key_states, weight_for_key_value)  # [1, 20, 128]
value_states = torch.nn.functional.linear(value_states, weight_for_key_value)  # [1, 20, 128]

# 对 Query 的特征维度进行拆分，将 896 个特征拆分为 [14, 64] 的特征
query_states = query_states.view(1, seq_length, num_attention_heads, feature_per_head)  # [1, 20, 14, 64]
# 对 Key 和 Value 的特征维度进行拆分，将 128 个特征拆分为 [2, 64] 的特征
key_states = key_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, 20, 2, 64]
value_states = value_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, 20, 2, 64]

# 将“头”这一维度（14 或 2）放在高维，使维度信息描述为：
# 1 个句子有 14（或 2）个头，每个头有 20 个 token 序列，每个 token 有 64 个特征
# Query 和 Key/Value 的区别在于头的个数不同
query_states = query_states.transpose(1, 2)  # [1, 14, 20, 64]
key_states = key_states.transpose(1, 2)  # [1, 2, 20, 64]
value_states = value_states.transpose(1, 2)  # [1, 2, 20, 64]

# --------------------  嵌入位置编码  ----------------------------
# 生成句子中每个 token 的位置 ID，例如有 20 个 token，生成的位置为 [0, 1, 2, ..., 19]
position_id = torch.arange(seq_length).reshape(1, seq_length)

# 从全局旋转矩阵中抽取出对应位置的旋转矩阵
# 由于全局旋转矩阵包含了所有位置的信息，这里只抽取前 20 个位置的 sin/cos 矩阵
cos = global_cos_matrix[:seq_length]
sin = global_sin_matrix[:seq_length]

# 使用 LLaMa 版本的旋转位置编码公式，应用位置编码到 Query 和 Key 上
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_id)
# ------------------- 嵌入位置编码完成 --------------------------

# -------------------- 嵌入 KVCache 优化 -------------------------
# 这里模拟第 0 层的缓存操作
# `kv_cache.update` 返回的是更新后的 Key 和 Value 状态
key_states, value_states = kv_cache.update(key_states, value_states, layer_idx=0)
# -------------------- 嵌入 KVCache 完成 -------------------------

# 在计算 Query 和 Key 的点积之前，由于 Key 的头的数量为 2，而 Query 的头的数量为 14，
# 直接计算会导致维度不匹配。为了体现分组 Query 共享一组 Key/Value 的逻辑（即 GQA 的目的），
# 需要将 Key 的头的数量复制 7 份，使 Query 和 Key 的头的数量一致，
# 这体现了 Query 共享相同 Key 的含义。
# 使用 torch.repeat_interleave 在第 1 维（头的维度）上复制 7 份，以匹配 Query 的头数量
key_states = torch.repeat_interleave(key_states, repeats=groups, dim=1)  # [1, 14, 20, 64]
value_states = torch.repeat_interleave(value_states, repeats=groups, dim=1)  # [1, 14, 20, 64]

# 使用 PyTorch 的 scaled_dot_product_attention 函数计算注意力输出
attention_out = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states)

# 将“头”这一维度再转置回原来的位置，得到 [1, 20, 14, 64]
attention_out = attention_out.transpose(1, 2)  # [1, 20, 14, 64]

# 将多个头的特征拼接回原来的 896 个特征
attention_out = attention_out.reshape(1, seq_length, hidden_size)  # [1, 20, 896]

# 使用线性映射层将拼接后的特征再映射回去
attention_out = torch.nn.functional.linear(attention_out, weight_for_query)  # [1, 20, 896]

# 打印最终输出的形状，验证结果是否正确
print(attention_out.shape)  # 输出应为 [1, 20, 896]
