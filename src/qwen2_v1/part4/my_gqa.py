# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import os
import sys

# 添加项目的上级目录到 Python 模块搜索路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# 从其他模块中导入必要的参数和函数
from part1.my_model_params import *
from part3.my_pos_embed import apply_rotary_pos_emb, global_cos_matrix, global_sin_matrix
from part4.my_kvcache import KVCache


def my_own_gqa(layer_idx, hidden_states, kv_cache, position_id):
    """
    实现自定义的分组查询注意力机制 (GQA)，并且应用位置编码和 KV 缓存优化。

    参数:
    - layer_idx: int，表示当前处理的是第几层的注意力层
    - hidden_states: torch.Tensor，输入的隐藏状态，形状为 [1, seq_length, hidden_size]
    - kv_cache: KVCache 对象，用于缓存 Key 和 Value
    - position_id: torch.Tensor，表示每个 token 的位置 ID，形状为 [1, seq_length]

    返回值:
    - attn_out: torch.Tensor，经过 GQA 和位置编码处理后的注意力输出
    - kv_cache: KVCache 对象，更新后的 KV 缓存
    """
    # 获取输入的序列长度, -2 代表倒数第二个维度
    seq_length = hidden_states.size()[-2]

    # 对 Query 进行线性映射，得到映射后的 Query 矩阵，形状为 [1, seq_length, 896]
    query_states = torch.nn.functional.linear(
        hidden_states,
        model.model.layers[layer_idx].self_attn.q_proj.weight,
        model.model.layers[layer_idx].self_attn.q_proj.bias,
    )

    # 对 Key 进行线性映射，得到映射后的 Key 矩阵，形状为 [1, seq_length, 128]
    key_states = torch.nn.functional.linear(
        hidden_states,
        model.model.layers[layer_idx].self_attn.k_proj.weight,
        model.model.layers[layer_idx].self_attn.k_proj.bias,
    )

    # 对 Value 进行线性映射，得到映射后的 Value 矩阵，形状为 [1, seq_length, 128]
    value_states = torch.nn.functional.linear(
        hidden_states,
        model.model.layers[layer_idx].self_attn.v_proj.weight,
        model.model.layers[layer_idx].self_attn.v_proj.bias,
    )

    # 对特征维度进行拆分，将 Query 的 896 个特征拆分为 [14, 64] 的特征
    query_states = query_states.view(1, seq_length, num_attention_heads, feature_per_head)  # [1, seq_length, 14, 64]

    # 将 Key 和 Value 的 128 个特征拆分为 [2, 64] 的特征
    key_states = key_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, seq_length, 2, 64]
    value_states = value_states.view(1, seq_length, num_key_value_heads, feature_per_head)  # [1, seq_length, 2, 64]

    # 交换维度，将 "头" 维度 (14 或 2) 放在第二维，方便后续计算
    query_states = query_states.transpose(1, 2)  # [1, 14, seq_length, 64]
    key_states = key_states.transpose(1, 2)  # [1, 2, seq_length, 64]
    value_states = value_states.transpose(1, 2)  # [1, 2, seq_length, 64]

    # 计算 KV 缓存后的总序列长度，包括当前序列长度和缓存序列长度
    kv_seq_len = seq_length + kv_cache.get_seq_length(layer_idx)

    # --------------------  应用旋转位置编码  ----------------------------
    # 从全局旋转矩阵中提取当前序列长度所需的 cos 和 sin 矩阵
    cos = global_cos_matrix[:kv_seq_len]
    sin = global_sin_matrix[:kv_seq_len]

    # 使用旋转位置编码将位置信息应用到 Query 和 Key 上
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_id)
    # ------------------- 旋转位置编码完成  ------------------------------

    # --------------------  更新 KV 缓存  -------------------------
    # 将当前计算的 Key 和 Value 状态更新到 KV 缓存中，返回更新后的 Key 和 Value
    key_states, value_states = kv_cache.update(key_states, value_states, layer_idx)
    # --------------------  KV 缓存更新完成  -------------------------

    # 在计算 Query 和 Key 的点积之前，需要处理头的数量不匹配的问题：
    # Key 的头数量为 2，而 Query 的头数量为 14，直接计算会导致维度不匹配。
    # 因此，需要将 Key 的头数量复制 7 份，使得 Query 和 Key 的头数量一致，体现分组共享的逻辑（GQA）。
    key_states = torch.repeat_interleave(key_states, repeats=groups, dim=1)  # [1, 14, seq_length, 64]
    value_states = torch.repeat_interleave(value_states, repeats=groups, dim=1)  # [1, 14, seq_length, 64]

    # 计算多头注意力，使用 PyTorch 的 scaled_dot_product_attention 函数
    # is_causal 表示是否为自回归模型，这里根据序列长度来确定
    is_causal = seq_length > 1
    attention_out = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=is_causal
    )

    # 将“头”维度再交换回来，恢复原来的形状 [1, seq_length, 14, 64]
    attention_out = attention_out.transpose(1, 2)  # [1, seq_length, 14, 64]

    # 将多个头的输出拼接回原来的特征维度，恢复为 [1, seq_length, 896]
    attention_out = attention_out.reshape(1, seq_length, hidden_size)

    # 通过线性映射层将拼接后的输出映射回去，最终得到 [1, seq_length, 896] 的输出
    attn_out = torch.nn.functional.linear(attention_out, model.model.layers[layer_idx].self_attn.o_proj.weight)

    return attn_out, kv_cache


if __name__ == "__main__":
    # 定义输入的隐藏状态，形状为 [1, 20, 896]，表示包含 20 个 token 的序列
    seq_length = 20
    hidden_states = torch.randn(1, seq_length, hidden_size).to(torch.bfloat16)

    # 实例化 KVCache，用于存储和管理注意力层的 Key 和 Value 缓存
    kv_cache = KVCache()

    # 生成位置 ID，表示每个 token 在序列中的位置
    position_id = torch.arange(seq_length).reshape(1, seq_length)

    # 测试第一层的自定义 GQA 机制
    layer_idx = 0
    attn_out, past_kv_cache = my_own_gqa(layer_idx, hidden_states, kv_cache, position_id)

    # 输出注意力结果的形状，验证计算是否正确
    print(attn_out.shape)

    # 打印 KV 缓存的信息，验证缓存是否更新
    past_kv_cache.print(0)
