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
from part4.my_kvcache import KVCache
from part4.my_gqa import my_own_gqa


def my_mlp(layer_idx, hidden_state: torch.tensor):
    """
    实现前馈神经网络 MLP 的计算，实际就是文章中的 FFN 层

    参数:
    - layer_idx: int，表示当前处理的是第几层
    - hidden_state: torch.Tensor，输入的隐藏状态，形状为 [batch_size, seq_length, hidden_size]

    返回值:
    - down_proj: torch.Tensor，经过 MLP 处理后的输出
    """
    # 线性变换，应用 gate_proj 权重矩阵
    gate_proj = torch.nn.functional.linear(hidden_state, model.model.layers[layer_idx].mlp.gate_proj.weight)
    # 线性变换，应用 up_proj 权重矩阵
    up_proj = torch.nn.functional.linear(hidden_state, model.model.layers[layer_idx].mlp.up_proj.weight)
    # 使用 SiLU 激活函数，并进行逐元素相乘，再通过 down_proj 权重矩阵进行线性变换
    down_proj = torch.nn.functional.linear(
        torch.functional.F.silu(gate_proj) * up_proj, model.model.layers[layer_idx].mlp.down_proj.weight
    )
    return down_proj


# 实现 RMSNorm 运算
def my_rms_norm(states, weights, eps):
    """
    实现 RMSNorm 层的计算，用于标准化输入张量。

    参数:
    - states: torch.Tensor，输入张量
    - weights: torch.Tensor，RMSNorm 层的权重
    - eps: float，防止除零的 epsilon 值

    返回值:
    - torch.Tensor，经过 RMSNorm 标准化后的张量
    """
    # 将输入转换为 float32 类型
    states = states.to(torch.float32)
    # 计算方差，沿最后一个维度进行平均
    variance = states.pow(2).mean(-1, keepdim=True)
    # 计算标准化后的张量
    states = states * torch.rsqrt(variance + eps)
    # 将标准化后的张量乘以权重，并转换回权重的类型
    return weights * states.to(weights.dtype)


# 实现解码器层的前向计算
def my_decoder(states, layer_idx, position_id, past_key_value):
    """
    实现解码器层的前向计算，包括注意力机制和 MLP 层。

    参数:
    - states: torch.Tensor，输入的隐藏状态
    - layer_idx: int，表示当前处理的是第几层
    - position_id: torch.Tensor，表示每个 token 的位置 ID
    - past_key_value: KVCache 对象，用于缓存 Key 和 Value

    返回值:
    - out: torch.Tensor，经过解码器层处理后的输出
    - present_key_value: KVCache 对象，更新后的 KV 缓存
    """
    residual = states

    # 输入层的 rms_norm 运算
    out = my_rms_norm(states, model.model.layers[layer_idx].input_layernorm.weight, rms_norm_eps)
    # 计算自定义的 GQA 注意力层输出，并更新 KV 缓存
    out, present_key_value = my_own_gqa(layer_idx, out, past_key_value, position_id)
    # 残差连接
    out = out + residual

    residual = out
    # 注意力后层的 rms_norm 运算
    out = my_rms_norm(out, model.model.layers[layer_idx].post_attention_layernorm.weight, rms_norm_eps)
    # 计算 mlp 的输出
    out = my_mlp(layer_idx, out)
    # 残差连接
    out = out + residual

    return out, present_key_value


# 实现整个模型的前向计算
def my_module(states, kv_cache, position_id):
    """
    实现整个模型的前向计算，包括所有的解码器层。

    参数:
    - states: torch.Tensor，输入的隐藏状态
    - past_key_value: KVCache 对象，用于缓存 Key 和 Value
    - position_id: torch.Tensor，表示每个 token 的位置 ID

    返回值:
    - states: torch.Tensor，经过所有解码器层处理后的输出
    - kv_cache: KVCache 对象，更新后的 KV 缓存
    """
    # 遍历所有的解码器层，依次进行处理
    for layer_idx in range(num_hidden_layers):
        states, kv_cache = my_decoder(states, layer_idx, position_id, kv_cache)

    # 最后进行 RMSNorm 归一化
    states = my_rms_norm(states, model.model.norm.weight, rms_norm_eps)

    return states, kv_cache


if __name__ == "__main__":
    # 定义一个包含 15 个 token 的输入状态，形状为 [1, 15, 896]
    seq_length = 15
    hidden_states = torch.randn(1, seq_length, hidden_size).to(torch.bfloat16)

    # 实例化 KVCache，用于存储和管理注意力层的 Key 和 Value 缓存
    kv_cache = KVCache()

    # 生成位置 ID，表示每个 token 的位置
    position_id = torch.arange(seq_length).reshape(1, seq_length)

    # 调用 my_module 进行前向计算
    states, past_kv_cache = my_module(hidden_states, kv_cache, position_id)

    # 输出结果的形状，验证计算是否正确
    print(states.shape)
    # 打印 KV 缓存的内容，验证缓存是否更新
    past_kv_cache.print(0)
