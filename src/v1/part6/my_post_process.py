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
from part5.my_decoder_module import my_module

# 定义生成模型的超参数
top_p = 0.8  # Top-p 采样的累积概率阈值
top_k = 20  # Top-k 采样的 k 值，表示只保留概率最高的 k 个 token
temperature = 0.7  # 温度系数，用于调整 logits 的分布，使生成更有随机性
repetition_penalty = 1.1  # 重复惩罚系数，用于抑制生成重复的 token


# 实现语言模型的输出层（lm_head），将最后一个 token 的隐藏状态映射为 logits
def my_lm_head(states):
    """
    将最后一个 token 的隐藏状态映射为 logits。

    参数:
    - states: torch.Tensor，模型的隐藏状态，形状为 [batch_size, seq_length, hidden_size]

    返回值:
    - logits: torch.Tensor，映射后的 logits，形状为 [1, vocab_size]
    """
    # [1, seq_length, hidden_size] -> [hidden_size] (states[-1,-1,:])
    # [hidden_size] * [hidden_size, vocab_size] = [vocab_size]
    logits = torch.nn.functional.linear(states[-1, -1, :], model.lm_head.weight)
    # [1, vocab_size]
    logits = logits[None, :]  # 保持 batch 维度
    return logits


# 实现 Top-k 策略的 logits 处理
def my_topk_process(logits):
    """
    实现 Top-k 采样策略，保留概率最高的 k 个 token，其余 token 的 logits 设置为负无穷。

    参数:
    - logits: torch.Tensor，模型的输出 logits，形状为 [1, vocab_size]

    返回值:
    - logits_processed: 经过处理后的 logits，只保留概率最高的 k 个 token
    """
    filter_value = -float("Inf")
    top_k_temp = min(top_k, logits.size(-1))  # 确保 top_k 值不会超过 logits 的大小

    # 找出所有 logits 中小于 top-k 第 k 大值的元素，并将其替换为负无穷
    indices_to_remove = logits < torch.topk(logits, top_k_temp)[0][..., -1]
    logits_processed = logits.masked_fill(indices_to_remove, filter_value)
    return logits_processed


# 实现 Top-p 策略的 logits 处理
def my_topp_process(logits):
    """
    实现 Top-p 采样策略，保留累积概率小于 top_p 的 token，其余 token 的 logits 设置为负无穷。

    参数:
    - logits: torch.Tensor，模型的输出 logits，形状为 [1, vocab_size]

    返回值:
    - logits_processed: 经过处理后的 logits，只保留累积概率小于 top_p 的 token
    """
    min_tokens_to_keep = 1
    filter_value = -float("Inf")

    # 对 logits 进行排序并计算累积概率
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # 找出累积概率大于 (1 - top_p) 的 token，并将其替换为负无穷
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0  # 保证至少保留一个 token

    # 将排序后的结果映射回原始的 logits
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits_processed = logits.masked_fill(indices_to_remove, filter_value)
    return logits_processed


# 实现温度系数的 logits 处理
def my_temperature_process(logits):
    """
    实现温度系数处理，将 logits 除以温度系数来控制生成的随机性。

    参数:
    - logits: torch.Tensor，模型的输出 logits

    返回值:
    - logits_processed: 经过温度系数处理后的 logits
    """
    logits_processed = logits / temperature  # 将 logits 除以温度系数
    return logits_processed


# 实现重复惩罚策略的 logits 处理
def my_repetition_penalty_process(input_ids, logits):
    """
    实现重复惩罚策略，对已经生成过的 token 施加惩罚，减少重复生成。

    参数:
    - input_ids: 当前输入的 token 序列
    - logits: torch.Tensor，模型的输出 logits

    返回值:
    - logits_processed: 经过重复惩罚处理后的 logits
    """
    score = torch.gather(logits, 1, input_ids)  # 获取 input_ids 对应的 logits 值
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits_processed = logits.scatter(1, input_ids, score)  # 更新 logits
    return logits_processed


# 针对 logits 后处理函数，依次使用重复惩罚、温度调节、Top-k 和 Top-p 策略
def my_post_process(input_ids, logits):
    """
    对 logits 进行处理，依次应用重复惩罚、温度、Top-k 和 Top-p 策略。

    参数:
    - input_ids: 当前输入的 token 序列
    - logits: torch.Tensor，模型的输出 logits

    返回值:
    - logits_processed: 经过多种策略处理后的 logits
    """
    logits_processed = my_repetition_penalty_process(input_ids, logits)
    logits_processed = my_temperature_process(logits_processed)
    logits_processed = my_topk_process(logits_processed)
    logits_processed = my_topp_process(logits_processed)
    return logits_processed


# 实现下一 token 的预测
def my_predict_next_token(logits, input_ids):
    """
    根据 logits 预测下一个 token，并应用采样策略。

    参数:
    - logits: torch.Tensor，模型的输出 logits
    - input_ids: 当前输入的 token 序列

    返回值:
    - next_token_id: 预测出的下一个 token 的 ID
    """
    next_token_logits = my_post_process(input_ids, logits.to(torch.float32))  # 处理 logits
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)  # 计算概率分布

    # 根据概率分布进行采样，选出下一个 token 的 ID
    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token_id


if __name__ == "__main__":
    # 定义输入的隐藏状态，形状为 [1, 20, 896]
    seq_length = 20
    hidden_states = torch.randn(1, seq_length, hidden_size).to(torch.bfloat16)
    input_ids = torch.arange(seq_length).reshape(1, seq_length)

    # 实例化 KVCache，用于缓存 Key 和 Value
    kv_cache = KVCache()
    position_id = torch.arange(seq_length).reshape(1, seq_length)

    # 调用 my_module 进行前向计算，得到隐藏状态和 KV 缓存
    states, past_kv_cache = my_module(hidden_states, kv_cache, position_id)

    # 通过 lm_head 获取 logits
    logits = my_lm_head(states)

    # 预测下一个 token 的 ID
    next_token_id = my_predict_next_token(logits, input_ids)

    # 将预测出的 token ID 解码为实际的文本 token
    next_token = tokenizer.decode(next_token_id)

    # 打印结果
    print(f"next_token_id = {next_token_id}")
    print(f"logits shape = {logits.shape}")
    past_kv_cache.print(0)
