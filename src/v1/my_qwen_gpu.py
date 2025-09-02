# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

# !pip3 install huggingface_hub
# !pip3 install transformers -U

import os, sys
import numpy as np
import random
import transformers
import torch
from torch import nn
import math
from typing import List
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2ForCausalLM, Qwen2Config
from transformers import GenerationConfig

SUPPORT_MODELS = {"Qwen2-0.5B-Instruct", "Qwen2-1.5B-Instruct", "Qwen2-7B-Instruct", "Qwen2-72B-Instruct"}


def get_model_name():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default="Qwen2-0.5B-Instruct")
    args = parser.parse_args()

    if args.model not in SUPPORT_MODELS:
        print(f"model `{args.model}` not supported")
        print(f"Supported models: {', '.join(SUPPORT_MODELS)}")
        sys.exit(1)
    else:
        print(f"Runing {args.model}...")
        return f"Qwen/{args.model}"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ------------------------------- Part1 -------------------------------------
# 获取模型的模型结构、tokenizer 和 config 参数，供其他 Part 的函数使用

# 定义要加载的模型名称，这里是 "Qwen/Qwen2-0.5B-Instruct"
model_name = get_model_name()

# 从预训练模型中加载模型结构
model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

# 从预训练模型中加载与文本生成相关的配置
generation_config = GenerationConfig.from_pretrained(model_name)

# 加载与模型匹配的 tokenizer，用于对输入文本进行分词（tokenization）
tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)

# 加载模型的配置文件 config，这里包含了模型的超参数和架构信息
config = Qwen2Config.from_pretrained(model_name)

# 从配置文件中提取一些关键的模型参数，方便在其他部分的代码中使用

# 提取模型中的注意力头的数量。注意力机制是 Transformer 模型的核心，每个头代表一个独立的注意力计算单元
num_attention_heads = config.num_attention_heads  # 14，模型中有14个注意力头

# 提取 GQA（分组查询注意力）中 key 和 value 的头的数量。GQA 将注意力头分组，以减少计算开销
num_key_value_heads = config.num_key_value_heads  # 2，key/value 的头的数量为 2

# 提取模型中每个 token 的特征维度，即嵌入向量的维度。
hidden_size = config.hidden_size  # 896，特征维度为 896

# 提取模型支持的最大位置编码数，即模型可以处理的最大 token 数（序列长度）
max_position_embeddings = config.max_position_embeddings  # 16384，模型支持的最大 token 个数为 16384

# 计算 GQA 中每个注意力头负责的特征维度，即将总特征维度平分到每个注意力头上
feature_per_head = (int)(hidden_size / num_attention_heads)  # 64，每个头负责 64 维特征

# 计算 GQA 中分组的数量，即将总的注意力头数分为多个组，每个组共享 key/value 头
groups = (int)(num_attention_heads / num_key_value_heads)  # 7，注意力头分为 7 组

# 提取解码器（decoder）模块的层数。Transformer 模型通常由多个堆叠的解码器层组成，每层包含注意力和前馈网络等组件
num_hidden_layers = config.num_hidden_layers  # 24，模型包含 24 层解码器

# 提取 RMS Norm（Root Mean Square Layer Normalization）运算中的 epsilon 参数。该参数用于避免归一化计算中的除零问题
rms_norm_eps = config.rms_norm_eps  # 归一化计算中的 eps 参数


# 定义自己的 chat template 函数，为用户输入的文本增加模板
def my_apply_chat_template(prompt):
    """
    为用户输入的 prompt 添加自定义的聊天模板，以模拟聊天对话框的格式。

    参数:
    prompt: str 用户输入的文本

    返回值:
    str 带有模板的完整文本
    """
    # 使用预训练的 tokenizer 将用户输入的文本编码为 token ID 列表
    prompt_encoding = tokenizer.encode(prompt)

    # 定义聊天模板的 token ID 列表
    # 这些 token ID 代表预设的系统提示、用户提示以及回答等部分
    template = (
        [
            151644,  # 开始 system 提示符的 token ID
            8948,  # "system" 关键字的 token ID
            198,  # 换行符 token ID
            2610,  # "You" 的 token ID
            525,  # "are" 的 token ID
            264,  # "a" 的 token ID
            10950,  # "helpful" 的 token ID
            17847,  # "assistant" 的 token ID
            13,  # 句号 token ID
            151645,  # 结束 system 提示符的 token ID
            198,  # 换行符 token ID
            151644,  # 开始 user 提示符的 token ID
            872,  # "user" 关键字的 token ID
            198,  # 换行符 token ID
        ]
        + prompt_encoding  # 拼接用户输入的编码（用户实际输入的文本 token ID 列表）
        + [
            151645,  # 结束 user 提示符的 token ID
            198,  # 换行符 token ID
            151644,  # 开始 assistant 提示符的 token ID
            77091,  # "assistant" 关键字的 token ID
            198,  # 换行符 token ID
        ]
    )

    # 将完整的 token ID 列表解码回文本，生成带有模板的最终输出
    return tokenizer.decode(template)


# 实现词嵌入的功能：将用户的输入文本 prompt 转换为词嵌入向量
def my_word_embedding_process(prompt):
    """
    将用户输入的文本转换为词嵌入向量。

    参数:
    prompt: str 用户输入的文本

    返回值:
    tuple 包含输入的 ID（input_ids）和对应的词嵌入向量（word_embeddings）
    """
    # 首先，将文本转换为 token ID，得到的结果为一个包含每个 token ID 的列表
    input_ids = tokenizer.encode(prompt)

    # 将 token ID 列表转换为 PyTorch 的张量（tensor），形状为 [seq_length]
    input_ids = torch.tensor(input_ids).to(device)

    # 增加一个 batch 维度，使输入张量的形状变为 [1, seq_length]，以适应模型的输入格式
    input_ids = input_ids[None, :]

    # 第一种方法实现词嵌入计算
    # vocab_size = config.vocab_size
    # hidden_size = config.hidden_size
    # embedding_layer = torch.nn.Embedding(vocab_size, hidden_size)
    # embedding_layer.weight.data.copy_(model.model.embed_tokens.weight)
    # word_embeddings = embedding_layer(input_ids).to(torch.bfloat16)

    # 词嵌入计算的第二种方法
    # 使用已经训练好的模型中的词嵌入权重（model.model.embed_tokens.weight）来进行词嵌入
    # torch.nn.functional.embedding 函数根据输入的 token ID 从嵌入矩阵中查找相应的向量
    word_embeddings = torch.nn.functional.embedding(input_ids, model.model.embed_tokens.weight)

    # 返回输入的 token ID 和词嵌入向量
    return input_ids, word_embeddings


# ------------------------------- Part3 -------------------------------------
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
    seq_list = torch.arange(0, hidden_size, 2, dtype=torch.int64).float().to(device)

    # 计算 `2i / hidden_size`，其中 i 为偶数维度索引
    # 该操作决定了不同维度上的位置编码频率（频率与维度有关）
    seq_list = seq_list / hidden_size

    # 计算 `10000^(2i/dim)`，这是位置编码公式中的一部分
    # 使用 10000 作为基数，是为了让不同维度有不同的频率
    seq_list = 10000 ** seq_list

    # 计算 `1/(10000^(2i/dim))`，即旋转角度 θ 序列
    # θ 是用于计算 sin/cos 的角度参数，定义了位置编码的频率
    theta = 1.0 / seq_list

    # 生成位置序列 [0, 1, 2, 3, ..., max_position_embeddings - 1]
    # 这个序列代表了每个位置的索引，用于位置编码
    t = torch.arange(max_position_embeddings, dtype=torch.int64).type_as(theta).to(device)

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


# ------------------------------- Part4 -------------------------------------
# 定义一个 KVCache 类，用于缓存注意力层的 Key 和 Value 状态
class KVCache:
    def __init__(self) -> None:
        """
        KVCache 类的构造函数。

        该类用于存储和管理多层注意力模型中的 Key 和 Value 缓存。
        每一层的 Key 和 Value 都存储在一个可变长度的列表中，方便在生成过程中逐步追加新计算的状态。
        """
        # 类中的 KCache 和 VCache 分别用于存储 Key 和 Value 的缓存
        # 这两个属性都是可变长度的列表，每一层注意力对应一个缓存列表
        self.KCache: List[torch.tensor] = []
        self.VCache: List[torch.tensor] = []

    def update(self, new_key_states, new_value_states, layer_idx):
        """
        更新 KVCache，将新的 Key 和 Value 状态添加到指定层的缓存中。

        参数:
        new_key_states: 新计算的 Key 状态，形状为 [batch_size, seq_length, hidden_size]
        new_value_states: 新计算的 Value 状态，形状同上
        layer_idx: 表示第几层注意力层（索引从 0 开始）

        返回值:
        更新后的这一层的 Key 和 Value 缓存。
        """
        # 如果缓存的层数少于 layer_idx，说明当前层的 Key 和 Value 是新的层，需要初始化缓存
        if len(self.KCache) <= layer_idx:
            # 在对应层的位置初始化缓存，存储当前的 Key 和 Value 状态
            self.KCache.append(new_key_states)
            self.VCache.append(new_value_states)
        else:
            # 如果当前层已经存在缓存，则将新生成的 Key 和 Value 状态追加到现有缓存的后面
            # 追加操作是在 token 序列长度的维度（dim=-2）上进行的
            self.KCache[layer_idx] = torch.cat([self.KCache[layer_idx], new_key_states], dim=-2)
            self.VCache[layer_idx] = torch.cat([self.VCache[layer_idx], new_value_states], dim=-2)

        # 返回更新后的这一层的 Key 和 Value 缓存
        return self.KCache[layer_idx], self.VCache[layer_idx]

    # 获取某一层中已经缓存的 token 的数量（即序列长度 seq_length）
    def get_seq_length(self, layer_idx) -> int:
        """
        获取指定层中已经缓存的 token 数量（即序列长度）。

        参数:
        layer_idx: 要查询的注意力层索引

        返回值:
        已缓存的 token 数量，如果该层还没有缓存数据，则返回 0。
        """
        # 如果请求的层还没有缓存，则返回 0
        if len(self.KCache) <= layer_idx:
            return 0
        # 返回该层缓存的 token 数量，即缓存的 Key 状态的序列长度
        return self.KCache[layer_idx].shape[-2]

    # 打印指定层缓存的 token 数量
    def print(self, layer_idx):
        """
        打印指定层中已经缓存的 token 数量。

        参数:
        layer_idx: 要查询的注意力层索引
        """
        # 如果缓存为空，输出提示信息
        if len(self.KCache) == 0:
            print("缓存为空")
        else:
            # 打印指定层缓存的 token 数量
            print(f"层 {layer_idx} 缓存的 token 数：", self.KCache[layer_idx].shape[-2])


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


# ------------------------------- Part5 -------------------------------------
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


# 实现 RMSNorm
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
    # 计算 mlp 输出
    out = my_mlp(layer_idx, out)
    # 残差连接
    out = out + residual

    return out, present_key_value


# 实现整个模型的前向计算
def my_module(states, past_key_value, position_id):
    """
    实现整个模型的前向计算，包括所有的解码器层。

    参数:
    - states: torch.Tensor，输入的隐藏状态
    - past_key_value: KVCache 对象，用于缓存 Key 和 Value
    - position_id: torch.Tensor，表示每个 token 的位置 ID

    返回值:
    - states: torch.Tensor，经过所有解码器层处理后的输出
    - past_key_value: KVCache 对象，更新后的 KV 缓存
    """
    # 遍历所有的解码器层，依次进行处理
    for layer_idx in range(num_hidden_layers):
        states, present_key_value = my_decoder(states, layer_idx, position_id, past_key_value)

    # 更新 KV 缓存
    past_key_value = present_key_value
    # 最后进行 RMSNorm 归一化
    states = my_rms_norm(states, model.model.norm.weight, rms_norm_eps)
    return states, past_key_value


# ------------------------------- Part6-------------------------------------

# 定义生成模型的超参数
top_p = generation_config.top_p  # Top-p 采样的累积概率阈值, 0.8
top_k = generation_config.top_k  # Top-k 采样的 k 值，表示只保留概率最高的 k 个 token
temperature = generation_config.temperature  # 温度系数，用于调整 logits 的分布，使生成更有随机性
repetition_penalty = generation_config.repetition_penalty  # 重复惩罚系数，用于抑制生成重复的 token


# 实现语言模型的输出层（lm_head），将最后一个 token 的隐藏状态映射为 logits
def my_lm_head(states):
    """
    将最后一个 token 的隐藏状态映射为 logits。

    参数:
    - states: torch.Tensor，模型的隐藏状态，形状为 [batch_size, seq_length, hidden_size]

    返回值:
    - logits: torch.Tensor，映射后的 logits，形状为 [1, vocab_size]
    """
    logits = torch.nn.functional.linear(states[-1, -1, :], model.lm_head.weight)
    logits = logits[None, :]  # 保持 batch 维度
    return logits


# 实现 Top-k 策略的 logits 处理
def topk_logits_warper(input_ids, logits):
    """
    实现 Top-k 采样策略，保留概率最高的 k 个 token，其余 token 的 logits 设置为负无穷。

    参数:
    - input_ids: 当前输入的 token 序列
    - logits: torch.Tensor，模型的输出 logits，形状为 [1, vocab_size]

    返回值:
    - logits_processed: 经过处理后的 logits，只保留概率最高的 k 个 token
    """
    filter_value = -float("Inf")
    top_k_temp = min(top_k, logits.size(-1))  # 确保 top_k 值不会超过 logits 的大小

    # 找出所有 logits 中小于 top-k 第 k 大值的元素，并将其替换为负无穷
    indices_to_remove = logits < torch.topk(logits, top_k_temp)[0][..., -1, None]
    logits_processed = logits.masked_fill(indices_to_remove, filter_value)
    return logits_processed


# 实现 Top-p 策略的 logits 处理
def topp_logits_warper(input_ids, logits):
    """
    实现 Top-p 采样策略，保留累积概率小于 top_p 的 token，其余 token 的 logits 设置为负无穷。

    参数:
    - input_ids: 当前输入的 token 序列
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
def temperature_logits_warper(input_ids, logits):
    """
    实现温度系数处理，将 logits 除以温度系数来控制生成的随机性。

    参数:
    - input_ids: 当前输入的 token 序列
    - logits: torch.Tensor，模型的输出 logits

    返回值:
    - logits_processed: 经过温度系数处理后的 logits
    """
    logits_processed = logits / temperature  # 将 logits 除以温度系数
    return logits_processed


# 实现重复惩罚策略的 logits 处理
def repetition_penalty_logits_processor(input_ids, logits):
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


# logits 后处理函数，依次使用重复惩罚、温度调节、Top-k 和 Top-p 策略
def logits_wrap_process(input_ids, logits):
    """
    对 logits 进行处理，依次应用重复惩罚、温度、Top-k 和 Top-p 策略。

    参数:
    - input_ids: 当前输入的 token 序列
    - logits: torch.Tensor，模型的输出 logits

    返回值:
    - logits_processed: 经过多种策略处理后的 logits
    """
    logits_processed = repetition_penalty_logits_processor(input_ids, logits)
    logits_processed = temperature_logits_warper(input_ids, logits_processed)
    logits_processed = topk_logits_warper(input_ids, logits_processed)
    logits_processed = topp_logits_warper(input_ids, logits_processed)
    return logits_processed


# 实现下一 token 的预测
def predict_next_token(logits, input_ids):
    """
    根据 logits 预测下一个 token，并应用采样策略。

    参数:
    - logits: torch.Tensor，模型的输出 logits
    - input_ids: 当前输入的 token 序列

    返回值:
    - next_token_id: 预测出的下一个 token 的 ID
    """
    next_token_logits = logits_wrap_process(input_ids, logits.to(torch.float32))  # 处理 logits
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)  # 计算概率分布

    # 根据概率分布进行采样，选出下一个 token 的 ID
    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token_id


# ------------------------------- Part7-------------------------------------
# 定义一个函数来判断 token 是否是结束符号（EOS token）
def is_token_eos(token_id):
    """
    判断 token_id 是否是结束符号（EOS token）。

    参数:
    - token_id: int 或 list，表示当前生成的 token ID

    返回值:
    - bool，表示是否为 EOS token
    """
    # eos_token_id = [151645, 151643]  # 定义结束符号的 token ID
    eos_token_id = generation_config.eos_token_id
    return token_id in eos_token_id


# 模拟用户输入的提示信息（prompt）
# user_prompt = input("\n我是手写的AI模型，请输入你的问题：")
user_prompt = "一个星期有几天?"

# 应用聊天模板，将用户的输入转换为带模板的输入
prompt = my_apply_chat_template(user_prompt)

# 初始化 KV 缓存，用于存储和管理注意力层的 Key 和 Value
past_key_value = KVCache()

# 初始化输入的 token ID 和位置 ID
input_ids = None
position_id = None

# 设置最大生成的 token 数量
max_new_tokens = 20

# 初始化答案文本
answers = ""
print(f"\n\nUser Input: {user_prompt}\n")

# 循环生成 token，直到生成出答案或达到最大 token 限制
for _ in range(max_new_tokens):
    # 将用户的输入 prompt 转换为词嵌入向量
    prompt_ids, embeddings = my_word_embedding_process(prompt)

    # 如果 input_ids 尚未初始化，则将其设置为 prompt_ids
    input_ids = prompt_ids if input_ids is None else input_ids

    # 初始化 position_id，表示 token 的位置
    if position_id is None:
        text_len = prompt_ids.size()[-1]
        position_id = torch.arange(text_len).reshape(1, text_len).to(device)
    else:
        # 更新 position_id，表示生成的下一个 token 的位置
        position_id = torch.tensor([[text_len]]).to(device)
        text_len += 1

    # 调用模型的解码器模块，生成隐藏状态
    states, past_key_value = my_module(embeddings, past_key_value, position_id)

    # 使用 lm_head 将隐藏状态映射为 logits
    logits = my_lm_head(states)

    # 根据 logits 预测下一个 token 的 ID
    next_token_id = predict_next_token(logits, input_ids)

    # 将预测的 token ID 解码为实际的文本 token
    next_token = tokenizer.decode(next_token_id)

    # 更新 input_ids，追加生成的下一个 token
    input_ids = torch.cat([input_ids, next_token_id[:, None]], dim=-1)

    # 更新 prompt，将生成的下一个 token 作为新的输入
    prompt = next_token

    # 检查是否生成了结束符号（EOS token），如果是，则结束生成
    if is_token_eos(next_token_id):
        break

    # 累积生成的答案
    answers += next_token

    # 打印生成的 token ID 和对应的文本 token
    print(f"predict next token id: {next_token_id}, next word: {next_token}")

# 打印最终生成的答案
print(f"\nAnswer: {answers}")
