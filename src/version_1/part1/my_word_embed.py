# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
import os
import sys

# 将项目的上一层目录添加到 Python 的模块搜索路径 sys.path 中
# 这样可以在当前文件中导入其他目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# 从 part1 目录中导入相关模块和函数
from part1.my_model_params import model, tokenizer, config
from part1.my_chat_template import my_apply_chat_template


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
    input_ids = torch.tensor(input_ids)

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


# 测试 my_word_embedding_process 函数
if __name__ == "__main__":
    # 用户输入的测试文本
    user_prompt = "一个星期有几天?"

    # 调用自定义的聊天模板函数，将用户输入的文本应用聊天模板
    prompt = my_apply_chat_template(user_prompt)

    # 调用词嵌入处理函数，得到输入的 token ID 和对应的词嵌入向量
    prompt_ids, embeddings = my_word_embedding_process(prompt)

    # 打印 token ID
    print(prompt_ids)

    # 使用 tokenizer 将 token ID 解码回原始文本，验证编码解码过程是否正确
    print(tokenizer.decode(prompt_ids[0]))

    # 打印词嵌入向量
    print(f"词嵌入向量：\n{embeddings}")

    # 打印词嵌入向量的维度，显示为 (batch_size, seq_length, hidden_size)
    print(f"词嵌入向量的维度为: {embeddings.shape}")
