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
from part1.my_chat_template import my_apply_chat_template
from part1.my_word_embed import my_word_embedding_process
from part4.my_kvcache import KVCache
from part5.my_decoder_module import my_module
from part6.my_post_process import my_lm_head, my_predict_next_token


# 定义一个函数来判断 token 是否是结束符号（EOS token）
def is_token_eos(token_id):
    """
    判断 token_id 是否是结束符号（EOS token）。

    参数:
    - token_id: int 或 list，表示当前生成的 token ID

    返回值:
    - bool，表示是否为 EOS token
    """
    eos_token_id = [151645, 151643]  # 定义结束符号的 token ID
    return token_id in eos_token_id


if __name__ == "__main__":
    # 用户输入
    user_prompt = "一个星期有几天?"

    # 使用聊天模板，将用户的输入转换为带模板的输入
    prompt = my_apply_chat_template(user_prompt)

    # 初始化 KV 缓存，用于存储和管理注意力层的 Key 和 Value
    past_key_value = KVCache()

    # 初始化输入的 token ID, 对模型而言，每次输入的 token id
    input_ids = None

    # 初始化输入的token的位置id
    position_id = None

    # 设置模型支持的生成最大的 token 数量
    max_new_tokens = 20

    # 初始化答案文本
    answers = ""
    print(f"\n\nUser Input: {user_prompt}\n")

    # 循环生成 token，直到生成出答案或达到最大 token 限制
    for _ in range(max_new_tokens):
        # 将用户的输入 prompt 转换为词嵌入向量，以及 token_id
        token_ids, embeddings = my_word_embedding_process(prompt)

        # 如果 input_ids 尚未初始化，则将其设置为 token_ids
        input_ids = token_ids if input_ids is None else input_ids

        # 初始化 position_id，表示 token 的位置
        # 第一次 prefill 阶段，position id 代表了用户输入的问题
        # 假设用户输入的问题是：“你好呀？”， 那么 position_id 则为 [0,1,2,3]
        if position_id is None:
            text_len = token_ids.size()[-1]
            position_id = torch.arange(text_len).reshape(1, text_len)
        else:
            # 更新 position_id，表示生成的下一个 token 的位置
            # decode 阶段，每次输出一个token， text_len 每次加1
            position_id = torch.tensor([[text_len]])
            text_len += 1

        # 调用自定义的模型进行推理
        states, past_key_value = my_module(embeddings, past_key_value, position_id)

        # 模型后面的全连接完成隐藏特征到样本特征的映射转换
        logits = my_lm_head(states)

        # 根据 logits 预测下一个 token id，包含后处理过程
        next_token_id = my_predict_next_token(logits, input_ids)
        print(next_token_id)
        # 将预测的 token ID 解码为实际的文本 token
        next_token = tokenizer.decode(next_token_id)
        # 更新 input_ids，将本次预测生成的token id 追加到历史 input_ids 中
        # input_ids 的维度为 [1, seq_lenth], next_token_id[:, None] 使 next_token_id维度
        # 变为 [1, 1], 方便 torch.cat 操作
        input_ids = torch.cat([input_ids, next_token_id[:, None]], dim=-1)

        # 更新 prompt，将预测生成的 token 作为模型下一次新的输入
        prompt = next_token

        # 判断本次生成的 token id 是否是结束符，如果是，模型便停止
        if is_token_eos(next_token_id):
            break

        # 将本次生成的 token 追加到 answers 中
        answers += next_token

        # 打印本次预测生成的 token ID 和对应的 token
        print(f"本次预测结果，下一个词:{next_token}, id:{next_token_id}")

    # 打印最终生成的答案
    print(f"\nAnswer: {answers}")
