# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from part1.my_model_params import tokenizer


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


# 测试函数
if __name__ == "__main__":
    # 定义测试用的用户输入
    prompt = "一个星期有几天？"

    # 调用模板函数生成完整的对话文本
    text = my_apply_chat_template(prompt)

    # 输出生成的文本
    print(text)
