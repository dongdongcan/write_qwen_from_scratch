# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

from transformers.models.qwen2 import (
    Qwen2TokenizerFast,
    Qwen2ForCausalLM,
    Qwen2Config,
)

# 获取模型的模型结构、tokenizer 和 config 参数，供其他 Part 的函数使用

# 定义要加载的模型名称，这里是 "Qwen/Qwen2-0.5B-Instruct"
model_name = "Qwen/Qwen2-0.5B-Instruct"

# 从预训练模型中加载模型结构
model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto")

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
