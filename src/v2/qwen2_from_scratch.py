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


class PostProcess:
    def __init__(self, config) -> None:
        self.config = config

    def topk_logits_warper(self, input_ids, logits):
        filter_value = -float("Inf")
        top_k_temp = min(self.config.top_k, logits.size(-1))

        indices_to_remove = logits < torch.topk(logits, top_k_temp)[0][..., -1, None]
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed

    def topp_logits_warper(self, input_ids, logits):
        min_tokens_to_keep = 1
        filter_value = -float("Inf")

        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - self.config.top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed

    def temperature_logits_warper(self, input_ids, logits):
        logits_processed = logits / self.config.temperature
        return logits_processed

    def repetition_penalty_logits_processor(self, input_ids, logits):
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * self.config.repetition_penalty, score / self.config.repetition_penalty)
        logits_processed = logits.scatter(1, input_ids, score)
        return logits_processed

    def logits_wrap_process(self, input_ids, logits):
        logits_processed = self.repetition_penalty_logits_processor(input_ids, logits)
        logits_processed = self.temperature_logits_warper(input_ids, logits_processed)
        logits_processed = self.topk_logits_warper(input_ids, logits_processed)
        logits_processed = self.topp_logits_warper(input_ids, logits_processed)
        return logits_processed

    def predict_next_token(self, logits, input_ids):
        next_token_logits = self.logits_wrap_process(input_ids, logits.to(torch.float32))
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_token_id


class KVCache:
    def __init__(self) -> None:
        self.KCache: List[torch.tensor] = []
        self.VCache: List[torch.tensor] = []

    def update(self, new_key_states, new_value_states, layer_idx):
        if len(self.KCache) <= layer_idx:
            self.KCache.append(new_key_states)
            self.VCache.append(new_value_states)
        else:
            self.KCache[layer_idx] = torch.cat([self.KCache[layer_idx], new_key_states], dim=-2)
            self.VCache[layer_idx] = torch.cat([self.VCache[layer_idx], new_value_states], dim=-2)
        return self.KCache[layer_idx], self.VCache[layer_idx]

    def get_seq_length(self, layer_idx) -> int:
        if len(self.KCache) <= layer_idx:
            return 0
        return self.KCache[layer_idx].shape[-2]

    def print(self, layer_idx):
        if len(self.KCache) == 0:
            print("缓存为空")
        else:
            print(f"层 {layer_idx} 缓存的 token 数：", self.KCache[layer_idx].shape[-2])


def generate_rope_matrix(hidden_size, max_position_embeddings):
    seq_list = torch.arange(0, hidden_size, 2, dtype=torch.int64).float()
    seq_list = seq_list / hidden_size
    seq_list = 10000 ** seq_list
    theta = 1.0 / seq_list
    t = torch.arange(max_position_embeddings, dtype=torch.int64).type_as(theta)
    freqs = torch.outer(t, theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_val = emb.cos().to(torch.bfloat16)
    sin_val = emb.sin().to(torch.bfloat16)
    return sin_val, cos_val


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids]
    sin = sin[position_ids]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen:
    def __init__(self, model_name, max_new_tokens=20):
        self.model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        self.config = Qwen2Config.from_pretrained(model_name)
        self.feature_per_head = (int)(self.config.hidden_size / self.config.num_attention_heads)
        self.groups = (int)(self.config.num_attention_heads / self.config.num_key_value_heads)

        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.post = PostProcess(self.generation_config)
        self.kv_cache = KVCache()
        self.sin_matrix, self.cos_matrix = generate_rope_matrix(
            self.feature_per_head, self.config.max_position_embeddings
        )

    def apply_chat_template(self, prompt):
        prompt_encoding = self.tokenizer.encode(prompt)
        template = (
            [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]
            + prompt_encoding
            + [151645, 198, 151644, 77091, 198]
        )

        return self.tokenizer.decode(template)

    def embedding(self, prompt):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids[None, :]
        word_embeddings = torch.nn.functional.embedding(input_ids, self.model.model.embed_tokens.weight)
        return input_ids, word_embeddings

    def mlp(self, layer_idx, hidden_state: torch.tensor):
        gate_proj = torch.nn.functional.linear(hidden_state, self.model.model.layers[layer_idx].mlp.gate_proj.weight)
        up_proj = torch.nn.functional.linear(hidden_state, self.model.model.layers[layer_idx].mlp.up_proj.weight)
        down_proj = torch.nn.functional.linear(
            torch.functional.F.silu(gate_proj) * up_proj, self.model.model.layers[layer_idx].mlp.down_proj.weight
        )
        return down_proj

    def gqa(self, layer_idx, hidden_states, kv_cache, position_id):
        seq_length = hidden_states.size()[-2]
        query_states = torch.nn.functional.linear(
            hidden_states,
            self.model.model.layers[layer_idx].self_attn.q_proj.weight,
            self.model.model.layers[layer_idx].self_attn.q_proj.bias,
        )

        key_states = torch.nn.functional.linear(
            hidden_states,
            self.model.model.layers[layer_idx].self_attn.k_proj.weight,
            self.model.model.layers[layer_idx].self_attn.k_proj.bias,
        )

        value_states = torch.nn.functional.linear(
            hidden_states,
            self.model.model.layers[layer_idx].self_attn.v_proj.weight,
            self.model.model.layers[layer_idx].self_attn.v_proj.bias,
        )

        query_states = query_states.view(1, seq_length, self.config.num_attention_heads, self.feature_per_head)

        key_states = key_states.view(1, seq_length, self.config.num_key_value_heads, self.feature_per_head)
        value_states = value_states.view(1, seq_length, self.config.num_key_value_heads, self.feature_per_head)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = seq_length + kv_cache.get_seq_length(layer_idx)
        cos = self.cos_matrix[:kv_seq_len]
        sin = self.sin_matrix[:kv_seq_len]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_id)

        key_states, value_states = kv_cache.update(key_states, value_states, layer_idx)
        key_states = torch.repeat_interleave(key_states, repeats=self.groups, dim=1)
        value_states = torch.repeat_interleave(value_states, repeats=self.groups, dim=1)
        is_causal = seq_length > 1
        attention_out = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=is_causal
        )
        attention_out = attention_out.transpose(1, 2)
        attention_out = attention_out.reshape(1, seq_length, self.config.hidden_size)
        attn_out = torch.nn.functional.linear(attention_out, self.model.model.layers[layer_idx].self_attn.o_proj.weight)

        return attn_out, kv_cache

    def rms_norm(self, states, weights, eps):
        states = states.to(torch.float32)

        variance = states.pow(2).mean(-1, keepdim=True)
        states = states * torch.rsqrt(variance + eps)
        return weights * states.to(weights.dtype)

    def decoder_layer(self, states, layer_idx, position_id, past_key_value):
        residual = states
        out = self.rms_norm(states, self.model.model.layers[layer_idx].input_layernorm.weight, self.config.rms_norm_eps)
        out, present_key_value = self.gqa(layer_idx, out, past_key_value, position_id)
        out = out + residual

        residual = out
        out = self.rms_norm(out, self.model.model.layers[layer_idx].post_attention_layernorm.weight, self.config.rms_norm_eps)
        out = self.mlp(layer_idx, out)
        out = out + residual
        return out, present_key_value

    def forward(self, states, past_key_value, position_id):
        for layer_idx in range(self.config.num_hidden_layers):
            states, present_key_value = self.decoder_layer(states, layer_idx, position_id, past_key_value)

        past_key_value = present_key_value
        states = self.rms_norm(states, self.model.model.norm.weight, self.config.rms_norm_eps)
        return states, past_key_value

    def lm_head(self, states):
        logits = torch.nn.functional.linear(states[-1, -1, :], self.model.lm_head.weight)
        logits = logits[None, :]
        return logits

    def is_token_eos(self, token_id):
        eos_token_id = self.generation_config.eos_token_id
        return token_id in eos_token_id

    def generate(self, user_input):
        prompt = self.apply_chat_template(user_input)
        past_key_value = KVCache()
        # 初始化输入的 token ID 和位置 ID
        input_ids = None
        position_id = None

        answers = ""
        print(f"\n\nUser Input: {user_input}\n")

        for _ in range(self.max_new_tokens):
            prompt_ids, embeddings = self.embedding(prompt)

            # 如果 input_ids 尚未初始化，则将其设置为 prompt_ids
            input_ids = prompt_ids if input_ids is None else input_ids

            # 初始化 position_id，表示 token 的位置
            if position_id is None:
                text_len = prompt_ids.size()[-1]
                position_id = torch.arange(text_len).reshape(1, text_len)
            else:
                # 更新 position_id，表示生成的下一个 token 的位置
                position_id = torch.tensor([[text_len]])
                text_len += 1

            # 调用模型的解码器模块，生成隐藏状态
            states, past_key_value = self.forward(embeddings, past_key_value, position_id)

            # 使用 lm_head 将隐藏状态映射为 logits
            logits = self.lm_head(states)

            # 根据 logits 预测下一个 token 的 ID
            next_token_id = self.post.predict_next_token(logits, input_ids)

            # 将预测的 token ID 解码为实际的文本 token
            next_token = self.tokenizer.decode(next_token_id)

            # 更新 input_ids，追加生成的下一个 token
            input_ids = torch.cat([input_ids, next_token_id[:, None]], dim=-1)

            # 更新 prompt，将生成的下一个 token 作为新的输入
            prompt = next_token

            # 检查是否生成了结束符号（EOS token），如果是，则结束生成
            if self.is_token_eos(next_token_id):
                break

            # 累积生成的答案
            answers += next_token

            # 打印生成的 token ID 和对应的文本 token
            print(f"predict next token id: {next_token_id}, next word: {next_token}")

        return answers


def main():
    user_input = "一个星期有几天?"
    model_name = get_model_name()
    model = Qwen(model_name)
    response = model.generate(user_input)
    print(f"{response}")


if __name__ == "__main__":
    main()
