# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

import torch
from typing import List


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


# 用于测试 KVCache 类的功能
if __name__ == "__main__":
    # 初始化新的 Key 和 Value 状态，假设有 5 个 token，每个 token 的特征维度为 896
    new_k_state = torch.randn(1, 5, 896)
    new_v_state = torch.randn(1, 5, 896)

    # 初始化 KVCache 实例
    kv_cache = KVCache()

    # 打印第 0 层的缓存状态，初始时应该为空
    kv_cache.print(0)

    # 更新第 0 层的缓存，将新生成的 Key 和 Value 状态添加进去
    kv_cache.update(new_k_state, new_v_state, 0)

    # 再次打印第 0 层的缓存状态，此时应该有 5 个 token 被缓存
    kv_cache.print(0)

    # 初始化新的 Key 和 Value 状态，假设有 1 个 token，每个 token 的特征维度为 896
    new_k_state1 = torch.randn(1, 1, 896)
    new_v_state1 = torch.randn(1, 1, 896)

    # 更新第 1 层的缓存，将新生成的 Key 和 Value 状态添加进去
    kv_cache.update(new_k_state1, new_v_state1, 1)

    # 打印第 1 层的缓存状态，此时应该有 1 个 token 被缓存
    kv_cache.print(1)

    # 初始化新的 Key 和 Value 状态，假设有 2 个 token，每个 token 的特征维度为 896
    new_k_state2 = torch.randn(1, 2, 896)
    new_v_state2 = torch.randn(1, 2, 896)

    # 更新第 0 层的缓存，将新生成的 Key 和 Value 状态添加进去
    kv_cache.update(new_k_state2, new_v_state2, 0)

    # 再次打印第 0 层的缓存状态，此时应该有 7 个 token 被缓存
    kv_cache.print(0)
