# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

from transformers import AutoModelForCausalLM

device = "cpu"

model_name = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)

state_dict = model.state_dict()

for name, param in state_dict.items():
    print(f"{name}")

print(model.model.embed_tokens.weight)
print(model.model.embed_tokens.weight.shape)
