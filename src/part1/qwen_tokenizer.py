# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Hello World, I am great"

input_ids = tokenizer.encode(prompt)
print(input_ids)

input_tokens = tokenizer.decode(input_ids)
print(input_tokens)
