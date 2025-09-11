# Copyright (c) 2024 dongdongcan
# This code is licensed under the Apache License.
# See the LICENSE file for details.

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tools.args import parse_args


def main(args):
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    if args.model:
        assert args.model == model_name, f"only support {model_name} now"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)

    prompt = "who are you ?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.verbose:
        print(f"prompt before template: {prompt}")
        print(f"prompt after template: {text}")

    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=10)
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"promot: {prompt}")
    print(f"response: {response}")


if __name__ == "__main__":
    args = parse_args(model_required=False)
    main(args)
