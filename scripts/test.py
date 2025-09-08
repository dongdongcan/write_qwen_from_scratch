#!/bin/python3

import os
import subprocess

# Get the absolute path of the current script and its directory
script_path = os.path.realpath(__file__)
abs_path = os.path.dirname(script_path)

# List of files to test (relative to the src directory)
files = [
    "src/qwen2_v1/part1/my_chat_template.py",
    "src/qwen2_v1/part1/my_word_embed.py",
    "src/qwen2_v1/part2/sdpa.py",
    "src/qwen2_v1/part2/mha.py",
    "src/qwen2_v1/part2/gqa.py",
    "src/qwen2_v1/part3/my_pos_embed.py",
    "src/qwen2_v1/part4/my_kvcache.py",
    "src/qwen2_v1/part4/gqa_with_rope.py",
    "src/qwen2_v1/part4/gqa_with_rope_kvcache.py",
    "src/qwen2_v1/part4/my_gqa.py",
    "src/qwen2_v1/part5/my_decoder_module.py",
    "src/qwen2_v1/part6/my_post_process.py",
    "src/qwen2_v1/part7/build_my_llm.py",
    "src/qwen2_v1/my_qwen2_cpu.py",
    "src/qwen2_v2/my_qwen2_cpu.py",
    "models/qwen2/qwen2_0.5B.py",
]

# Iterate through each file in the list
for file in files:
    # Construct the full file path relative to the src directory
    file_path = os.path.join(abs_path, "..", file)
    print(f"Testing {file_path} ...")

    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"Cmd: python3 {file_path}")
        try:
            # Execute the Python file using subprocess
            result = subprocess.run(["python3", file_path], capture_output=True, text=True)

            # Check the return code to determine success or failure
            if result.returncode == 0:
                print(f"{file_path} executed successfully.")
            else:
                print(f"Error occurred while executing {file_path}.")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Exception occurred while executing {file_path}: {str(e)}")
    else:
        print(f"File {file_path} does not exist.")

    print("-----------------------------")
