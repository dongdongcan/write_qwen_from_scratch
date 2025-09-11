#!/bin/python3

import os
import subprocess
import sys

# Get the absolute path of the current script and its directory
script_path = os.path.realpath(__file__)
abs_path = os.path.dirname(script_path)

# List of files to test (relative to the src directory)
cases = {
    "src/qwen2_v1/part1/my_chat_template.py": None,
    "src/qwen2_v1/part1/my_word_embed.py": None,
    "src/qwen2_v1/part2/sdpa.py": None,
    "src/qwen2_v1/part2/mha.py": None,
    "src/qwen2_v1/part2/gqa.py": None,
    "src/qwen2_v1/part3/my_pos_embed.py": None,
    "src/qwen2_v1/part4/my_kvcache.py": None,
    "src/qwen2_v1/part4/gqa_with_rope.py": None,
    "src/qwen2_v1/part4/gqa_with_rope_kvcache.py": None,
    "src/qwen2_v1/part4/my_gqa.py": None,
    "src/qwen2_v1/part5/my_decoder_module.py": None,
    "src/qwen2_v1/part6/my_post_process.py": None,
    "src/qwen2_v1/part7/build_my_llm.py": None,
    "src/qwen2_v1/my_qwen2_cpu.py": None,
    "src/qwen2_v2/my_qwen2_cpu.py": {
        "model": "Qwen2-0.5B-Instruct",
        "to_console": True,
    },
    "models/qwen2/qwen2_0.5B.py": {"to_console": True},
}


for file, params in cases.items():
    file_path = os.path.join(abs_path, "..", file)
    print(f"Testing {file_path} ...")

    cmd = ["python3", file_path]
    # when capture_output is False, the output will be printed directly to the console
    to_console = False
    if params is not None:
        if "model" in params:
            cmd.append(f"--model={params['model']}")
        if "to_console" in params:
            to_console = params["to_console"]

    if os.path.isfile(file_path):
        print(f"Cmd: {cmd}")
        try:
            # Execute the Python file using subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)

            if to_console:
                print(result.stdout.strip())

            if result.returncode == 0:
                print(f"{file_path} executed successfully.")
            else:
                print(f"Error occurred while executing {file_path}.")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Exception occurred while executing {file_path}: {str(e)}")
            sys.exit(1)
    else:
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    print("-" * 50)
