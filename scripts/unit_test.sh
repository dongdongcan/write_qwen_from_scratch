#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
ABS_PATH=$(dirname "$SCRIPT_PATH")

FILES=(
    "v1/part1/my_chat_template.py"
    "v1/part1/my_word_embed.py"
    "v1/part2/sdpa.py"
    "v1/part2/mha.py"
    "v1/part2/gqa.py"
    "v1/part3/my_pos_embed.py"
    "v1/part4/my_kvcache.py"
    "v1/part4/gqa_with_rope.py"
    "v1/part4/gqa_with_rope_kvcache.py"
    "v1/part4/my_gqa.py"
    "v1/part5/my_decoder_module.py"
    "v1/part6/my_post_process.py"
    "v1/part7/build_my_llm.py"
    "v1/my_qwen.py"
    "v2/qwen2_from_scratch.py"
)


for F in "${FILES[@]}"; do
    FILE=$ABS_PATH/../src/$F
    echo "Testing $FILE ..."

    if [[ -f $FILE ]]; then
	echo "Cmd: python3" $FILE
        python3 $FILE
        
        if [[ $? -eq 0 ]]; then
            echo "$FILE executed successfully."
        else
            echo "Error occurred while executing $FILE."
        fi
    else
        echo "File $FILE does not exist."
    fi
    
    echo "-----------------------------"
done
