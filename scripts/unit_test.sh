#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
ABS_PATH=$(dirname "$SCRIPT_PATH")

FILES=(
    "part1/my_chat_template.py"
    "part1/my_word_embed.py"
    "part2/sdpa.py"
    "part2/mha.py"
    "part2/gqa.py"
    "part3/my_pos_embed.py"
    "part4/my_kvcache.py"
    "part4/gqa_with_rope.py"
    "part4/gqa_with_rope_kvcache.py"
    "part4/my_gqa.py"
    "part5/my_decoder_module.py"
    "part6/my_post_process.py"
    "part7/build_my_llm.py"
    "my_qwen.py"
)


for F in "${FILES[@]}"; do
    FILE=$ABS_PATH/../src/$F
    echo "Testing $FILE ..."

    if [[ -f $FILE ]]; then
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
