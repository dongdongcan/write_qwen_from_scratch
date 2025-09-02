#!/bin/bash

SCRIPT_PATH=$(readlink -f "$0")
ABS_PATH=$(dirname "$SCRIPT_PATH")

FILES=(
    # "version_1/part1/my_chat_template.py"
    # "version_1/part1/my_word_embed.py"
    # "version_1/part2/sdpa.py"
    # "version_1/part2/mha.py"
    # "version_1/part2/gqa.py"
    # "version_1/part3/my_pos_embed.py"
    # "version_1/part4/my_kvcache.py"
    # "version_1/part4/gqa_with_rope.py"
    # "version_1/part4/gqa_with_rope_kvcache.py"
    # "version_1/part4/my_gqa.py"
    # "version_1/part5/my_decoder_module.py"
    # "version_1/part6/my_post_process.py"
    # "version_1/part7/build_my_llm.py"
    # "version_1/my_qwen.py"
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
