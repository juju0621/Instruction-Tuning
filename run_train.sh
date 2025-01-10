python3 train.py \
        --model_name_or_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" \
        --train "./data/train.json" \
        --train_val_split 0.2 \
        --max_length 512 \
        --num_train_epochs 1 \
        --gradient_accumulation_step 1 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --lr_scheduler "cosine" \
        --num_warmup_steps 300 \
        --weight_decay 1e-4 \
        --lora_rank 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --checkpointing_steps 5000 \
        --output_dir "./output_dir" \
        --seed 1337