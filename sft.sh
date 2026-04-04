# Example training script for COBOL-Coder
# Fine-tune Qwen2.5-Coder-14B on COBOL-specific datasets

CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3 deepspeed src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-Coder-14B-Instruct \
    --dataset  COBOL_text_QA,new_COBOL_code,mainframe_instruct,cobol_to_java_semanticChecking,cobol_problem_from_java_semanticChecking,cobol_java_pairs_semanticChecking,mainframe_instruct_code_generation,glaive_code_assistant,self_oss_instruct,fineTome-100k\
    --template qwen \
    --finetuning_type full \
    --output_dir ./output/qwen2.5-coder-14b/sft \
    --cache_dir .cache \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --plot_loss \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --bf16 True \
    --flash_attn fa2 \
    --enable_liger_kernel True \
    --gradient_checkpointing True \
    --seed 42 \
    --packing False \
    --preprocessing_num_workers 16 \
    --report_to wandb
