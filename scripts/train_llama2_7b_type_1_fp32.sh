python train.py \
    --mode train \
    --use_trans_e True \
    --out_path_aware True \
    --out_path_shuffle True \
    --state_embed_size 4096 \
    --hidden_dim 4096 \
    --max_step_length 2 \
    --train_step_length 1 \
    --max_patience 5 \
    --evaluate_freq 5 \
    --batch_size 4096 \
    --mini_batch_size 1024 \
    --train_batch_size 2 \
    --test_batch_size 32 \
    --stater_type llama \
    --instruction_type 1 \
    --stater_path /root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc \
    --stater_cache_dir "" \
    --actor_checkpoint_path "runs/LLM-ARK/TRAINASSISTANT/2024-01-31_11-06-39/actor.pkt" \
    --rl_train_data_path datasets/OpenDialKG/Reason/train_type_1.json
    