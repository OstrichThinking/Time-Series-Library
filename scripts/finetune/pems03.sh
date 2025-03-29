for pred_len in 12 24 36 48; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path PEMS03.npz \
        --model_id PEMS03 \
        --model TimeDART \
        --data PEMS03 \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --n_heads 8 \
        --d_model 512 \
        --d_ff 512 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 32 \
        --gpu 0 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0005 \
        --pct_start 0.3
done
