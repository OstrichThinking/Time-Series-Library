python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic \
    --model SimMTM \
    --data Traffic \
    --features M \
    --input_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 8 \
    --train_epochs 50 \
    --gpu 0
