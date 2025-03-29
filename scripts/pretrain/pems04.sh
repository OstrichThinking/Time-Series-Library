python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04 \
    --model TimeDART \
    --data PEMS04 \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --n_heads 8 \
    --d_model 512 \
    --d_ff 512 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.9 \
    --learning_rate 0.0005 \
    --batch_size 8 \
    --train_epochs 20 \
    --gpu 0 \
    --use_norm 0
