python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07 \
    --model TimeDART \
    --data PEMS07 \
    --features M \
    --input_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
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
    --gpu 1 \
    --use_norm 0
