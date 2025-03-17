export CUDA_VISIBLE_DEVICES=0

model_name=AHCformer

swan_project='tsl'
swan_workspace='Jude'

# +--------------------+--------------------+--------------------+
# |        MSE         |        MAE         |        DTW         |
# +--------------------+--------------------+--------------------+
# | 0.7017099857330322 | 0.6542699933052063 |   Not calculated   |
# +--------------------+--------------------+--------------------+

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
    --swan_project $swan_project \
  --swan_workspace $swan_workspace \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1 \