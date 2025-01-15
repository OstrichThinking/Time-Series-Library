export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=iTransformer

root_path=/home/share/ioh/VitalDB_IOH/
data_path=vitaldb

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_450_150 \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --train_epochs 10 \
  --batch_size 64 \
  --use_multi_gpu