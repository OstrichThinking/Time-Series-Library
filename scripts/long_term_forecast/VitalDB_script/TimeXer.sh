export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=TimeXer

root_path=/home/share/ioh/VitalDB_IOH/
data_path=vitaldb

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_450_150_global_standardization \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len 450 \
  --label_len 150 \
  --pred_len 150 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 5 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 32 \
  --itr 1 \
  --train_epochs 10 \
  --use_multi_gpu