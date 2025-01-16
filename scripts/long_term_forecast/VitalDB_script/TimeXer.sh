export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeXer

root_path=/home/fist/ostrich/VitalDB_IOH
data_path=vitaldb

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_450_150_without_static_and_medicine \
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
  --itr 1 \
  --batch_size 64 \
  --train_epochs 10 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0,1,2