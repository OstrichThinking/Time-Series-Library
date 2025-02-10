export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeXer

root_path=/home/data/ioh/VitalDB_IOH/
data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_in_450_150_without_medicine \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len 450 \
  --label_len 225 \
  --pred_len 150 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 64 \
  --train_epochs 10 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0,1,2