export CUDA_VISIBLE_DEVICES=7

root_path=/home/data/ioh/VitalDB_IOH/
data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv
model_name=Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_in_450_150_aaai_50_epochs \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len 450 \
  --label_len 225 \
  --pred_len 150 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 64 \
  --train_epochs 50 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0