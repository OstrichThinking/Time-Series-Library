export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=iTransformer

# A100
# root_path=/home/data/ioh/VitalDB_IOH/
# data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv

# V100
root_path=/home/share/ioh/VitalDB_IOH/ioh_dataset_with_medication/
data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_450_150_aaai \
  --model $model_name \
  --data VitalDB \
  --features MS \
  --seq_len 450 \
  --label_len 225 \
  --pred_len 150 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --train_epochs 10 \
  --num_workers 32 \
  --batch_size 64 \
  --use_multi_gpu \
  --devices 0,1,2,3 \
  --inverse