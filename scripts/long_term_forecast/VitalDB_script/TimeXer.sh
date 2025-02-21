export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=TimeXer

# A100
# root_path=/home/data/ioh/VitalDB_IOH/
# data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv

# V100
root_path=/home/share/ioh/VitalDB_IOH/ioh_dataset_with_medication/
data_path=vitaldb_ioh_dataset_with_medication_invasive_group.csv

static_features="caseid sex age bmi"
dynamic_features="window_sample_time \
                  Solar8000/ART_DBP_window_sample \
                  Solar8000/ART_MBP_window_sample \
                  Solar8000/ART_SBP_window_sample \
                  Solar8000/BT_window_sample \
                  Solar8000/HR_window_sample \
                  Orchestra/PPF20_CE_window_sample \
                  Orchestra/PPF20_CP_window_sample \
                  Orchestra/PPF20_CT_window_sample \
                  Orchestra/PPF20_RATE_window_sample \
                  Orchestra/RFTN20_CE_window_sample \
                  Orchestra/RFTN20_CP_window_sample \
                  Orchestra/RFTN20_CT_window_sample \
                  Orchestra/RFTN20_RATE_window_sample \
                  Solar8000/ETCO2_window_sample \
                  Solar8000/FEO2_window_sample \
                  Solar8000/FIO2_window_sample \
                  Solar8000/INCO2_window_sample \
                  Solar8000/VENT_MAWP_window_sample \
                  Solar8000/VENT_MV_window_sample \
                  Solar8000/VENT_RR_window_sample \
                  prediction_window_time \
                  prediction_maap"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id vitaldb_450_150_aaai_with_medicine_with_respiratory_with_classification_with_time_embedding\
  --model $model_name \
  --data VitalDB \
  --features MS \
  --static_features $static_features \
  --dynamic_features $dynamic_features \
  --seq_len 450 \
  --label_len 225 \
  --pred_len 150 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 23 \
  --dec_in 23 \
  --c_out 1 \
  --embed surgicalF \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 64 \
  --train_epochs 50 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0,1,2,3 \
  --inverse