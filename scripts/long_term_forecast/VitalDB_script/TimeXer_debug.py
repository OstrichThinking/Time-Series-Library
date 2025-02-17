import os
import runpy
import sys

os.chdir("/home/zhud/fist/ioh/Time-Series-Library/")

# 设置只使用一张 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 定义模型名称和路径
model_name = 'TimeXer'
root_path = '/home/share/ioh/VitalDB_IOH/ioh_dataset_with_medication/'
data_path = 'vitaldb_ioh_dataset_with_medication_invasive_group.csv'

# args = 'python -m src.test'
# args = 'python -m src.dataloader

static_features = ['caseid', 'sex', 'age', 'bmi']  
dynamic_features = ['Solar8000/ART_DBP_window_sample',
                    'Solar8000/ART_MBP_window_sample',
                    'Solar8000/ART_SBP_window_sample',
                    'Solar8000/BT_window_sample',
                    'Solar8000/HR_window_sample',
                    # 用药
                    'Orchestra/PPF20_CE_window_sample',
                    'Orchestra/PPF20_CP_window_sample',
                    'Orchestra/PPF20_CT_window_sample',
                    'Orchestra/PPF20_RATE_window_sample',
                    # 'Orchestra/PPF20_VOL',
                    'Orchestra/RFTN20_CE_window_sample',
                    'Orchestra/RFTN20_CP_window_sample',
                    'Orchestra/RFTN20_CT_window_sample',
                    'Orchestra/RFTN20_RATE_window_sample',
                    # 'Orchestra/RFTN20_VOL',
                    # 呼吸相关
                    'Solar8000/ETCO2_window_sample',
                    'Solar8000/FEO2_window_sample',
                    'Solar8000/FIO2_window_sample',
                    'Solar8000/INCO2_window_sample',
                    'Solar8000/VENT_MAWP_window_sample',
                    'Solar8000/VENT_MV_window_sample',
                    'Solar8000/VENT_RR_window_sample',
                    'prediction_maap'] 
static_features_str = ' '.join(static_features)
dynamic_features_str = ' '.join(dynamic_features)

args=f"python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path {root_path} \
  --data_path {data_path} \
  --model_id vitaldb_450_150_aaai_with_medicine_with_respiratory \
  --model {model_name} \
  --data VitalDB \
  --features MS \
  --static_features {static_features_str} \
  --dynamic_features {dynamic_features_str} \
  --seq_len 450 \
  --label_len 225 \
  --pred_len 150 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 23 \
  --dec_in 23 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 64 \
  --train_epochs 1 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0 \
  --inverse"


args = args.split()
if args[0]== 'python':
    """pop up the first in the args"""
    args.pop(0)
if args[0]=='-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0],run_name='__main__')