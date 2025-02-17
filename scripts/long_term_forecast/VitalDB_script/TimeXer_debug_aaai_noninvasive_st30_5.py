import os
import runpy
import sys

# nohup python -u scripts/long_term_forecast/VitalDB_script/TimeXer_debug_aaai_noninvasive_st30_5.py > output.log 2>&1 &
os.chdir("/home/cuiy/project/Time-Series-Library/")

# 设置只使用一张 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 定义模型名称和路径
model_name = 'TimeXer'
root_path = '/home/share/ioh/VitalDB_IOH/cma_ioh/'
# data_path = 'vitaldb_ioh_dataset_with_medication_invasive_group.csv'
data_path = 'ioh_dataset_noninvasive_st30_5.csv'

# 定义IOH需要处理的静态特征和波形数据
static_features = ['caseid', 'sex', 'age', 'bmi']  
dynamic_features = ['Solar8000/NIBP_DBP_window_sample',     # 无创舒张压
                    'Solar8000/NIBP_MBP_window_sample',     # 无创平均动脉压
                    'Solar8000/BT_window_sample',           # 体温
                    'Solar8000/HR_window_sample',           # 心率
                    'prediction_maap'] 
# dynamic_features = ['Solar8000/ART_DBP_window_sample', 
#                     'Solar8000/ART_MBP_window_sample',
#                     'Solar8000/ART_SBP_window_sample',
#                     'Solar8000/BT_window_sample',
#                     'Solar8000/HR_window_sample',
#                     'prediction_maap'] 
static_features_str = ' '.join(static_features)
dynamic_features_str = ' '.join(dynamic_features)

# args = 'python -m src.test'
# args = 'python -m src.dataloader
  
# 单GPU
args=f"python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path {root_path} \
  --data_path {data_path} \
  --model_id vitaldb_aaai_noninvasive_st30_5 \
  --model {model_name} \
  --data VitalDB \
  --features MS \
  --static_features {static_features_str} \
  --dynamic_features {dynamic_features_str} \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 10 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --des Exp \
  --d_model 256 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 64 \
  --train_epochs 1 \
  --num_workers 32 \
  --use_multi_gpu \
  --devices 0,1,2,3 \
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