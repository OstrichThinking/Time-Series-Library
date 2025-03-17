import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 Transformer 模型，对 VitalDB 数据集进行长期预测。
        - 30个点预测10个点 (15min预测5min)
    
    🏠数据集：
        - ioh_dataset_noninvasive_st30_5.csv 
        - 无创组，总计 2065 个cases
        - 每隔30s取一个点，15min预测15min，滑动窗口步长150s（2.5min）
        - 使用“性别、年龄、BMI、观察窗口时间、无创舒张压、无创平均动脉压、体温、心率、预测窗口时间”预测“无创平均动脉压”
    
    🚀模型：
        - Transformer
    
    🔍训练参数：
        - 训练轮数: 50
        - 批次大小: 64
        - 学习率: 0.0001
    
    👋 实验后台启动命令
        nohup python -u scripts/long_term_forecast/VitalDB_script/Transformer_noninvasive_st30_5_surgicalF.py > checkpoints/output_Transformer_noninvasive_st30_5_surgicalF.log 2>&1 &
    
    🌞实验结果:

"""

# A100项目路径
os.chdir("/home/temporal/cuiy/Time-Series-Library/")
# # V100项目路径
# os.chdir("/home/cuiy/project/Time-Series-Library/")

# 设置只使用一张 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO 定义模型名称和数据集路径
model_name = 'Transformer'
task_name = 'long_term_forecast'
model_id = f'invasive_st2_ops2e'  

# A100数据集路径
# root_path = '/home/data/ioh/cma_ioh/'
root_path = '/home/data/ioh/cma_ioh/invasive_ops2e/'
# # V100数据集路径
# root_path = '/home/share/ioh/VitalDB_IOH/cma_ioh/'

# data_path = 'ioh_dataset_noninvasive_st30_5.csv'
data_path = 'dataset_vitaldb_cma_invasive_st2_ops2e.jsonl'

seq_len = 450   # 预测窗口数据点数
label_len = 225 # 预测窗口加入label数据的点数
pred_len = 150  # 预测窗口数据点数

# TODO定义IOH需要处理的静态特征和波形数据
static_features = ['caseid', 'sex', 'age', 'bmi', 'time']
dynamic_features = [
    'seq_time_stamp_list',
    'pred_time_stamp_list',
    'Solar8000/BT',
    'Solar8000/HR',
    'Solar8000/ART_DBP',
    'Solar8000/ART_MBP', 
    'prediction_maap'
]
# static_features = ['caseid', 'sex', 'age', 'bmi']  
# dynamic_features = ['window_sample_time',                   # 观察窗口采样时间范围
#                     'Solar8000/NIBP_DBP_window_sample',     # 无创舒张压                  
#                     'Solar8000/BT_window_sample',           # 体温
#                     'Solar8000/HR_window_sample',           # 心率          
#                     'Solar8000/NIBP_MBP_window_sample',     # 无创平均动脉压
#                     'prediction_window_time',               # 预测窗口时间范围
#                     'prediction_maap']                      # 需要预测的有创/无创平均动脉压

static_features_str = ' '.join(static_features)
dynamic_features_str = ' '.join(dynamic_features)

# TODO 定义swanlab
swan_project='tsl'
swan_workspace='ccyy'

args=f"python run.py \
  --task_name {task_name} \
  --is_training 1 \
  --root_path {root_path} \
  --data_path {data_path} \
  --model_id {model_id} \
  --model {model_name} \
  --swan_project {swan_project} \
  --swan_workspace {swan_workspace} \
  --data VitalDB_JSONL \
  --features MS \
  --static_features {static_features_str} \
  --dynamic_features {dynamic_features_str} \
  --freq s \
  --seq_len {seq_len} \
  --label_len {label_len} \
  --pred_len {pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 1 \
  --c_out 1 \
  --embed surgicalF \
  --des Exp \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 64 \
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