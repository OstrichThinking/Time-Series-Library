import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 TimeXer 模型，对 VitalDB 数据集进行长期预测。
        - 15分钟预测5分钟
    
    🏠数据集：
        - AHCformer_invasive_ops2e_st2_10_nosurgicalF_cma
        - (残差+滤波)*2 + 均值填充
        - /home/share/ioh/VitalDB_IOH/timeseries_by_caseids/cma/invasive_ops2e/dataset_vitaldb_cma_invasive_st2_ops2e.jsonl
    
    🚀模型：
        - TimeXer
    
    🔍训练参数：
        - 训练轮数: 50
        - 批次大小: 64
        - 学习率: 0.0001
    
    👋 实验后台启动命令
        nohup python -u scripts/long_term_forecast/VitalDB_script/invasive_ops2e/TimeXer_nosurgicalF_cma_l1.py > checkpoints/invasive_ops2e/TimeXer_nosurgicalF_cma_l1.log 2>&1 &
    
    🌞实验结果:
        - 测试集 (V100): 
          
"""

# os.chdir("/home/zhud/fist/ioh/Time-Series-Library/")
os.chdir(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# 定义模型名称和路径
model_name = 'TimeXer'
task_name = 'long_term_forecast'
model_id = 'TimeXer_nosurgicalF_cma_l1'

root_path = '/home/data/ioh/cma_ioh/invasive_ops2e/'
# root_path = '/home/share/ioh/VitalDB_IOH/timeseries_by_caseids/cma/invasive_ops2e/'
data_path = 'dataset_vitaldb_cma_invasive_st2_ops2e.jsonl'

# 采样间隔 30 s
# 数据集原采样间隔（秒）、实验设置采样间隔（秒）
stime = 2
exp_stime = 30
# 按照15分钟预测5分钟计算预测点数
seq_len = 30   # 预测窗口数据点数
label_len = 5  # 预测窗口加入label数据的点数
pred_len = 10  # 预测窗口数据点数
s_win = 2      # 滑动窗口步长（点）


static_features = ['caseid', 'sex', 'age', 'bmi', 'time']
dynamic_features = [
    'seq_time_stamp_list',
    'pred_time_stamp_list',
    'Solar8000/BT',
    'Solar8000/HR',
    'Solar8000/ART_DBP',
    'Solar8000/ART_MBP', # TimeXer内生变量放在最后
    'prediction_maap'
]
static_features_str = ' '.join(static_features)
dynamic_features_str = ' '.join(dynamic_features)

swan_project='tsl'
swan_workspace='Jude'

args=f"python run.py \
  --task_name long_term_forecast \
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
  --seq_len {seq_len} \
  --label_len {label_len} \
  --pred_len {pred_len} \
  --s_win {s_win} \
  --stime {stime} \
  --exp_stime {exp_stime} \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --embed surgicalF \
  --des Exp \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0005 \
  --loss MAE \
  --batch_size 128 \
  --train_epochs 50 \
  --num_workers 16 \
  --use_multi_gpu \
  --devices 0,1 \
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