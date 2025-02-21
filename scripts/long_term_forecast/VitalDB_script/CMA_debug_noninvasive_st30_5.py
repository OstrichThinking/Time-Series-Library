import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 CMA 模型，对 VitalDB 数据集进行长期预测。
        - 30个点预测5个点
    
    🏠数据集：
        - ioh_dataset_noninvasive_st30_5.csv 
        - 无创组，总计 2065个cases
        - 每隔30s取一个点, 15min预测15min, 滑动窗口步长150s (2.5min)
        - 使用“性别、年龄、BMI、观察窗口时间、无创舒张压、无创平均动脉压、体温、心率、预测窗口时间”预测“无创平均动脉压”
    
    🚀模型：
        - CMA
    
    🔍实验参数：
        - 训练轮数: 50
        - 批次大小: 64
        - 学习率: 0.0001
        - 优化器: Adam
        - 损失函数: MAE
    
    👋实验后台启动命令:
        nohup python -u scripts/long_term_forecast/VitalDB_script/CMA_noninvasive_st30_5.py > checkpoints/output_CMA_vitaldb_noninvasive_st30_5_surgicalF.log 2>&1 &
    
    🌞实验结果:
        - 测试集: 
"""

# 项目根目录
os.chdir("/home/cuiy/project/Time-Series-Library/")

# 设置只使用一张 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# TODO 定义模型名称和数据集路径
model_name = 'CMA'
task_name = 'long_term_forecast'
model_id = f'vitaldb_aaai_noninvasive_st30_5_surgicalF'  

root_path = '/home/share/ioh/VitalDB_IOH/cma_ioh/'
# data_path = 'vitaldb_ioh_dataset_with_medication_invasive_group.csv'
data_path = 'ioh_dataset_noninvasive_st30_5.csv'

# TODO定义IOH需要处理的静态特征和波形数据
static_features = ['caseid', 'sex', 'age', 'bmi']  
dynamic_features = ['window_sample_time',                   # 观察窗口采样时间范围
                    'Solar8000/NIBP_DBP_window_sample',     # 无创舒张压
                    'Solar8000/NIBP_MBP_window_sample',     # 无创平均动脉压
                    'Solar8000/BT_window_sample',           # 体温
                    'Solar8000/HR_window_sample',           # 心率
                    'prediction_window_time',               # 预测窗口时间范围
                    'prediction_maap']                      # 需要预测的有创/无创平均动脉压
# dynamic_features = ['Solar8000/ART_DBP_window_sample', 
#                     'Solar8000/ART_MBP_window_sample',
#                     'Solar8000/ART_SBP_window_sample',
#                     'Solar8000/BT_window_sample',
#                     'Solar8000/HR_window_sample',
#                     'prediction_maap'] 
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
  --data VitalDB \
  --features MS \
  --static_features {static_features_str} \
  --dynamic_features {dynamic_features_str} \
  --freq s \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 10 \
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