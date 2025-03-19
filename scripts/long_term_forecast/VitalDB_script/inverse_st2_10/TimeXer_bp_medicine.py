import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 TimeXer 模型，对 VitalDB 数据集进行长期预测。
        - 450个点预测150个点
        - 舒张压、收缩压、平均动脉、
    
    🏠数据集：
        - vitaldb_ioh_dataset_with_medication_invasive_group.csv
        - 有创组，总计 1840 个cases
        - 每隔2s取一个点，15min预测5min，滑动窗口步长20s
    
    🚀模型：
        - TimeXer
    
    🔍训练参数：
        - 训练轮数: 50
        - 批次大小: 64
        - 学习率: 0.0001
    
    👋 实验后台启动命令
        nohup python -u scripts/long_term_forecast/VitalDB_script/inverse_st2_10/TimeXer_bp_medicine.py > checkpoints/TimeXer_bp_medicine.log 2>&1 &
    
    🌞实验结果:
        - 测试集 (V100): 
        波形预测性能比较:
        +--------------------+--------------------+--------------------+
        |        MSE         |        MAE         |        DTW         |
        +--------------------+--------------------+--------------------+
        | 46.136749267578125 | 4.109020233154297  |   Not calculated   |
        +--------------------+--------------------+--------------------+
        分类性能比较:
        +--------------------+--------------------+--------------------+
        |        AUC         |      Accuracy      |       Recall       |
        +--------------------+--------------------+--------------------+
        |      0.78647       |      0.95788       |      0.59022       |
        +--------------------+--------------------+--------------------+
        |     Precision      |    Specificity     |         F1         |
        +--------------------+--------------------+--------------------+
        |      0.69765       |      0.98272       |      0.63946       |
        +--------------------+--------------------+--------------------+
        混淆矩阵:
        +--------------------+--------------------+--------------------+
        |         TP         |         FN         |         --         |
        +--------------------+--------------------+--------------------+
        |        1606        |        1115        |         --         |
        +--------------------+--------------------+--------------------+
        |         FP         |         TN         |         --         |
        +--------------------+--------------------+--------------------+
        |        696         |       39576        |         --         |
        +--------------------+--------------------+--------------------+
          
"""

os.chdir("/home/zhud/fist/ioh/Time-Series-Library/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 定义模型名称和路径
model_name = 'TimeXer'
task_name = 'long_term_forecast'
model_id = 'TimeXer_bp_medicine'


root_path = '/home/share/ioh/VitalDB_IOH/ioh_dataset_with_medication/'
data_path = 'vitaldb_ioh_dataset_with_medication_invasive_group.csv'

seq_len = 450   # 预测窗口数据点数
label_len = 225 # 预测窗口加入label数据的点数
pred_len = 150  # 预测窗口数据点数
stime = 2       # 采样间隔


static_features = ['caseid']
dynamic_features = [
                    'window_sample_time',                   # 观察窗口采样时间范围
                    'prediction_window_time',               # 预测窗口时间范围
                    # 用药（效应部位浓度）
                    'Orchestra/PPF20_CE_window_sample',
                    'Orchestra/RFTN20_CE_window_sample',
                    'Solar8000/ART_DBP_window_sample',
                    'Solar8000/ART_SBP_window_sample',
                    'Solar8000/ART_MBP_window_sample',   # TimeXer内生变量放在最后
                    'prediction_maap'] 
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
  --data VitalDB \
  --features MS \
  --static_features {static_features_str} \
  --dynamic_features {dynamic_features_str} \
  --seq_len {seq_len} \
  --label_len {label_len} \
  --pred_len {pred_len} \
  --stime {stime} \
  --e_layers 3 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
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