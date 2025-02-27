import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 iTransformer 模型，对 VitalDB 数据集进行长期预测。
        - 450个点预测150个点
    
    🏠数据集：
        - vitaldb_ioh_dataset_with_medication_invasive_group.csv
        - 有创组，总计 1840 个cases
        - 每隔2s取一个点，15min预测5min，滑动窗口步长20s
    
    🚀模型：
        - iTransformer
    
    🔍训练参数：
        - 训练轮数: 50
        - 批次大小: 64
        - 学习率: 0.0001
    
    👋 实验后台启动命令
        nohup python -u scripts/long_term_forecast/VitalDB_script/iTransformer_invasive_st2_10_surgicalF_d128.py > checkpoints/iTransformer_invasive_st2_10_surgicalF_d128.log 2>&1 &
    
    🌞实验结果:
        - 测试集 (V100): 
        mse:48.724021911621094, mae:4.210161209106445, dtw:Not calculated
        precision:0.8908242612752721, recall:0.41615809357744843, F1:0.567297216995147, accuracy:0.8983788058521155, specificity:0.9902799700922157, auc:0.703219031834832

        - d_model=256, d_ff=256
        mse:47.92493438720703, mae:4.174158573150635, dtw:Not calculated
        precision:0.9081346423562412, recall:0.3763440860215054, F1:0.5321553318265873, accuracy:0.8940757797781034, specificity:0.9927445930602863, auc:0.6845443395408959

        - d_model=128, d_ff=128
        mse:47.82950973510742, mae:4.154483318328857, dtw:Not calculated
        precision:0.9111876075731498, recall:0.3846265620459169, F1:0.540921630734648, accuracy:0.8954946154025074, specificity:0.992855362631885, auc:0.6887409623389009

            
"""

os.chdir("/home/zhud/fist/ioh/Time-Series-Library/")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 定义模型名称和路径
model_name = 'iTransformer'
task_name = 'long_term_forecast'
model_id = 'iTransformer_invasive_st2_10_surgicalF_d128'

root_path = '/home/share/ioh/VitalDB_IOH/ioh_dataset_with_medication/'
data_path = 'vitaldb_ioh_dataset_with_medication_invasive_group.csv'

seq_len = 450   # 预测窗口数据点数
label_len = 225 # 预测窗口加入label数据的点数
pred_len = 150  # 预测窗口数据点数
stime = 20      # 采样间隔

static_features = ['caseid', 'sex', 'age', 'bmi']  
dynamic_features = [
                    'window_sample_time',                   # 观察窗口采样时间范围
                    'Solar8000/ART_DBP_window_sample',
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
                    'prediction_window_time',               # 预测窗口时间范围
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
  --freq s \
  --seq_len {seq_len} \
  --label_len {label_len} \
  --pred_len {pred_len} \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 23 \
  --dec_in 23 \
  --c_out 1 \
  --embed surgicalF \
  --use_embed \
  --des Exp \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --train_epochs 50 \
  --num_workers 10 \
  --batch_size 64 \
  --use_multi_gpu \
  --devices 0,1,2,3 \
  --inverse"           # 测试时是否恢复


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