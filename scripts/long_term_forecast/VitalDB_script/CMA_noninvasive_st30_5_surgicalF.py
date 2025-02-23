import os
import runpy
import sys

"""
    🌟实验简述：
        - 使用 CMA 模型，对 VitalDB 数据集进行长期预测。
        - 30个点预测10个点 (15min预测5min)
    
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
        nohup python -u scripts/long_term_forecast/VitalDB_script/CMA_noninvasive_st30_5_surgicalF.py > checkpoints/output_CMA_noninvasive_st30_5_surgicalF.log 2>&1 &
    
    🌞实验结果:
        - 测试集 (A100, 使用'CMA_v1_64.97'模型, dmodel=512):
        mse:63.297428131103516, 
        mae:5.388845920562744
        
        precision:0.8779840848806366, 
        recall:0.48063891577928364, 
        F1:0.6212073819205505, 
        accuracy:0.9255135933079099, 
        specificity:0.9902762119503946, 
        auc:0.7354575638648391
        
        - 测试集 (A100, 使用'CMA_v1_64.97'模型, dmodel=512, batchsize=256):
        mse:64.08560943603516, mae:5.450108528137207
        precision:0.8599527930763179, 
        
        recall:0.5290416263310745, 
        F1:0.6550794126460894, 
        accuracy:0.929204084143191, 
        specificity:0.9874577226606539, 
        auc:0.7582496744958641
        
        - 测试集 (A100, 使用'CMA_v1_64.97'模型, dmodel=512, lastnorm (CMA_lastnormalbad.py)):
        mse:278.8521728515625, mae:12.1730375289917, dtw:Not calculated
        precision:0.00625, recall:0.000484027105517909, F1:0.0008984725965858041, accuracy:0.8632058063722475, specificity:0.9887965050732808, auc:0.49464026608939937

        - ！！测试集 (A100, maby),这是无时间编码的，先写在这里
        mse:67.97599029541016, mae:5.695897579193115, dtw:Not calculated
        precision:0.8495575221238938, recall:0.46466602129719264, F1:0.6007509386733416, accuracy:0.9215155615696887, specificity:0.988021420518602, auc:0.7263437209078973
        
        - ！！测试集 (A100, maby),这是时间编码的，先写在这里
        mse:71.73130798339844, mae:5.937591552734375, dtw:Not calculated
        precision:0.8377281947261663, recall:0.3998063891577928, F1:0.5412844036697247, accuracy:0.9138885471767745, specificity:0.9887260428410372, auc:0.694266215999415

"""

# TODO 修改项目路径
# A100项目路径
os.chdir("/home/temporal/cuiy/Time-Series-Library/")
# # V100项目路径
# os.chdir("/home/cuiy/project/Time-Series-Library/")

# TODO 修改需要使用的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# TODO 修改模型名称和数据集路径
model_name = 'CMA'
task_name = 'long_term_forecast'
model_id = 'vitaldb_noninvasive_st30_5_surgicalF'  

# A100数据集路径
root_path = '/home/data/ioh/cma_ioh/'
# # V100数据集路径
# root_path = '/home/share/ioh/VitalDB_IOH/cma_ioh/'

data_path = 'ioh_dataset_noninvasive_st30_5.csv'

# TODO 修改数据集采样窗口、预测窗口情况以及采样频率
seq_len = 30    # 预测窗口数据点数
label_len = 15  # 预测窗口加入label数据的点数
pred_len = 10   # 预测窗口数据点数
stime = 30      # 采样频率

# TODO修改IOH需要处理的静态特征和波形数据
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

# TODO 修改swanlab
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
  --seq_len {seq_len} \
  --label_len {label_len} \
  --pred_len {pred_len} \
  --stime {stime} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 64 \
  --embed surgicalF \
  --use_embed \
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