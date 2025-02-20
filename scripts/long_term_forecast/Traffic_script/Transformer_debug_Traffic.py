import os
import runpy
import sys

# nohup python -u scripts/long_term_forecast/VitalDB_script/TimeXer_debug_aaai_noninvasive_st30_5.py > output.log 2>&1 &
os.chdir("/home/cuiy/project/Time-Series-Library/")

# 设置只使用一张 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 定义模型名称和路径
model_name = 'Transformer'
root_path = '/home/cuiy/project/Time-Series-Library/dataset/traffic/'
data_path = 'traffic.csv'

# # args = 'python -m src.test'
# # args = 'python -m src.dataloader

args=f"python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path {root_path} \
  --data_path {data_path} \
  --model_id traffic_96_192 \
  --model {model_name} \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des Exp \
  --itr 1 \
  --train_epochs 3"


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