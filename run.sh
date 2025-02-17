# ----------------------------------------- 训练模型 TimeXer ------------------------------------------------------
# 训练模型AAAI变量
nohup ./scripts/long_term_forecast/VitalDB_script/TimeXer.sh >./checkpoints/timexer_train_on_vitaldb_aaai.log 2>&1 &

# 训练模型AAAI变量+用药
nohup ./scripts/long_term_forecast/VitalDB_script/TimeXer.sh >./checkpoints/timexer_train_on_vitaldb_aaai_with_medicine.log 2>&1 &

# 训练模型AAAI变量+用药+呼吸相关
nohup ./scripts/long_term_forecast/VitalDB_script/TimeXer.sh >./checkpoints/timexer_train_on_vitaldb_aaai_with_respiratory.log 2>&1 &



# ----------------------------------------- 训练模型 iTransformer ------------------------------------------------------
# 训练模型iTransformer变量
nohup ./scripts/long_term_forecast/VitalDB_script/iTransformer.sh >./checkpoints/itransformer_train_on_vitaldb_aaai.log 2>&1 &