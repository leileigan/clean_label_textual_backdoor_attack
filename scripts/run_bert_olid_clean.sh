 python attack/train_bert_sst_clean.py \
 --gpu_id 3 \
 --data olid \
 --clean_data_path data/clean_data/offenseval \
 --pre_model_path /home/ganleilei/data/bert/bert-large-uncased \
 --save_path /home/ganleilei/data/attack/models/clean_bert_tune_large_olid_mlp1_adam_lr2e-5_bs32_weight0.002/ \
 --mlp_layer_num 1