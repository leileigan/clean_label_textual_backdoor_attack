 python attack/train_bert_sst_clean.py \
 --gpu_id 6 \
 --data sst \
 --clean_data_path data/clean_data/sst-2/ \
 --pre_model_path /data/home/ganleilei/bert/bert-base-uncased \
 --save_path /data/home/ganleilei/attack/models/clean_bert_base_tune_sst_mlp1_adam_lr2e-5_bs32_weight0.002/ \
 --mlp_layer_num 1 \
