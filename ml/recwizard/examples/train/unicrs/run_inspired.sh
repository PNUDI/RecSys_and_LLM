# INSPIRED
 CUDA_VISIBLE_DEVICES=1 python examples/train/unicrs/pretrain.py --output_dir save/unicrs_pre/inspired --dataset inspired_unicrs --num_warmup_steps 168 --learning_rate 1e-4
CUDA_VISIBLE_DEVICES=1 python examples/train/unicrs/train_conv.py --pretrained_model save/unicrs_pre/inspired/model_best --dataset inspired_unicrs --run_infer --num_train_epochs 10 --output_dir save/unicrs_conv/inspired --num_warmup_steps 976
CUDA_VISIBLE_DEVICES=1 python examples/train/unicrs/train_rec.py --pretrained_model save/unicrs_pre/inspired/model_best --dataset inspired_unicrs --num_train_epochs 10 --output_dir save/unicrs_rec/inspired --num_warmup_steps 33 --gen_data 'save/unicrs_conv/inspired/{}_gen.jsonl'