#!/bin/bash
python3 inference.py --data_path ./data/NOLA-10K-1mon \
                 --save_path ./models \
                 --encoder_type transformer \
                 --if_normalize True \
                 --max_daily_seq 32 \
                 --batch_size 128 \
                 --device cuda:0 \
                 --input_dim 64 \
                 --hidden_dim 64 \
                 --output_dim 64 \
                 --num_layers 4 \
                 --num_epochs 200 \
                 --lr 5e-3 \
                 --weight_decay_step_size 50 \
                 --weight_decay 0.9 