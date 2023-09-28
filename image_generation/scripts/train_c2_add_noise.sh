python train.py \
    --gpus=1 \
    --resume=./data/lafite/pre-trained-google-cc-best-fid.pkl \
    --outdir=./data/lafite/ckpt/ \
    --data=./data/lafite/train_set \
    --test_data=./data/lafite/val_set \
    --logger_save_dir=./data/logger/ \
    --logger_project=lafite_img_gen_v2 \
    --experiment_name_prefix=stage1_coco_lafite_c2 \
    --temp=0.5 \
    --itd=10 \
    --itc=10 \
    --gamma=10 \
    --mirror=1 \
    --kimg=12000 \
    --mixing_prob=1.0 \
    --add_noise

# Default noise level