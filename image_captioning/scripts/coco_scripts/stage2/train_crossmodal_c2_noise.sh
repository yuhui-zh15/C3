STAGE1_CHECKPOINT=

for seed in 1234 5678 910
do
    # 1% of pre-training data size
    python3 run.py --config configs/finetuning/coco_stage2_image_captioning.yaml --train --test --sample_frac 0.01 --random_seed $seed --normalize_prefix --checkpoint $STAGE1_CHECKPOINT --val_eval --cross_modal_val

    # 5% of pre-training data size
    python3 run.py --config configs/finetuning/coco_stage2_image_captioning.yaml --train --test --sample_frac 0.05 --random_seed $seed --normalize_prefix --checkpoint $STAGE1_CHECKPOINT --val_eval --cross_modal_val

    # 10% of pre-training data size
    python3 run.py --config configs/finetuning/coco_stage2_image_captioning.yaml --train --test --sample_frac 0.1 --random_seed $seed --normalize_prefix --checkpoint $STAGE1_CHECKPOINT --val_eval --cross_modal_val

    # 25% of pre-training data size
    python3 run.py --config configs/finetuning/coco_stage2_image_captioning.yaml --train --test --sample_frac 0.25 --random_seed $seed --normalize_prefix --checkpoint $STAGE1_CHECKPOINT --val_eval --cross_modal_val
done

python3 run.py --config configs/finetuning/coco_stage2_image_captioning.yaml --train --test --normalize_prefix --checkpoint $STAGE1_CHECKPOINT --val_eval --cross_modal_val