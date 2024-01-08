# Image Captioning
The code for image captioning using C3 is adapted from [CapDec](https://github.com/DavidHuji/CapDec) and refactored with Pytorch Lightning. Wandb was integrated for logging.

## Getting Started
1. Create conda environment
```
conda env create -f environment.yml
```

2. Preprocess COCO labels
```
cd src/parse_data/
python3 create_labels_json.py
```

3. Embed COCO dataset with CLIP and compute modality means
```
python3 parse_coco.py
python3 compute_embed_means.py
```

4. (Optional) Embed COCO dataset with ImageBind and compute modality means
```
python3 parse_coco_imagebind.py
python3 compute_embed_means_imagebind.py
```

## Training

Training, model, logging and data configurations are provided in `configs`. 

Scripts to run all C3 experiments on COCO using CLIP and ImageBind are provided in `scripts`. We provide the uni-modal text training (`stage1`) and cross-modal image-to-text training (`stage2`) scripts for CLIP in `scripts/coco_scripts` and the uni-modal text training (`stage1`) scripts for ImageBind in `imagebind_coco_scripts`.

To run,
```
bash ./scripts/coco_scripts/stage1/train_unimodal_c3.sh
```