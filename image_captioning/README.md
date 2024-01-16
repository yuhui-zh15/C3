# Image Captioning
The code for image captioning using C3 is adapted from [CapDec](https://github.com/DavidHuji/CapDec) and refactored with Pytorch Lightning. Wandb was integrated for logging.

## Datasets
Download the MSCOCO dataset from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). 

## Getting Started

### Prepare Environment
Create conda environment
```
conda env create -f environment.yml
```

Note: if using Imagebind, follow the [official repo](https://github.com/facebookresearch/ImageBind) to create a separate imagebind conda environment.

### Prepare Labels and Embeddings
1. Preprocess COCO labels
```
python3 src/parse_data/create_labels_json.py
```

2. Embed COCO dataset with CLIP and compute modality means
```
python3 src/parse_data/parse_coco.py
python3 src/parse_data/compute_embed_means.py
```

3. (Optional) Embed COCO dataset with ImageBind and compute modality means
```
conda activate imagebind
python3 src/parse_data/parse_coco_imagebind.py
python3 src/parse_data/compute_embed_means_imagebind.py
conda deactivate imagebind
```

## Training

Training, model, logging and data configurations are provided in `configs`. 

Scripts to run all C3 experiments on COCO using CLIP and ImageBind are provided in `scripts`. We provide the uni-modal text training (`stage1`) and cross-modal image-to-text training (`stage2`) scripts for CLIP in `scripts/coco_scripts` and the uni-modal text training (`stage1`) scripts for ImageBind in `imagebind_coco_scripts`.

To run,
```
bash ./scripts/coco_scripts/stage1/train_unimodal_c3.sh
```