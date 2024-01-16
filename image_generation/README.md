# Image Generation
The code for image generation using C3 is minimally adapted from [Lafite](https://github.com/drboog/Lafite). Wandb was integrated for logging. 

## Preparing Datasets
Please refer to the instructions to prepare MSCOCO from the [Lafite official repo](https://github.com/drboog/Lafite?tab=readme-ov-file#preparing-datasets). We directly use their preprocessed [training](https://drive.google.com/file/d/1b82BCh65XxwR-TiA8zu__wwiEHLCgrw2/view?usp=sharing) and [validation](https://drive.google.com/file/d/1b82BCh65XxwR-TiA8zu__wwiEHLCgrw2/view?usp=sharing) sets.

## Pre-Trained Model
For all our experiments, we finetune [Lafite pre-trained on Google CC3M](https://drive.google.com/file/d/17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq/view?usp=sharing).

## Getting Started

### Prepare Environment
Create conda environment
```
conda env create -f environment.yml
```

### Prepare Embeddings
Compute embedding means
```
python3 compute_embed_mean.py
```

## Training
Scripts for C1, C21, C22, C3 and Lafite (baseline) have been provided in `scripts`.

To run C3,
```
bash ./scripts/train_c3.sh
```

## Generation
Images can be generated using the notebook `generate.ipynb`.