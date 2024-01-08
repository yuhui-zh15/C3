# Image Generation
The code for image generation using C3 is minimally adapted from [Lafite](https://github.com/drboog/Lafite). Wandb was integrated for logging. 

## Getting Started
1. Create conda environment
```
conda env create -f environment.yml
```

2. Compute embedding means
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