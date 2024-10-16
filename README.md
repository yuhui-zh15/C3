# C3 

## Usage

### Data Processing: 

`imagebind.ipynb`

### Train: 

`python train.py --method="c1" --file_name="data/data_image_coco_imagebind.pkl"`

`python train.py --method="c21" --file_name="data/data_image_coco_imagebind.pkl"`

`python train.py --method="c22" --file_name="data/data_image_coco_imagebind.pkl"`

`python train.py --method="c3" --file_name="data/data_image_coco_imagebind.pkl"`

`python train.py --method="c1" --file_name="data/data_image_coco_clip.pkl"`

`python train.py --method="c21" --file_name="data/data_image_coco_clip.pkl"`

`python train.py --method="c22" --file_name="data/data_image_coco_clip.pkl"`

`python train.py --method="c3" --file_name="data/data_image_coco_clip.pkl"`

> When running CLIP, change "self.proj = nn.Linear(1024, self.decoder.config.hidden_size).to(device)" in model.py from 1024 to 512 (CLIP VIT-B/32)

### Eval: 

`utils.ipynb`

## Processed Data

https://drive.google.com/drive/folders/1M9rXIKHPz58YJJ1l-c9zx-tfeCLXLti7?usp=sharing
