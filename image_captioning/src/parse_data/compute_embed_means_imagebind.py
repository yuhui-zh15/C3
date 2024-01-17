import pickle

import torch

TEXT_EMBED_MEAN = "./data/coco/imagebind_normalized_text_embed_mean.pkl"
IMAGE_EMBED_MEAN = "./data/coco/imagebind_normalized_image_embed_mean.pkl"


def main():
    data_path = "./data/coco/oscar_split_imagebind_train.pkl"

    print(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)

    # {image_id: {"img_path": ..., "embed": ...}}
    images = all_data["images"]
    # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
    captions = all_data["captions"]

    text_mean = torch.zeros(1, 1024)
    img_mean = torch.zeros(1, 1024)

    # Compute text mean
    for cap_id in captions:
        cap_embed = captions[cap_id]["embed"]
        text_mean += cap_embed / cap_embed.norm()

    text_mean = text_mean / len(captions)

    # Compute image mean
    for img_id in images:
        img_embed = images[img_id]["embed"]
        img_mean += img_embed / img_embed.norm()

    img_mean = img_mean / len(images)

    with open(TEXT_EMBED_MEAN, "wb") as f:
        pickle.dump(text_mean, f)

    with open(IMAGE_EMBED_MEAN, "wb") as f:
        pickle.dump(img_mean, f)

    print(f"Saved text_embed_mean to {TEXT_EMBED_MEAN}")
    print(f"Saved img_embed_mean to {IMAGE_EMBED_MEAN}")


if __name__ == "__main__":
    main()
