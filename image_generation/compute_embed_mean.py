import pickle

import dnnlib
import numpy as np
import torch
from torch_utils import misc
from tqdm import tqdm

TEXT_EMBED_MEAN_LAFITE = "./data/coco/normalized_text_embed_mean_lafite.pkl"
IMAGE_EMBED_MEAN_LAFITE = "./data/coco/normalized_image_embed_mean_lafite.pkl"

TRAIN_DATA_PATH = "./data/lafite/train_set"


def compute_embeds(batch_size=16, num_gpus=1):
    training_set_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset",
        path=TRAIN_DATA_PATH,
        use_labels=False,
        max_size=None,
        xflip=False,
        use_clip=True,
        ratio=1.0,
        remove_mean=False,
        add_noise=False,
        noise_level=0.0,
        add_lafite_noise=False,
    )

    data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=False, num_workers=0
    )  # , prefetch_factor=2)

    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)

    training_set_loader = torch.utils.data.DataLoader(
        dataset=training_set, batch_size=batch_size // num_gpus, **data_loader_kwargs
    )

    img_mean = torch.zeros(512)
    txt_mean = torch.zeros(512)

    count = 0
    print(f"Begin iterating over {len(training_set)} images")
    for real_img, real_c, img_features, txt_features in training_set_loader:
        print("Count:", count)
        img_mean += img_features.sum(dim=0)  # normalized in dataset
        txt_mean += txt_features.sum(dim=0)  # normalized in dataset
        count += img_features.shape[0]

    img_mean /= count
    txt_mean /= count

    with open(TEXT_EMBED_MEAN_LAFITE, "wb") as f:
        pickle.dump(txt_mean, f)

    with open(IMAGE_EMBED_MEAN_LAFITE, "wb") as f:
        pickle.dump(img_mean, f)

    print(f"Dumped at {TEXT_EMBED_MEAN_LAFITE}")
    print(f"Dumped at {IMAGE_EMBED_MEAN_LAFITE}")


if __name__ == "__main__":
    compute_embeds()
