"""
Code adapted from: https://github.com/rmokady/CLIP_prefix_caption/blob/main/parse_coco.py
"""

import json
import os
import pickle

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from tqdm import tqdm

tokenizer = SimpleTokenizer(
    bpe_path=os.getcwd() + "/src/parse_data/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"
)


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


from create_labels_json import DATA_ROOT, MASTER_JSON

# captions -- {caption_id: {caption_raw: .., image_id: ..}}
# embeddings -- {image_id: embedding}

splits = ["train", "val", "test"]

device = torch.device("cuda:0")


def main():
    out_paths = [
        f"{DATA_ROOT}/data/coco/oscar_split_imagebind_{split}.pkl" for split in splits
    ]
    out_paths = dict(zip(splits, out_paths))

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    with open(MASTER_JSON, "r") as f:
        raw_data = json.load(f)["images"]
    print("%0d images loaded from json " % len(raw_data))

    all_images = dict(zip(splits, [{}, {}, {}]))
    all_captions = dict(zip(splits, [{}, {}, {}]))
    for i in tqdm(range(len(raw_data))):
        d = raw_data[i]
        split, filepath, filename = d["split"], d["filepath"], d["filename"]

        if split == "restval":
            split = "train"

        # Get and save image and image embed
        img_id = d["imgid"]
        filename = f"{DATA_ROOT}/data/coco/{filepath}/{filename}"
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([filename], device)
        }
        with torch.no_grad():
            image_embed = model(inputs)[ModalityType.VISION]

        ## Note: changes for all the keys!
        all_images[split][img_id] = {
            "img_path": filename,
            "embed": image_embed[0].cpu(),
        }

        # Get + save caption and caption embed
        for caption_data in d["sentences"]:
            assert img_id == caption_data["imgid"]
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(
                    [caption_data["raw"]], device
                )
            }
            with torch.no_grad():
                text_embed = model(inputs)[ModalityType.TEXT]

            sent_id = caption_data["sentid"]

            ## Note: changes for all the keys!!
            all_captions[split][sent_id] = {
                "caption": caption_data["raw"],
                "img_id": img_id,
                "embed": text_embed[0].cpu(),
            }

        if (i + 1) % 10000 == 0:
            for split in splits:
                with open(out_paths[split], "wb") as f:
                    pickle.dump(
                        {"images": all_images[split], "captions": all_captions[split]},
                        f,
                    )

            print_totals(all_images, all_captions)

    for split in splits:
        with open(out_paths[split], "wb") as f:
            pickle.dump(
                {"images": all_images[split], "captions": all_captions[split]}, f
            )

    print("Done")
    print_totals(all_images, all_captions)
    return 0


def print_totals(all_images, all_captions):
    print("Done")
    print("Total number of images (so far)")
    embed_totals = [len(all_images[split].values()) for split in splits]
    print(dict(zip(splits, embed_totals)))
    print("Total number of captions (so far)")
    caption_totals = [len(all_captions[split].values()) for split in splits]
    print(dict(zip(splits, caption_totals)))


if __name__ == "__main__":
    exit(main())
