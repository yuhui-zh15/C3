import pickle
import json
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import wandb
from model import ClipCap
import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(file_name: str):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    train_data = [item for item in data if item["split"] == "train"]
    valid_data = [item for item in data if item["split"] == "test"]
    x_avg = np.mean([item["x_embed"] for item in train_data], axis=0)
    y_avg = np.mean([item["y_embed"] for item in train_data], axis=0)
    avg = {
        "x_embed": torch.from_numpy(x_avg).to(device),
        "y_embed": torch.from_numpy(y_avg).to(device),
    }
    return train_data, valid_data, avg


def train(model, train_data_, optimizer, avg, method, noise_scale, batch_size=32):
    model.train()

    idxs = list(range(len(train_data_)))
    random.shuffle(idxs)
    train_data = [train_data_[i] for i in idxs]

    input_key = "y_embed"
    for i in trange(0, len(train_data), batch_size):
        items = train_data[i : i + batch_size]
        texts = [item["y"] for item in items]
        input_embs = torch.stack(
            [torch.from_numpy(item[input_key]) for item in items]
        ).to(device)
        if method in ["c21", "c3"]:
            input_embs = input_embs - avg[input_key]
        if method in ["c22", "c22_modified", "c3"]:
            gaussian_noise = torch.randn(input_embs.shape).to(device)
            if method in ["c22_modified"]:
                modality_gap = avg["x_embed"] - avg["y_embed"]
                normed_gap = F.normalize(modality_gap, p=2, dim=-1)
                proj_length = gaussian_noise @ normed_gap.view(-1, 1)
                proj_vec = proj_length * normed_gap
                gaussian_noise = gaussian_noise - proj_vec
                # print(modality_gap.shape, normed_gap.shape, proj_length.shape, proj_vec.shape, gaussian_noise.shape)
            input_embs = input_embs + gaussian_noise * noise_scale
        loss, _ = model(input_embs, texts)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"train/loss": loss.item()})


def evaluate(model, valid_data, avg, method, batch_size=32, input_key="y_embed"):
    model.eval()

    losses = []
    for i in trange(0, len(valid_data), batch_size):
        items = valid_data[i : i + batch_size]
        texts = [item["y"] for item in items]
        input_embs = torch.stack(
            [torch.from_numpy(item[input_key]) for item in items]
        ).to(device)
        if method in ["c21", "c3"]:
            input_embs = input_embs - avg[input_key]
        with torch.no_grad():
            loss, _ = model(input_embs, texts)

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    wandb.log({f"val/loss_{input_key[0]}": avg_loss})
    return avg_loss


def evaluate_gen(model, valid_data, avg, method, input_key="y_embed"):
    model.eval()

    generations = []
    for item in tqdm(valid_data):
        input_embs = torch.from_numpy(item[input_key]).unsqueeze(0).to(device)
        if method in ["c21", "c3"]:
            input_embs = input_embs - avg[input_key]
        generation = model.generate(input_embs)
        generations.append(generation)

    originals = [item["y"] for item in valid_data]

    return generations, originals

    # df = pd.DataFrame({"original": originals, "generated": generations})
    # tbl = wandb.Table(data=df)
    # acc = np.mean(
    #     [
    #         text.split(".")[0].strip() == generation.split(".")[0].strip()
    #         for text, generation in zip(originals, generations)
    #     ]
    # )
    # wandb.log({"val/acc": acc, "val/tbl": tbl})
    # return acc


@click.command()
@click.option("--training_percentage", type=float, default=1.0)
@click.option("--n_epochs", type=int, default=20)
@click.option("--batch_size", type=int, default=64)
@click.option("--lr", type=float, default=0.001)
@click.option("--eval_freq", type=int, default=1)
@click.option("--file_name", type=str, default="data_audio_clotho_imagebind.pkl")
@click.option("--method", type=str, default="c1")
@click.option("--noise_scale", type=float, default=0.1)
def main(
    training_percentage: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    eval_freq: int,
    file_name: str,
    method: str,
    noise_scale: float,
):
    wandb.init(project="c3")

    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    train_data, valid_data, avg = load_data(f"data/{file_name}")
    print(f"Length of train data: {len(train_data)}, valid data: {len(valid_data)}")

    model = ClipCap()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(1, n_epochs + 1):
        train(model, train_data, optimizer, avg, method, noise_scale, batch_size)
        _ = evaluate(
            model, valid_data, avg, method, batch_size, input_key="y_embed"
        )
        _ = evaluate(
            model, valid_data, avg, method, batch_size, input_key="x_embed"
        )
        if epoch % eval_freq == 0:
            REPEAT_TIMES = 5  # TODO: special for dataset
            generations, originals = evaluate_gen(model, valid_data[::REPEAT_TIMES], avg, method, input_key="x_embed")
            json.dump(list(zip([generations, originals])), open(f"cache/generation_{file_name}_{method}_{noise_scale}_{epoch}.json", "w"), indent=2)

        torch.save(model.state_dict(), f"model_{file_name}_{method}_{noise_scale}.pt")


# @click.command()
# @click.option("--model_path", type=str, default="")
# @click.option("--file_name", type=str, default="data_audio_clotho_imagebind.pkl")
# def infer(model_path: str, file_name: str):
#     train_data, valid_data = load_data(file_name)
#     print(f"Length of valid data: {len(valid_data)}")
#     print(random.sample(valid_data, 5))

#     model = ClipCap()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     for item in valid_data:
#         input_embs = torch.from_numpy(item["y_embed"]).unsqueeze(0).to(device)
#         generation = model.generate(input_embs)
#         print(f"Original: {item['y']}")
#         print(f"Generated: {generation}")
#         print()


if __name__ == "__main__":
    # train_data, valid_data, _ = load_data("data_audio_clotho_imagebind.pkl")
    # print(valid_data[:20])

    main()
    # infer()
