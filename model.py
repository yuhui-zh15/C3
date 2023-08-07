from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


class ClipCap(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer_decoder = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
        self.decoder = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self.proj = nn.Linear(1024, self.decoder.config.hidden_size).to(device)

    def forward(self, input_embs: torch.Tensor, texts: List[str]):
        output_prefix = self.proj(input_embs)
        output_ids = self.tokenizer_decoder(
            texts, padding=True, return_tensors="pt"
        ).to(device)
        output_embs = self.decoder.transformer.wte(output_ids["input_ids"])

        decoder_input_embs = torch.cat(
            [
                output_prefix.view(output_prefix.shape[0], 1, output_prefix.shape[1]),
                output_embs,
            ],
            dim=1,
        )
        decoder_labels = torch.cat(
            [
                torch.zeros(
                    (output_prefix.shape[0], 1), dtype=torch.long, device=device
                ),
                output_ids["input_ids"],
            ],
            dim=1,
        )
        decoder_attention_mask = torch.cat(
            [
                torch.ones(
                    (output_prefix.shape[0], 1), dtype=torch.long, device=device
                ),
                output_ids["attention_mask"],
            ],
            dim=1,
        )

        # decoder in: [p] a b c d
        # decoder out: [p] a b c d
        # decoder attn mask: 1 1 1 1 1
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input_embs,
            labels=decoder_labels,
            attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        # generated = decoder_outputs["logits"].argmax(dim=-1)
        # input_text = self.tokenizer_decoder.batch_decode(output_ids["input_ids"], skip_special_tokens=False)
        # print("Input text:", input_text)
        # decoded_text = self.tokenizer_decoder.batch_decode(generated, skip_special_tokens=False)
        # print("Decoded text:", decoded_text)
        return decoder_outputs["loss"], decoder_outputs

    def generate(self, input_embs: torch.Tensor) -> List[str]:
        output_prefix = self.proj(input_embs)
        decoder_input_embs = output_prefix.view(
            output_prefix.shape[0], 1, output_prefix.shape[1]
        )
        generation = generate_beam(
            self.decoder,
            self.tokenizer_decoder,
            embed=decoder_input_embs,
            beam_size=1,
            entry_length=30,
            stop_token="<|endoftext|>",
        )[0]

        return generation


if __name__ == "__main__":
    model = ClipCap()
    texts = ["hello world"]
    input_embs = torch.randn(1, 1024).to(device)
    loss, _ = model(input_embs, texts)
    print(loss)
    generated = model.generate(input_embs)
    print(generated)
