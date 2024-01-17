from transformers import GPT2LMHeadModel


def build_huggingface_model(hf_model_name):
    if hf_model_name == "gpt2":
        return GPT2LMHeadModel.from_pretrained(hf_model_name)
    else:
        raise NotImplementedError(
            f"Huggingface model builder not implemented for {hf_model_name}"
        )
