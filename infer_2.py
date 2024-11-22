from trains.train_2_model import GPT, GPTConfig
from transformers import (
    GPT2TokenizerFast,
)
import torch

tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
    "openai-community/gpt2"
)
model = GPT(
    GPTConfig(
        vocab_size=len(tokenizer),
        n_layer=32,
        n_head=3,
        n_embd=384,
    )
)
model.load_state_dict(torch.load("./ckpt.pt", weights_only=True))
ids = model.generate(
    tokenizer(
        "Hello, my name is", return_tensors="pt", add_special_tokens=True
    ).input_ids,
    20,
    0.4,
).squeeze()
print(ids, tokenizer.decode(ids))
