from trains.train_2_model import GPT, GPTConfig
from transformers import (
    GPT2TokenizerFast,
)
import torch

device = "cuda:0"
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
model.to(device, dtype=torch.bfloat16)
model.load_state_dict(torch.load("./ckpt.pt", weights_only=True))
model = torch.compile(
    model, options={"triton.cudagraphs": True}, fullgraph=True, dynamic=True
)
with torch.inference_mode():
    while True:
        ids = model.generate(
            tokenizer(
                input("Message: "), return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device),
            200,
            0.4,
            repeat_penalty=1.2,
        ).squeeze()
        print(f"Output: {tokenizer.decode(ids)}")
