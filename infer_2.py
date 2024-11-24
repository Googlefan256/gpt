from trains.train_2_model import GPT, GPTConfig
from transformers import GPT2TokenizerFast, TextIteratorStreamer
import torch
from threading import Thread

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
if __name__ == "__main__":
    model: GPT = torch.compile(
        model, dynamic=True, fullgraph=True, options={"triton.cudagraphs": True}
    )
    with torch.inference_mode():
        while True:
            msg = input("Message: ")
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=False, skip_special_tokens=True
            )
            kwargs = dict(
                idx=tokenizer(
                    msg,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(device),
                max_new_tokens=2000,
                temperature=0.4,
                repeat_penalty=1.2,
                eos=tokenizer.eos_token_id,
                streamer=streamer,
            )
            thread = Thread(target=model.generate, kwargs=kwargs)
            print(f"Output: {msg}", end="", flush=True)
            thread.start()
            for x in streamer:
                print(x, end="", flush=True)
            print("")
