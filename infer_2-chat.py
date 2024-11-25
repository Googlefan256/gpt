from trains.train_2_model import GPT, GPTConfig, CastedLinear
from transformers import GPT2TokenizerFast, TextIteratorStreamer
import torch
from huggingface_hub import hf_hub_download
from threading import Thread

device = "cuda:0"
tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
    "neody/npt-100m-it", subfolder="tokenizer"
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
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
model.load_state_dict(
    torch.load(hf_hub_download("neody/npt-100m-it", "ckpt.pt"), weights_only=True)
)
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
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": msg},
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(device),
                max_new_tokens=2000,
                temperature=0.4,
                repeat_penalty=1.2,
                eos=tokenizer.convert_tokens_to_ids("<|end_of_turn|>"),
                streamer=streamer,
            )
            thread = Thread(target=model.generate, kwargs=kwargs)
            print(f"Output: ", end="", flush=True)
            thread.start()
            for x in streamer:
                print(x, end="", flush=True)
            print("")
