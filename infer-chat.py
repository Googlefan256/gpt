from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
import torch
from threading import Thread

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("./ckpt/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "./ckpt/final",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
tokenizer.add_tokens(["<|start_of_turn|>", "<|end_of_turn|>"])
model = torch.compile(
    model, options={"triton.cudagraphs": True}, fullgraph=True, dynamic=True
)
with torch.inference_mode():
    while True:
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": input("Message: ")}],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                return_tensors="pt",
            ).to(device),
            generation_config=GenerationConfig(
                max_new_tokens=2000,
                repetition_penalty=1.2,
                num_beams=1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            ),
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        print("Output: ", end="", flush=True)
        thread.start()
        for text in streamer:
            print(text, end="", flush=True)
        print()
