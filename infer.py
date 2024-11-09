from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
import torch
from threading import Thread

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("neody/nemma-100m")
model = AutoModelForCausalLM.from_pretrained(
    "neody/nemma-100m",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model = torch.compile(
    model, options={"triton.cudagraphs": True}, fullgraph=True, dynamic=True
)
with torch.inference_mode():
    while True:
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=False, skip_special_tokens=True
        )
        kwargs = dict(
            **tokenizer(
                input("Message: "),
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
