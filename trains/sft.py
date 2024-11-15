from datasets import load_dataset, Dataset
import torch
from transformers import (
    Gemma2ForCausalLM,
    GPT2TokenizerFast,
)
from trl import SFTTrainer, SFTConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)

if __name__ == "__main__":
    max_seq_len = 3096
    model: Gemma2ForCausalLM = Gemma2ForCausalLM.from_pretrained(
        "neody/nemma-100m",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = torch.compile(model, options={"triton.cudagraphs": True}, fullgraph=True)
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("neody/nemma-100m")
    tokenizer.add_tokens(["<|start_of_turn|>", "<|end_of_turn|>"])
    model.resize_token_embeddings(len(tokenizer))
    with open("./template.jinja", "r") as r:
        tokenizer.chat_template = r.read()

    def formatting(example):
        conv = example["conversations"]
        text = tokenizer.apply_chat_template(
            [
                [
                    {
                        "role": "user" if y["from"] == "human" else "assistant",
                        "content": y["value"],
                    }
                    for y in x
                ]
                for x in conv
            ],
            tokenize=False,
        )
        return {"text": text}

    ds: Dataset = load_dataset("BAAI/Infinity-Instruct", "0625", split="train")
    ds = ds.map(formatting, batched=True, remove_columns=ds.column_names, num_proc=6)
    ds_len = len(ds)
    ds = ds.to_iterable_dataset()
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("./ckpt/tokenizer")
    training_args = SFTConfig(
        optim="adamw_8bit",
        output_dir="./ckpt",
        logging_steps=5,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        max_seq_length=max_seq_len,
        max_steps=ds_len // 4,
        bf16=True,
        learning_rate=6e-5,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_num_workers=1,
        dataset_text_field="text",
        num_of_sequences=64,
        packing=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()
