from datasets import load_dataset, Dataset
import torch
from transformers import (
    Gemma2ForCausalLM,
    GPT2TokenizerFast,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)
from bitsandbytes import optim
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)


def main(
    bsz: int,
    train_accumulation_steps: int,
    save_steps: int,
    max_seq_len: int,
    device: str,
    warmup_ratio: int,
):
    model: Gemma2ForCausalLM = Gemma2ForCausalLM.from_pretrained(
        "neody/nemma-100m",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
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
    ds = ds.map(formatting, batched=True, remove_columns=ds.column_names, num_proc=20)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("./ckpt/tokenizer")
    ds: ConstantLengthDataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=ds,
        dataset_text_field="text",
        seq_length=max_seq_len,
        num_of_sequences=64,
        infinite=True,
        shuffle=False,
    )
    train_loader = iter(
        DataLoader(ds, batch_size=bsz, num_workers=20, collate_fn=default_data_collator)
    )
    train_steps = len(ds) * 3
    b = next(train_loader)
    optimizer = optim.AdamW8bit(model.parameters(), lr=1.5e-4, betas=(0.8, 0.99))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_steps * warmup_ratio // 1, train_steps
    )
    ctx = torch.amp.autocast(
        device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16
    )
    pbar = tqdm(range(1, train_steps + 1))
    for step in pbar:
        step_loss = 0
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                loss = model(
                    input_ids=b["input_ids"].to(device),
                    labels=b["labels"].to(device),
                    return_dict=True,
                ).loss
                step_loss += loss.detach().item()
            # advance the dataset for the next batch
            b = next(train_loader)
            # backward pass
            loss.backward()  # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_description(
            f"Step: {step}, Loss: {step_loss / train_accumulation_steps}"
        )
        if step % save_steps == 0:
            model.save_pretrained("./ckpt")


if __name__ == "__main__":
    main(6, 8, 5000, 3096, "cuda", 0.05)
