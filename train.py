from transformers import (
    Gemma2ForCausalLM,
    Gemma2Config,
    get_cosine_schedule_with_warmup,
    GPT2TokenizerFast,
    default_data_collator,
)
from bitsandbytes import optim
import torch
from datasets import load_dataset, IterableDataset, DownloadConfig
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)


def train(
    warmup_steps: int,
    train_steps: int,
    max_seq_length: int,
    train_accumulation_steps: int,
    device: str,
    save_steps: int,
):
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )
    config = Gemma2Config(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=max_seq_length,
    )
    config._attn_implementation = "sdpa"
    model = Gemma2ForCausalLM(config)
    model: Gemma2ForCausalLM = torch.compile(
        model.to(device).train(),
        options={"triton.cudagraphs": True},
    )
    print(
        f"Model size: {sum([x.numel() for x in model.parameters()]) * 100 // 1000_000 / 100}M"
    )
    optimizer = optim.AdamW8bit(model.parameters(), lr=1.5e-4, betas=(0.8, 0.99))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    ctx = torch.amp.autocast(
        device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16
    )
    ds: IterableDataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
        download_config=DownloadConfig(resume_download=True),
        name="default",
    )
    ds: ConstantLengthDataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=ds,
        dataset_text_field="text",
        seq_length=max_seq_length,
        num_of_sequences=64,
        infinite=True,
        shuffle=False,
    )
    train_loader = iter(
        DataLoader(ds, batch_size=1, num_workers=4, collate_fn=default_data_collator)
    )
    b = next(train_loader)
    for step in range(1, train_steps + 1):
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
        print(f"Step: {step}, Loss: {step_loss / train_accumulation_steps}")
        if step % save_steps == 0:
            model.save_pretrained("./ckpt")


if __name__ == "__main__":
    train(125000, 1250000, 8192, 4, "cuda:0", 5000)
