from xlstm import (
    XLSTMLMModel,
    XLSTMLMModelConfig,
    MLSTMBlockConfig,
    SLSTMBlockConfig,
    MLSTMLayerConfig,
    SLSTMLayerConfig,
    FeedForwardConfig,
)
import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    GPT2TokenizerFast,
    default_data_collator,
)
from bitsandbytes import optim
import torch
from torch.nn import functional as F
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
    bsz: int,
    train_accumulation_steps: int,
    device: str,
    save_steps: int,
):
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )
    model = XLSTMLMModel(
        XLSTMLMModelConfig(
            mlstm_block=MLSTMBlockConfig(
                mlstm=MLSTMLayerConfig(
                    num_heads=4, qkv_proj_blocksize=4, conv1d_kernel_size=4
                )
            ),
            slstm_block=SLSTMBlockConfig(
                slstm=SLSTMLayerConfig(num_heads=4, num_gates=4, num_states=4),
                feedforward=FeedForwardConfig(),
            ),
            context_length=3072,
            num_blocks=32,
            embedding_dim=768,
            vocab_size=len(tokenizer),
            slstm_at=[1, 15, 27, 39],
            tie_weights=True,
            dropout=0.1,
        )
    ).to(device, torch.bfloat16)
    print(
        f"Model size: {sum([x.numel() for x in model.parameters()]) * 100 // 1000_000 / 100}M"
    )
    optimizer = optim.AdamW8bit(model.parameters(), lr=1.5e-4, betas=(0.8, 0.99))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    ctx = torch.amp.autocast(
        device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16
    )
    ds: IterableDataset = load_dataset(
        "Zyphra/Zyda-2",
        split="train",
        name="zyda_crossdeduped-filtered",
        streaming=True,
        download_config=DownloadConfig(resume_download=True),
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
        DataLoader(ds, batch_size=bsz, num_workers=4, collate_fn=default_data_collator)
    )
    print("Created train loader")
    b = next(train_loader)
    for step in range(1, train_steps + 1):
        step_loss = 0
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                labels = b["labels"].to(device)
                outputs = model(
                    b["input_ids"].to(device),
                )
                loss = F.cross_entropy(
                    outputs.view(-1, model.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
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
            torch.save(model.state_dict(), "./ckpt")


if __name__ == "__main__":
    train(5000, 500000, 3072, 6, 8, "cuda:0", 5000)
