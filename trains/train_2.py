import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    GPT2TokenizerFast,
    default_data_collator,
)
from bitsandbytes import optim
import torch
from datasets import load_dataset, IterableDataset, DownloadConfig
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader
from .train_2_model import GPT, GPTConfig
import torch._inductor.config as config
from .train_2_optimizer import Muon

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)


def train(
    warmup_steps: int,
    train_steps: int,
    max_seq_length: int,
    bsz: int,
    train_accumulation_steps: int,
    device: str,
    save_steps: int,
):
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )
    model = (
        GPT(
            GPTConfig(
                vocab_size=len(tokenizer),
                n_layer=32,
                n_head=3,
                n_embd=384,
            )
        )
        .to(device, torch.bfloat16)
        .train()
    )
    print(
        f"Model size: {sum([x.numel() for x in model.parameters()]) * 100 // 1000_000 / 100}M"
    )
    raw_model = model
    model = torch.compile(model)
    optimizer1 = optim.AdamW8bit(
        [model.transformer.wte.weight], lr=5e-4, betas=(0.9, 0.95)
    )
    optimizer2 = optim.AdamW8bit(
        [raw_model.lm_head.weight], lr=3.5e-4, betas=(0.9, 0.95)
    )
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = Muon(matrix_params, lr=3.5e-4, momentum=0.95)
    optimizer4 = optim.AdamW8bit(
        scalar_params, lr=3.5e-4, betas=(0.9, 0.95)
    )  # note that this learning rate is neither sensitive nor tuned
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    schedulers = [
        get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
        for optimizer in optimizers
    ]
    ds: IterableDataset = load_dataset(
        "Zyphra/Zyda-2",
        split="train",
        name="default",
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
            logits, loss = model(
                b["input_ids"].to(device),
                b["labels"].to(device),
            )
            step_loss += loss.detach().item()
            # advance the dataset for the next batch
            b = next(train_loader)
            # backward pass
            loss.backward()  # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        frac = min(step / warmup_steps, 1)
        optimizer1.param_groups[0]["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)
        print(f"Step: {step}, Loss: {step_loss / train_accumulation_steps}")
        if step % save_steps == 0:
            torch.save(raw_model.state_dict(), "ckpt.pt")
    torch.save(raw_model.state_dict(), "ckpt.pt")


if __name__ == "__main__":
    train(5000, 500000, 4096, 4, 16, "cuda:0", 500)
