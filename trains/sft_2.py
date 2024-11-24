import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    GPT2TokenizerFast,
    default_data_collator,
)
from torch import nn
from bitsandbytes import optim
from datasets import load_dataset, Dataset
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader
from .train_2_model import GPT, CastedLinear, GPTConfig
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
    w = torch.load("./ckpt.pt", weights_only=True)
    tokenizer.add_tokens(
        ["<|start_of_turn|>", "<|end_of_turn|>", "<|user|>", "<|assistant|>"]
    )
    new_tokens = ["<|start_of_turn|>", "<|end_of_turn|>", "<|user|>", "<|assistant|>"]
    # Get the original embedding and LM head weights
    orig_embed_weights = w["transformer.wte.weight"]
    orig_lm_head_weights = w["lm_head.weight"]
    tokenizer.add_tokens(new_tokens)
    # Create new embedding weights with expanded size
    new_vocab_size = len(tokenizer)
    embed_dim = orig_embed_weights.shape[1]
    # Initialize new weights tensors with the expanded size
    new_embed_weights = torch.empty(
        new_vocab_size,
        embed_dim,
        dtype=orig_embed_weights.dtype,
        device=orig_embed_weights.device,
    )
    new_lm_head_weights = torch.empty(
        new_vocab_size,
        embed_dim,
        dtype=orig_lm_head_weights.dtype,
        device=orig_lm_head_weights.device,
    )
    # Copy original weights
    new_embed_weights[: orig_embed_weights.shape[0]] = orig_embed_weights
    new_lm_head_weights[: orig_lm_head_weights.shape[0]] = orig_lm_head_weights
    # Initialize new token embeddings using statistics of existing embeddings
    with torch.inference_mode():
        embed_mean = orig_embed_weights.mean(dim=0)
        embed_std = orig_embed_weights.std(dim=0)
        for i in range(orig_embed_weights.shape[0], new_vocab_size):
            new_embed_weights[i] = torch.normal(embed_mean, embed_std)
            new_lm_head_weights[i] = torch.normal(embed_mean, embed_std)
    w["transformer.wte.weight"] = new_embed_weights
    w["lm_head.weight"] = new_lm_head_weights
    model.transformer.wte = nn.Embedding(new_vocab_size, embed_dim)
    model.lm_head = nn.Linear(embed_dim, new_vocab_size, bias=False)
    model.load_state_dict(w)
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    print(
        f"Model size: {sum([x.numel() for x in model.parameters()]) * 100 // 1000_000 / 100}M"
    )
    raw_model = model
    model = torch.compile(model)
    optimizer1 = optim.AdamW8bit(
        [model.transformer.wte.weight], lr=1e-4, betas=(0.9, 0.95)
    )
    optimizer2 = optim.AdamW8bit([raw_model.lm_head.weight], lr=1e-4, betas=(0.9, 0.95))
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = Muon(matrix_params, lr=1e-4, momentum=0.95)
    optimizer4 = optim.AdamW8bit(
        scalar_params, lr=1e-4, betas=(0.9, 0.95)
    )  # note that this learning rate is neither sensitive nor tuned
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    schedulers = [
        get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
        for optimizer in optimizers
    ]
    ctx = torch.amp.autocast(
        device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16
    )
    ds: Dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")

    def formatting(example):
        text = tokenizer.apply_chat_template(
            [
                [
                    {
                        "role": "user" if x["from"] == "human" else "assistant",
                        "content": x["value"],
                    }
                    for x in x
                ]
                for x in example["conversations"]
            ],
            tokenize=False,
        )
        return {"text": text}

    def has_no_code(x):
        return all(
            [
                all(
                    y not in x["content"]
                    for y in ["```", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                )
                for x in x
            ]
        )

    def filtering(example):
        return [has_no_code(x) for x in example["messages"]]

    ds = ds.filter(filtering, batched=True, num_proc=20)
    ds = ds.map(formatting, batched=True, remove_columns=ds.column_names, num_proc=20)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("./ckpt/tokenizer")
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
    train(5000, 50000, 4096, 4, 16, "cuda:0", 500)
