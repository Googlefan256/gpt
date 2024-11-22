import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    GPT2TokenizerFast,
    default_data_collator,
    Gemma2ForCausalLM,
    Gemma2Config,
)
from bitsandbytes import optim
import torch
from datasets import load_dataset, IterableDataset, DownloadConfig
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader
import os
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


zeropower_backends = dict(
    svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5
)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        backend="newtonschulz5",
        backend_steps=5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]
            zeropower_backend = zeropower_backends[group["backend"]]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group["params"])
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(group["params"]):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if not "WORLD_SIZE" in os.environ or i % int(
                    os.environ["WORLD_SIZE"]
                ) == int(os.environ["RANK"]):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()
            if "WORLD_SIZE" in os.environ:
                # sync updates across devices. we are not memory-constrained so can do this simple deserialization
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group["params"]:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


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
    config = Gemma2Config(
        vocab_size=len(tokenizer),
        hidden_size=384,
        intermediate_size=768,
        num_hidden_layers=40,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=max_seq_length,
    )
    config._attn_implementation = "sdpa"
    model = Gemma2ForCausalLM(config).to(device, torch.bfloat16).train()
    print(
        f"Model size: {sum([x.numel() for x in model.parameters()]) * 100 // 1000_000 / 100}M"
    )
    model: Gemma2ForCausalLM = torch.compile(model, options={"triton.cudagraphs": True})
    print("Compiled")
    params = list(model.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2]
    optimizer1 = Muon(matrix_params, lr=2e-4, momentum=0.95)
    optimizer2 = optim.AdamW8bit(scalar_params, lr=2e-4, betas=(0.9, 0.95), fused=True)
    optimizers = [optimizer1, optimizer2]
    schedulers = [
        get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
        for optimizer in optimizers
    ]
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
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)
        print(f"Step: {step}, Loss: {step_loss / train_accumulation_steps}")
        if step % save_steps == 0:
            model.save_pretrained("./ckpt")


if __name__ == "__main__":
    train(5000, 500000, 3072, 2, 16, "cuda:0", 5000)
