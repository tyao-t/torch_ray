import torch
from foundation.loss import calc_loss_batch, evaluate_model
from inference.generate import generate_text_stream_concat_flex
import itertools, functools, contextlib
from foundation.model import Qwen3Model
from text_and_data.pretrain_dataset import create_pretrain_dataloader
from transformers import AutoTokenizer
# @find_executable_batch_size(starting_batch_size=512, auto_find_batch_size=True)
# @auto_find_executable_batch_size(starting_batch_size=512)
def train_model(model, train_loader_wo_batch_size, val_loader_wo_batch_size, 
                optimizer, device, *, batch_size, tokenizer, warmup_steps=10, n_epochs=5, 
                eval_freq=10, eval_iter=1, initial_lr=3e-05, min_lr=1e-6, ctx=None):
    train_loader, val_loader = train_loader_wo_batch_size(batch_size), \
        val_loader_wo_batch_size(batch_size)
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    total_training_steps = len(train_loader) * n_epochs

    peak_lr = optimizer.param_groups[0]["lr"]
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        for input_batch, target_batch in train_loader:
            # if ddp:
            #     # with model.no_sync():
            #     model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            model.train()
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment # Warmup

            else:
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps)) 
                lr = peak_lr - (peak_lr - min_lr) * (0.5 * (1 - torch.cos(torch.pi * progress))) # Cosine annealing

            track_lrs.append(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        start_context = "To be or not to be"
        generate_text_stream_concat_flex(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs

if __name__ == "__main__":
    HPARAM_GRID = {
        "batch_size": [2, 4, 8, 16],
        "drop_rate": [0.0, 0.1, 0.2],
        "warmup_steps": [10, 20, 30],
        "weight_decay": [0.1, 0.01, 0.0],
        "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
        "initial_lr": [0.00005, 0.0001],
        "min_lr": [0.00005, 0.00001, 0.0001],
        "n_epochs": [5, 10, 15, 20, 25],
    }

    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}")

    # Placeholder for the best loss and best hyperparameters
    best_val_loss = float("inf")
    best_hparams = {}

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}. With device {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        if torch.cuda.get_device_capability()[0] >= 7:  # Hopper (9.0+), Ampere (8.0+), Turing (7.5+), Volta (7.0+)
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}\n")

    # 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    ctx = contextlib.nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    hparam_combs = list(itertools.product(*HPARAM_GRID.values()))
    best_val_loss = torch.inf
    best_hparams = None # {k: v[0] for k, v in HPARAM_GRID.items()}

    text_data = "To be or not to be, that is a question" # TO BE REPLACED

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True) # tiktoken.get_encoding("gpt2")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    interrupted = False
    current_config_idx_count = 0

    for combination in hyperparameter_combinations:
        try:
            current_config_idx_count += 1
            print(f"Evaluating configuration {current_config_idx_count} of {total_combinations}")

            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            MODEL_CONFIG = {
                "vocab_size": 50304,
                "context_length": 256,
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": HPARAM_CONFIG["drop_rate"],
                "qkv_bias": False,
            }

            train_loader_wo_batch_size = functools.partial(
                create_pretrain_dataloader,
                text=text_data[:split_idx],
                max_length=MODEL_CONFIG["context_length"],
                stride=MODEL_CONFIG["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0
            )

            train_loader_wo_batch_size = functools.partial(
                create_pretrain_dataloader,
                text=text_data[split_idx:],
                max_length=MODEL_CONFIG["context_length"],
                stride=MODEL_CONFIG["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0
            )

            model = Qwen3Model(MODEL_CONFIG)
            model = torch.compile(model)
            # model._orig_mod.state_dict(), model.module if DDP
            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=HPARAM_CONFIG["peak_lr"], 
                weight_decay=HPARAM_CONFIG["weight_decay"],
                fused=True # Fused AdamW optimizer Uses the fused kernels for AdamW
            )

            train_loss, val_loss = train_model(
                model, train_loader_wo_batch_size, train_loader_wo_batch_size, 
                optimizer, device, 
                batch_size=HPARAM_CONFIG["batch_size"],
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_iter=1,
                warmup_steps=HPARAM_CONFIG["warmup_steps"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"],
                ctx=ctx
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            break

    print("Hyperparameter search completed.")
    print(f"Best hyperparameters: {best_hparams}")
    print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")

# def get_device(enable_tensor_cores=True):
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("Using NVIDIA CUDA GPU")
        
#         if enable_tensor_cores:
#             major, minor = map(int, torch.__version__.split(".")[:2])
#             if (major, minor) >= (2, 9):
#                 torch.backends.cuda.matmul.fp32_precision = "tf32"
#                 torch.backends.cudnn.conv.fp32_precision = "tf32"
#             else:
#                 torch.backends.cuda.matmul.allow_tf32 = True
#                 torch.backends.cudnn.allow_tf32 = True
 
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("Using Apple Silicon GPU (MPS)")
 
#     elif torch.xpu.is_available():
#         device = torch.device("xpu")
#         print("Using Intel GPU")
 
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")
 
#     return device