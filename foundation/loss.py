import torch
from contextlib import nullcontext
# from transformers.trainer_utils import find_executable_batch_size

def calc_loss_batch(input_batch, target_batch, model, device, ctx=nullcontext()):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    if ctx is None:
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss
    
    with ctx:
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None, ctx=nullcontext()):
    total_loss = 0.
    if not data_loader: return float("nan")

    if num_batches is None: num_batches = len(data_loader)
    num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        loss = calc_loss_batch(input_batch, target_batch, model, device, ctx)
        total_loss += loss.item()

    return total_loss / num_batches

# @torch.no_grad() # torch.inference_mode()
def evaluate_model(model, train_loader, val_loader, device, iters):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=iters)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=iters)
    model.train()
    return train_loss, val_loss    
