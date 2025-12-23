import torch
import torch.nn as nn
from foundation.model import Qwen3Model

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=torch.sqrt(5)) # similar to standard weight initialization
        # torch.nn.init.uniform_(A, -1.0 / torch.sqrt(in_dim), 1.0 / torch.sqrt(in_dim))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)

class AdapterLayer(nn.Module):
    """
    Adapter: x -> DownProject -> Activation -> UpProject -> (optionally scaled)
    """
    def __init__(self, in_features, adapter_size, activation=nn.GELU(), init_scale=1e-3, init_up="zero"):
        """
        init_up:
        "zero": up_proj weights/biases are exactly 0 (Pfeiffer-style "no-op" at start)
        "normal": up_proj weights ~ N(0, init_scale) so the adapter starts near-no-op but not exactly
        """
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_size)
        self.up_proj = nn.Linear(adapter_size, in_features)
        self.activation = activation

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=torch.sqrt(torch.tensor(5.0)))
            nn.init.zeros_(self.down_proj.bias)

            if init_up == "zero":
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.up_proj.bias)
            elif init_up == "normal":
                nn.init.normal_(self.up_proj.weight, mean=0.0, std=init_scale)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.up_proj(z)
        return z

class LinearWithAdapter(nn.Module):
    # Pfeiffer-style: output = original_layer(x) + adapter(original_layer(x))
    def __init__(self, linear_layer, adapter_size, init_scale=1e-3, init_up="zero"):
        super().__init__()
        self.linear_layer = linear_layer
        self.adapter = AdapterLayer(linear_layer.out_features, adapter_size, init_scale=init_scale, init_up=init_up)

    def forward(self, x):
        y = self.linear_layer(x)
        return y + self.adapter(y)

if __name__ == "__main__":
    model = Qwen3Model({"..."})
    torch.manual_seed(23)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters before: {total_params:,}")
    # Total trainable parameters before: 124,441,346

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")
    # Total trainable parameters after: 0

    replace_linear_with_lora(model, rank=16, alpha=16)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")
    # Total trainable LoRA parameters: 2,666,528

    # Also, since we initialized matrix with 0's, we expect the initial model performance to be unchanged compared to before