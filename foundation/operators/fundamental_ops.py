import torch
import torch.nn.functional as F

def softmax(x, dim=-1):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def softmax(x, **kwargs):
    return F.softmax(x, **kwargs)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def silu(x):
    return x * sigmoid(x)

def silu(x):
    return x / (1 + torch.exp(-x))

def silu(x):
    return F.silu(x)

def swish(x, beta=torch.tensor(1)):
    return x / (1 + torch.exp(-beta*x))

def gelu_exact(x):
    return 0.5 * x * (1 + torch.erf(x / torch.sqrt(2)))

# Standard tanh approx.
def gelu_tanh_approx(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2/torch.pi) * (x + 0.044715 * x**3)))

# Sigmoid approx.
def gelu_sigmoid_approx(x):
    return x * torch.sigmoid(1.702 * x)

# Fast approx.
def gelu_quick_approx(x):
    return x * torch.sigmoid(1.4 * x)

def gelu(x):
    return F.gelu(x, approximate="sigmoid")

def gelu(x):
    return F.gelu(x, approximate="none")

def gelu(x):
    return F.gelu(x, approximate="tanh")
