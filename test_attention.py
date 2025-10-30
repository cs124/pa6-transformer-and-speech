import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import CausalSelfAttention

# Define a simple config class for testing
@dataclass
class Config:
    n_embd: int
    n_head: int
    block_size: int
    dropout: float = 0.0
    bias: bool = True


# Load the saved tensors and check outputs match when run through attention
saved = torch.load('attention_test_tensors.pt')
x1 = saved['input1']
out1_saved = saved['output1']
x2 = saved['input2']
out2_saved = saved['output2']

# Ensure deterministic CausalSelfAttention
torch.manual_seed(42)
test_config = Config(n_embd=8, n_head=2, block_size=10, dropout=0.0, bias=True)
causal_attn = CausalSelfAttention(test_config)
causal_attn.eval()  # turn off dropout

# Pass loaded inputs through the attention module
with torch.no_grad():
    out1_new = causal_attn(x1)
    out2_new = causal_attn(x2)

# Check that the outputs match the saved outputs
match1 = torch.allclose(out1_new, out1_saved, atol=1e-6)
match2 = torch.allclose(out2_new, out2_saved, atol=1e-6)

print("Test 1 output match:", match1)
print("Test 2 output match:", match2)

if not match1 or not match2:
    # Print diffs for debugging
    print("Output 1 max diff:", (out1_new - out1_saved).abs().max().item())
    print("Output 2 max diff:", (out2_new - out2_saved).abs().max().item())