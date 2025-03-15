import torch
from torch.utils.cpp_extension import load

abs_forward_module = load(name='abs_forward', sources=['abs_cuda.cpp', 'abs_cuda_kernel.cu'])


x = torch.randn(3, 3, device='cuda')

torch_abs = x.abs()
our_abs = abs_forward_module.abs_forward(x)[0] # Result is always vector<Tensor>

print(x) 
print(torch_abs)
print(our_abs)
print("All close:", torch.allclose(torch_abs, our_abs))