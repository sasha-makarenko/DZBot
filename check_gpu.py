import torch 

print("Pytorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("GPU computation OK, result shape:", tuple(y.shape))
else:
    print("CUDA not available: скорее всего стоит CPU сборка")

