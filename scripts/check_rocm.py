#!/usr/bin/env python3
"""Preflight only; does not guarantee that the model will fit or run."""
import sys, shutil, importlib.metadata as md
print("Python:", sys.version.split()[0])
for p in ("torch","transformers","accelerate","Pillow"):
    try: print(p,md.version(p))
    except md.PackageNotFoundError: print(p,"NOT INSTALLED")
for p in ("rocminfo","rocm-smi"):
    print(p,shutil.which(p) or "not found")
try:
    import torch
    print("torch.version.hip:",torch.version.hip)
    print("cuda API available:",torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print("device",i,torch.cuda.get_device_name(i))
except Exception as e: print("torch check error:",repr(e))
print("Check ROCm kernels, multi-GPU support and memory with a one-sample inference.")
