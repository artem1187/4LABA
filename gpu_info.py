import torch

# Проверяем доступность CUDA
if torch.cuda.is_available():
rint("❌ CUDA недоступно")