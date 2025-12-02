
# ================================================================
# 1. Импорты и проверка устройства
# ================================================================

import torch, torchvision, time, onnx, onnxruntime, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
