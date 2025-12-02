
# ================================================================
# 1. Импорты и проверка устройства
# ================================================================

import torch, torchvision, time, onnx, onnxruntime, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================================================================
# 2. Преобразования и загрузка CIFAR-10
# ================================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.CIFAR10(root="data", train=True,  download=True, transform=transform)
val_ds   = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=6, shuffle=True,  num_workers=8)
val_loader   = DataLoader(val_ds,   batch_size=6, shuffle=False, num_workers=8)

# ================================================================
# 3. Берём предобученную «сверхточную» модель
#    EfficientNet-B3 даёт ≈ 84 % Top-1 на ImageNet → хороший выбор
# ================================================================
model = torchvision.models.efficientnet_b3(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def accuracy(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (net(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct/total

for epoch in range(3):
    model.train()
    # оборачиваем именно тренировочный loader
    with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            # обновляем строку прогресса текущим лоссом
            pbar.set_postfix(loss=loss.item())

    # после эпохи — метрика
    print(f"Epoch {epoch+1}  val-acc={accuracy(model, val_loader):.3f}")
