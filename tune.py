import torch
from time import time
import multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

# input data resolution 에 따라서 수치 변경 필요
res=224
batch_size=64
# res=512
# batch_size=8

def check():
    _transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=(res, res)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=_transforms
    )

    # workers 2 개씩 증가
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(training_data,shuffle=True,num_workers=num_workers,batch_size=batch_size,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

if __name__ == "__main__":
    check()