import time
import torch
from base_vit import ViT
from lora import LoRA_ViT
import numpy as np


BATCH_SIZE = 12
GPU = True

img = torch.randn(BATCH_SIZE, 3, 384, 384)
target = torch.randn(BATCH_SIZE, 1000)
criterion = torch.nn.MSELoss()

if GPU:
    img = img.to("cuda")
    target = target.to("cuda")


class TimeProfile:
    @staticmethod
    def test_base():
        model = ViT('B_16_imagenet1k')
        if GPU:
            model = model.to("cuda")
        preds = model(img)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = criterion(preds, target)
        start_time = time.time()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"base backpropagation took {time_cost:.4f} seconds")
        return time_cost

    @staticmethod
    def test_lora():
        model = ViT('B_16_imagenet1k')
        lora_model = LoRA_ViT(model, r=4, num_classes=1000)
        if GPU:
            lora_model = lora_model.to("cuda")
        preds = lora_model(img)
        optimizer = torch.optim.SGD(lora_model.parameters(), lr=0.1)
        loss = criterion(preds, target)
        start_time = time.time()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"LoRA backpropagation took {time_cost:.4f} seconds")
        return time_cost


class GRAMProfile:
    @staticmethod
    def test_base():
        model = ViT('B_16_imagenet1k')
        if GPU:
            model = model.to("cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        preds = model(img)
        torch.cuda.reset_peak_memory_stats()
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        print(f"Max memory used during backpropagation: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        return torch.cuda.max_memory_allocated() / 1024**2
    
    @staticmethod
    def test_lora():
        model = ViT('B_16_imagenet1k')
        lora_model = LoRA_ViT(model, r=4, num_classes=1000, lora_layer=)
        if GPU:
            lora_model = lora_model.to("cuda")
        optimizer = torch.optim.SGD(lora_model.parameters(), lr=0.1)
        optimizer.zero_grad()
        preds = lora_model(img)
        torch.cuda.reset_peak_memory_stats()
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        print(f"Max memory used during backpropagation: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        return torch.cuda.max_memory_allocated() / 1024**2


results_base = np.array([TimeProfile.test_base() for _ in range(10)])
results_lora = np.array([TimeProfile.test_lora() for _ in range(10)])
print(f"Base\nMean:{results_base.mean()} Std:{results_base.std()}")
print(f"LoRA\nMean:{results_lora.mean()} Std:{results_lora.std()}")


results_base = np.array([GRAMProfile.test_base() for _ in range(10)])
results_lora = np.array([GRAMProfile.test_lora() for _ in range(10)])
print(f"Base\nMean:{results_base.mean()} Std:{results_base.std()}")
print(f"LoRA\nMean:{results_lora.mean()} Std:{results_lora.std()}")