import os                                                                                                             
import random                                                                                                         
import pandas as pd                                                                                                   
import matplotlib.pyplot as plt                                                                                       
                  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 1. 基本设置                                                                             
SEED = 42
BATCH_SIZE = 128                                                                                                      
EPOCHS = 30     
LR = 1e-3
NUM_CLASSES = 10
GCE_Q = 0.7  # GCE 损失函数的 q 值
PATIENCE = 5  # 早停的耐心值                                                                                          
NOISE_RATE = 0.3
VAL_RATIO = 0.1  # 验证集比例                                                                                         
                                                                                                                        
RESULT_DIR = "results_gce_early_stop_1"
os.makedirs(RESULT_DIR, exist_ok=True)                                                                                
                  
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
                                                                                                                        
def get_device():
    if torch.cuda.is_available():                                                                                     
          return torch.device("cuda")
    if torch.backends.mps.is_available():
          return torch.device("mps")
    return torch.device("cpu")

set_seed(SEED)
device = get_device()
print(f"Using device: {device}")
                                                                                                                        
# 2. GCE 损失函数 
class GCELoss(nn.Module):                                                                                             
    """         
      Generalized Cross Entropy Loss
      L = (1 - y_pred^q) / q
      q=0.7 表示对噪声标签更鲁棒                                                                                        
    """
    def __init__(self, q=0.7):                                                                                        
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        y_pred = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0 - 1e-7)                                                     
        loss = (1 - y_pred.pow(self.q)) / self.q
        return loss.mean()                                                                                            
                  
# 3. 带噪标签数据集                                                                         
class NoisyLabelDataset(Dataset):
    def __init__(self, base_dataset, noise_rate=0.3, num_classes=10, seed=42):                                        
        self.base_dataset = base_dataset

        if hasattr(base_dataset, "targets"):                                                                          
            targets = torch.tensor(base_dataset.targets).long()
        else:                                                                                                         
            targets = torch.tensor([base_dataset[i][1] for i in range(len(base_dataset))]).long()

        self.targets = targets.clone()

        if noise_rate > 0:
            generator = torch.Generator().manual_seed(seed)
            n = len(self.targets)                                                                                     
            n_noisy = int(noise_rate * n)
                                                                                                                        
            noisy_indices = torch.randperm(n, generator=generator)[:n_noisy]
            random_labels = torch.randint(0, num_classes, (n_noisy,), generator=generator)
                                                                                                                        
            same_mask = random_labels == self.targets[noisy_indices]
            random_labels[same_mask] = (random_labels[same_mask] + 1) % num_classes                                   
                  
            self.targets[noisy_indices] = random_labels
            print(f"Added noisy labels: {n_noisy}/{n} = {noise_rate:.0%}")
                                                                                                                        
    def __len__(self):
        return len(self.base_dataset)                                                                                 
                  
    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        y = int(self.targets[idx])
        return x, y                                                                                                   
   
  # 4. 数据加载（划分验证集）                                                                
def get_dataloaders(noise_rate=0.3, val_ratio=0.1):
    transform = transforms.ToTensor()

    train_full = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    n = len(train_full)
    n_val = int(n * val_ratio)

    # 固定随机划分，保证可复现
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator).tolist()

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # 先划分出干净的训练子集和验证子集
    train_clean_subset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(train_full, val_indices)

    # 只对训练子集加噪声，验证集保持干净
    train_dataset = NoisyLabelDataset(
        train_clean_subset,
        noise_rate=noise_rate,
        num_classes=NUM_CLASSES,
        seed=SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader                                                                  
                  
  #  5. 模型定义 
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(28 * 28, NUM_CLASSES)
                                                                                                                        
    def forward(self, x):
        x = x.view(x.size(0), -1)                                                                                     
        return self.classifier(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),                                                                                      
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):                                                                                             
        return self.net(x)
                                                                                                                        
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(2),                                                                                          
              nn.Conv2d(32, 64, kernel_size=3, padding=1),
              nn.ReLU(),                                                                                                
              nn.MaxPool2d(2)
          )

    def forward(self, x):
        return self.features(x)
                                                                                                                        
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):                                                                        
          super().__init__()
          hidden_dim = max(channels // reduction, 1)
          self.avg_pool = nn.AdaptiveAvgPool2d(1)
          self.fc1 = nn.Linear(channels, hidden_dim)
          self.fc2 = nn.Linear(hidden_dim, channels)                                                                    
   
    def forward(self, x):                                                                                             
          batch_size, channels, _, _ = x.size()
          z = self.avg_pool(x).view(batch_size, channels)
          a = self.fc2(F.relu(self.fc1(z)))
          s = 2.0 * torch.sigmoid(a)                                                                                    
          s = s.view(batch_size, channels, 1, 1)
          return x * s                                                                                                  
                  

class ConvGapHead(nn.Module):
    """
    1 × 1 Conv + GAP
    [B, 64, 7, 7] -> [B, 10, 7, 7] -> [B, 10]
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1) #输出64个通道，出来10个通道

    def forward(self, x):
        x = self.conv1x1(x)
        logits = x.mean(dim=(2, 3)) #在高度和宽度维度上求平均
        return logits


class FlattenFCHead(nn.Module):
    """
    Flatten + FC
    [B, 64, 7, 7] -> [B, 64*7*7] -> [B, 10]
    """
    def __init__(self, in_channels=64, h=7, w=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(), #展平
            nn.Linear(in_channels * h * w, NUM_CLASSES) #全连接层
        )

    def forward(self, x):
        return self.classifier(x)


class SPPHead(nn.Module):
    """
    SPP + FC
    使用 1×1, 2×2, 4×4 三个尺度。
    每个通道得到 1 + 4 + 16 = 21 个特征。
    [B, 64, 7, 7] -> [B, 64*21] -> [B, 10]
    """
    def __init__(self, in_channels=64, levels=(1, 2, 4)):
        super().__init__()
        self.levels = levels
        spp_dim = in_channels * sum(level * level for level in levels) 
        self.fc = nn.Linear(spp_dim, NUM_CLASSES)  

    def forward(self, x):
        pooled_features = []

        for level in self.levels: #遍历1，2，4
            pooled = F.adaptive_max_pool2d(x, output_size=(level, level))
            pooled = pooled.view(pooled.size(0), -1)
            pooled_features.append(pooled)

        x = torch.cat(pooled_features, dim=1) #拼接
        logits = self.fc(x)
        return logits
                                                                                                 
                  
class CNNModel(nn.Module):
    def __init__(self, head_type="spp", use_se=False): #head_type为分类头类型，use_se为是否使用注意力机制
        super().__init__()

        self.features = CNNFeatureExtractor()   #CNN特征提取器
        self.use_se = use_se

        if use_se:
            self.se = SEBlock(channels=64, reduction=8) #有SE模块
        else:
            self.se = nn.Identity()   #不加入注意力机制，对应Basic CNN

        if head_type == "gap": 
            self.head = ConvGapHead(in_channels=64)
        elif head_type == "flatten":
            self.head = FlattenFCHead(in_channels=64, h=7, w=7)
        elif head_type == "spp":
            self.head = SPPHead(in_channels=64, levels=(1, 2, 4))
        else:
            raise ValueError("head_type must be one of: 'gap', 'flatten', 'spp'")

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        logits = self.head(x)
        return logits
      
  #  6. 工具函数 
def count_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)
                                                                                                                        
def train_one_epoch(model, train_loader, criterion, optimizer, device):
      model.train()                                                                                                     
      total_loss, correct, total = 0.0, 0, 0

      for images, labels in train_loader:                                                                               
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()                                                                                         
          logits = model(images)
          loss = criterion(logits, labels)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #割掉过大的梯度，防止权重突然跳变
          optimizer.step()

          total_loss += loss.item() * images.size(0)
          preds = logits.argmax(dim=1)
          correct += (preds == labels).sum().item()                                                                     
          total += labels.size(0)
                                                                                                                        
      return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()   # 评估固定使用 CE
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

#  7. 早停法                                                                                
class EarlyStopping:
      def __init__(self, patience=5, min_delta=0.0001):                                                                 
          self.patience = patience
          self.min_delta = min_delta
          self.counter = 0
          self.best_loss = None
          self.early_stop = False                                                                                       
   
      def __call__(self, val_loss):                                                                                     
          if self.best_loss is None:
              self.best_loss = val_loss
          elif val_loss > self.best_loss - self.min_delta:
              self.counter += 1
              if self.counter >= self.patience:
                  self.early_stop = True                                                                                
          else:
              self.best_loss = val_loss                                                                                 
              self.counter = 0

  # 8. 运行实验 
def run_experiment(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    use_gce=True,
    use_early_stopping=True
):
    model = model.to(device)

    train_criterion = GCELoss(q=GCE_Q) if use_gce else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=1e-4)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    loss_name = "GCE" if use_gce else "CE"
    es_name = "ES" if use_early_stopping else "NoES"

    print(f"\n Training {model_name} :")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Loss: {loss_name}, Early Stopping: {'Yes' if use_early_stopping else 'No'}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train: {train_loss:.4f}/{train_acc * 100:.1f}% | "
            f"Val: {val_loss:.4f}/{val_acc * 100:.1f}%"
        )

        # 始终记录验证集损失最低的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # 只有 use_early_stopping=True 时才触发早停
        if use_early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f">>> Early stopping at epoch {epoch}")
                break

    # 如果使用早停，则恢复验证损失最低的模型；
    # 如果不使用早停，则保留最后一轮模型作为最终模型。
    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc = evaluate(model, val_loader, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, device)

    result = {
        "model": model_name,
        "base_model": model_name.split(" (")[0],
        "loss": loss_name,
        "early_stopping": use_early_stopping,
        "strategy": f"{loss_name}+ES" if use_early_stopping else loss_name,
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc * 100,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc * 100,
    }

    return result, history

#  9. 主程序 
def main():
    train_loader, val_loader, test_loader = get_dataloaders(noise_rate=NOISE_RATE)

    model_builders = [
        ("Linear", lambda: LinearClassifier()),
        ("MLP", lambda: MLP()),
        ("CNN-SPP", lambda: CNNModel(head_type="spp", use_se=False)),
        ("SE-CNN-SPP", lambda: CNNModel(head_type="spp", use_se=True)),
    ]

    strategies = [
        ("CE", False, False),
        ("CE+ES", False, True),
        ("GCE", True, False),
        ("GCE+ES", True, True),
    ]

    all_results = []
    all_histories = {}

    for base_name, build_model in model_builders:
        for strategy_name, use_gce, use_early_stopping in strategies:
            set_seed(SEED)
            model = build_model()

            model_name = f"{base_name} ({strategy_name})"

            result, history = run_experiment(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                use_gce=use_gce,
                use_early_stopping=use_early_stopping
            )

            # 注意：append 一定要在内层循环里，且不能写进 if use_gce and use_early_stopping 里面
            all_results.append(result)
            all_histories[model_name] = history

    print("Number of results:", len(all_results))
    print([r["model"] for r in all_results])

    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(RESULT_DIR, "ablation_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("ABLATION STUDY: CE / CE+ES / GCE / GCE+ES")
    print("=" * 120)

    cols = [
        "base_model",
        "strategy",
        "loss",
        "early_stopping",
        "params",
        "best_epoch",
        "epochs_trained",
        "best_val_loss",
        "final_test_loss",
        "final_test_acc"
    ]

    disp_df = results_df[cols].sort_values(
        by=["base_model", "strategy"]
    )

    print(disp_df.to_string(index=False))
    print(f"\nResults saved to: {csv_path}")                                                                  
                  
if __name__ == "__main__":                                                                                            
      main()
            