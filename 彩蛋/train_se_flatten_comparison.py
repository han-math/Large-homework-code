import os
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 1. 基本实验设置（与原实验完全一致）
# ============================================================
SEED = 42
BATCH_SIZE = 120
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 10
NOISE_RATE = 0.0  # 无噪声

RESULT_DIR = "results_se_flatten_comparison"
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


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


# ============================================================
# 2. 数据加载（无噪声）
# ============================================================
def get_dataloaders():
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


# ============================================================
# 3. CNN 特征提取器（与原实验完全一致）
# ============================================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.features(x)


# ============================================================
# 4. 分类头：SPP Head（与原实验完全一致）
# ============================================================
class SPPHead(nn.Module):
    def __init__(self, in_channels=64, levels=(1, 2, 4)):
        super().__init__()
        self.levels = levels
        spp_dim = in_channels * sum(level * level for level in levels)
        self.fc = nn.Linear(spp_dim, NUM_CLASSES)

    def forward(self, x):
        pooled_features = []
        for level in self.levels:
            pooled = F.adaptive_max_pool2d(x, output_size=(level, level))
            pooled = pooled.view(pooled.size(0), -1)
            pooled_features.append(pooled)
        x = torch.cat(pooled_features, dim=1)
        return self.fc(x)


# ============================================================
# 5. 原版 SE 模块：Squeeze = 全局平均池化 (GAP)
# ============================================================
class SEBlockGAP(nn.Module):
    """
    标准 SE 模块：Squeeze 用全局平均池化
    输入: [B, C, H, W] → 输出: [B, C, H, W]
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_dim = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        a = self.fc2(F.relu(self.fc1(z)))
        s = 2.0 * torch.sigmoid(a).view(b, c, 1, 1)
        return x * s


# ============================================================
# 6. 改进版 SE 模块：Squeeze = Flatten（替代全局平均池化）
# ============================================================
class SEBlockFlatten(nn.Module):
    """
    改进 SE 模块：Squeeze 阶段用 Flatten 替代全局平均池化。
    保留所有空间位置信息，让 FC 层自己学习哪些空间位置对通道注意力重要。
    输入: [B, C, H, W] → 输出: [B, C, H, W]

    hidden_dim 取 channels 的 2 倍作为默认，保证有足够容量编码空间+通道信息。
    """
    def __init__(self, channels, h=7, w=7, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channels * 2  # 64 → 128
        self.channels = channels
        self.h = h
        self.w = w
        input_dim = channels * h * w
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        z = x.view(b, -1)  # [B, C*H*W] — Flatten 替代 GAP
        a = self.fc2(F.relu(self.fc1(z)))
        s = 2.0 * torch.sigmoid(a).view(b, c, 1, 1)
        return x * s


# ============================================================
# 7. 统一 CNN 模型（通过 se_type 切换 SE 变体）
# ============================================================
class CNNModel(nn.Module):
    def __init__(self, se_type="none"):
        """
        se_type: "none" → Basic CNN（无注意力）
                 "gap"   → 原版 SE-CNN（GAP Squeeze）
                 "flatten" → 改进 SE-CNN（Flatten Squeeze）
        """
        super().__init__()
        self.features = CNNFeatureExtractor()
        self.se_type = se_type

        if se_type == "gap":
            self.se = SEBlockGAP(channels=64, reduction=8)
        elif se_type == "flatten":
            self.se = SEBlockFlatten(channels=64, h=7, w=7, hidden_dim=128)
        else:
            self.se = nn.Identity()

        self.head = SPPHead(in_channels=64, levels=(1, 2, 4))

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        return self.head(x)


# ============================================================
# 8. 训练与评估（与原实验完全一致）
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def run_experiment(model_name, model, train_loader, test_loader):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%"
        )

    best_test_acc = max(history["test_acc"])
    best_epoch = history["epoch"][history["test_acc"].index(best_test_acc)]

    return {
        "model": model_name,
        "params": n_params,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_acc": history["test_acc"][-1],
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
    }, history


# ============================================================
# 9. 绘图
# ============================================================
def plot_metric(histories, metric_name, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for model_name, history in histories.items():
        plt.plot(history["epoch"], history[metric_name],
                 marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 10. Grad-CAM
# ============================================================
class GradCAM:
    """
    不依赖 backward hook（MPS 兼容），改用 torch.autograd.grad 直接求梯度。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output

    def remove_hooks(self):
        self.forward_handle.remove()

    def __call__(self, input_tensor, target_class=None):
        # 全程在 CPU 上计算，避免 MPS backward hook 兼容问题
        original_device = next(self.model.parameters()).device
        cpu_model = self.model.to("cpu")
        cpu_input = input_tensor.to("cpu")
        cpu_model.eval()

        # 在 CPU 模型上重新挂 forward hook
        cpu_target = cpu_model.features.features[3]
        self.remove_hooks()
        self.forward_handle = cpu_target.register_forward_hook(self._save_activation)

        # 需要梯度，手工用 autograd.grad 求
        cpu_input_grad = cpu_input.detach().clone().requires_grad_(True)
        logits = cpu_model(cpu_input_grad)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # autograd.grad 求 logits 对 target_layer 输出的梯度
        score = logits[:, target_class].sum()
        grads = torch.autograd.grad(
            score, self.activations,
            retain_graph=False,
            create_graph=False
        )[0]  # [1, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=cpu_input.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        pred_class = logits.argmax(dim=1).item()
        pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()

        # 恢复模型到原设备
        self.model = cpu_model.to(original_device)

        return cam.detach(), pred_class, pred_prob


@torch.no_grad()
def collect_one_correct_sample_per_class(models, test_loader, device, class_ids, num_classes=10):
    """
    每个类别选一张样本，要求所有模型都预测正确。
    models: dict {model_name: model}
    """
    for m in models.values():
        m.eval()

    selected = {}

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            label = labels[i].item()

            if label not in class_ids:
                continue
            if label in selected:
                continue

            # 要求所有模型都预测正确
            all_correct = True
            for model in models.values():
                pred = model(images[i:i+1]).argmax(dim=1).item()
                if pred != label:
                    all_correct = False
                    break

            if all_correct:
                selected[label] = images[i:i+1].detach().cpu()

            if len(selected) == len(class_ids):
                return selected

    return selected


def save_gradcam_comparison(models, test_loader, device, save_dir, class_ids):
    """
    为指定类别生成 Grad-CAM 对比图：
    三模型并排：CNN_SPP | SE-CNN (GAP) | SE-CNN (Flatten)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 所有模型共用同一个 target_layer：features.features[3]（第二个 Conv2d）
    target_layer = list(models.values())[0].features.features[3]

    selected = collect_one_correct_sample_per_class(
        models, test_loader, device, class_ids
    )

    print(f"\nGrad-CAM selected classes: {[CLASS_NAMES[c] for c in sorted(selected.keys())]}")

    for label, image_cpu in selected.items():
        image = image_cpu.to(device)

        # 为每个模型创建 GradCAM
        cam_results = {}
        for model_name, model in models.items():
            cam_extractor = GradCAM(model, target_layer)
            heatmap, pred_class, pred_prob = cam_extractor(image, target_class=label)
            cam_results[model_name] = (heatmap, pred_class, pred_prob)
            cam_extractor.remove_hooks()

        original = image.squeeze().detach().cpu().numpy()

        # 四列：Original | CNN_SPP | SE-CNN (GAP) | SE-CNN (Flatten)
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

        # 原图
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[label]}", fontsize=10)
        axes[0].axis("off")

        # 三个模型的 Grad-CAM
        for ax, (model_name, (heatmap, pred_class, pred_prob)) in zip(axes[1:], cam_results.items()):
            ax.imshow(original, cmap="gray")
            ax.imshow(heatmap.numpy(), alpha=0.5, cmap="jet")
            short_name = model_name.replace("SE-CNN_SPP", "SE-CNN")
            ax.set_title(
                f"{short_name}\nPred: {CLASS_NAMES[pred_class]}\nProb: {pred_prob:.3f}",
                fontsize=9
            )
            ax.axis("off")

        plt.tight_layout()
        file_name = f"gradcam_{label}_{CLASS_NAMES[label].replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, file_name), dpi=300)
        plt.close()

    print(f"Grad-CAM figures saved to: {save_dir}")


# ============================================================
# 11. 随机权重对照：Flatten SE 的注意力分支随机初始化并冻结
# ============================================================
def build_frozen_random_se_model():
    """
    构建 SE-CNN (Flatten)，将其 SE 模块的 fc1/fc2 随机初始化后冻结，
    只训练特征提取器和分类头。
    用于验证 Flatten SE 的注意力是否真正学到了有用的东西。
    """
    model = CNNModel(se_type="flatten")
    # 随机初始化 SE 分支（保持默认 Kaiming 初始化即可，已经是随机的）
    # 冻结 SE 模块的所有参数
    for name, param in model.se.named_parameters():
        param.requires_grad = False
    return model


# ============================================================
# 12. 主程序：四模型对比 + Grad-CAM 可视化
# ============================================================
def main():
    train_loader, test_loader = get_dataloaders()

    model_builders = {
        "CNN_SPP":                          lambda: CNNModel(se_type="none"),
        "SE-CNN_SPP (GAP)":                 lambda: CNNModel(se_type="gap"),
        "SE-CNN_SPP (Flatten)":             lambda: CNNModel(se_type="flatten"),
        "SE-CNN_SPP (Flatten, RandFrozen)": build_frozen_random_se_model,
    }

    all_results = []
    all_histories = {}
    trained_models = {}

    for model_name, build_model in model_builders.items():
        set_seed(SEED)
        model = build_model()
        result, history = run_experiment(model_name, model, train_loader, test_loader)
        all_results.append(result)
        all_histories[model_name] = history
        # 仅保存前三个主模型用于 Grad-CAM，冻结对照组不参与
        if "RandFrozen" not in model_name:
            trained_models[model_name] = model

    # ---- 打印结果对比 ----
    results_df = pd.DataFrame(all_results)
    results_df["final_train_acc"] = results_df["final_train_acc"] * 100
    results_df["final_test_acc"] = results_df["final_test_acc"] * 100
    results_df["best_test_acc"] = results_df["best_test_acc"] * 100

    csv_path = os.path.join(RESULT_DIR, "summary_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("Summary Results (Test Accuracy)")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<25s}  "
            f"Best Test Acc: {row['best_test_acc']:.2f}%  "
            f"(epoch {int(row['best_epoch'])})  "
            f"Params: {int(row['params']):,}"
        )
    print(f"\nResults saved to: {csv_path}")

    # ---- 保存训练曲线 ----
    plot_metric(all_histories, "train_loss", "Training Loss",
                os.path.join(RESULT_DIR, "train_loss_curves.png"))
    plot_metric(all_histories, "test_loss", "Test Loss",
                os.path.join(RESULT_DIR, "test_loss_curves.png"))
    plot_metric(all_histories, "train_acc", "Training Accuracy",
                os.path.join(RESULT_DIR, "train_acc_curves.png"))
    plot_metric(all_histories, "test_acc", "Test Accuracy",
                os.path.join(RESULT_DIR, "test_acc_curves.png"))

    print(f"Figures saved to: {RESULT_DIR}/")

    # ---- Grad-CAM 可视化：T-shirt(0), Pullover(2), Coat(4), Shirt(6) ----
    print("\n" + "=" * 70)
    print("Generating Grad-CAM Comparison")
    print("=" * 70)
    save_gradcam_comparison(
        models=trained_models,
        test_loader=test_loader,
        device=device,
        save_dir=os.path.join(RESULT_DIR, "gradcam"),
        class_ids=[0, 2, 4, 6]  # T-shirt, Pullover, Coat, Shirt
    )


if __name__ == "__main__":
    main()
