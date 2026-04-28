import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 1. 基本实验设置
SEED = 42
BATCH_SIZE = 120
# EPOCHS = 10
EPOCHS = 20 #抗噪实验选长一点的epoch数，有早停措施
LR = 1e-3
NUM_CLASSES = 10

# 抗噪训练设置
LOSS_TYPE = "gce"          # "ce" 表示普通交叉熵；"gce" 表示 Generalized Cross Entropy
GCE_Q = 0.7                # 常用可先取 0.7；越接近 0 越像 CE，越接近 1 越抗噪
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3    # 连续 3 个 epoch 测试准确率没有提升就停止
EARLY_STOP_MIN_DELTA = 1e-4

## 如果要恢复正常训练，把EPOCHS改回为10，LOSS_TYPE="ce",USE_EARLY_STOPPING = False

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


#设为0.0不含噪声，改为0.3即30%带噪标签
NOISE_RATE = 0.3

#设置存储结果的文件夹
RESULT_DIR = "results_main_k_noise_4"
os.makedirs(RESULT_DIR, exist_ok=True)

#设置可复现的随机种子
def set_seed(seed=42):
    random.seed(seed)        # Python random 模块
    torch.manual_seed(seed)      # PyTorch CPU 随机
    torch.cuda.manual_seed_all(seed)    # PyTorch GPU 随机

#灵活选用计算设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps") 
    return torch.device("cpu")

set_seed(SEED)
device = get_device()
print(f"Using device: {device}")


# 2. 构造带噪标签数据集
class NoisyLabelDataset(Dataset):
    """
    对训练集加入随机错误标签。
    noise_rate=0.3 表示随机挑选 30% 样本，并把它们的标签改成错误类别。
    """
    def __init__(self, base_dataset, noise_rate=0.3, num_classes=10, seed=42): #噪声时改为0.3
        self.base_dataset = base_dataset
        self.num_classes = num_classes

        #提取数据集标签
        if hasattr(base_dataset, "targets"):
            targets = torch.tensor(base_dataset.targets).long()
        else:
            targets = torch.tensor([base_dataset[i][1] for i in range(len(base_dataset))]).long()
        # 把原始标签复制一份保存
        self.targets = targets.clone()
        #有噪声情况
        if noise_rate > 0:
            generator = torch.Generator().manual_seed(seed) #设置随机种子，保证可复现
            n = len(self.targets) #总训练集样本量
            n_noisy = int(noise_rate * n)  #噪声标签数

            noisy_indices = torch.randperm(n, generator=generator)[:n_noisy] #把训练集样本打乱，取前n_noisy个
            random_labels = torch.randint(
                low=0,
                high=num_classes,
                size=(n_noisy,),
                generator=generator
            ) #生成n_noisy个随机标签

            #保证变成错误标签，避免随机到原标签
            same_mask = random_labels == self.targets[noisy_indices] #创建布尔数组，记录哪些随机标签与原标签相同
            random_labels[same_mask] = (random_labels[same_mask] + 1) % num_classes #修正相同情况

            self.targets[noisy_indices] = random_labels #覆写标签

            print(f"Added noisy labels: {n_noisy}/{n} = {noise_rate:.0%}")
    
    #返回数据集大小
    def __len__(self):
        return len(self.base_dataset)

    #返回（图片，标签）
    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        y = int(self.targets[idx])
        return x, y


def get_dataloaders(noise_rate=0.3): #噪声时改为0.3
    transform = transforms.ToTensor()  #像素自动缩放到 [0, 1]
    
    #创建训练集
    train_base = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    #创建测试集
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    #给训练集加噪声
    train_dataset = NoisyLabelDataset(
        train_base,
        noise_rate=noise_rate,
        num_classes=NUM_CLASSES,
        seed=SEED
    )
    #用mini - batch包装成 DataLoader（训练）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,    #打乱顺序
        num_workers=0
    )
    #用mini - batch包装成 DataLoader（测试）
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


# 3. 模型一：线性分类器
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(28 * 28, NUM_CLASSES) #线性分类器，输入784个像素点，输出10个类别

    def forward(self, x):
        x = x.view(x.size(0), -1)  #展平图像
        logits = self.classifier(x)  #通过线性分类器
        return logits


# 4. 模型二：MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        #两个隐藏层，后面对应CNN也两个卷积层
        self.net = nn.Sequential(
            nn.Flatten(),  #展平
            nn.Linear(28 * 28, 256),  #输入28*28个，输出256个，下同理
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


# 5. CNN 特征提取部分
class CNNFeatureExtractor(nn.Module):
    """
    输入:  [B, 1, 28, 28], B = batch size
    输出:  [B, 64, 7, 7], 64个特征图, 7x7大小
    """
    def __init__(self):
        super().__init__()
        #卷积层步长为1，用3x3卷积核，图像周围一圈填充0
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  #输入[B,1,28,28],输出[B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                 #输出[B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), #输出[B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  #输出[B, 64, 7, 7]
        )

    def forward(self, x):
        return self.features(x) #特征张量


# 6. Basic CNN 的三种分类头
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


# 7. SE 注意力模块
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    输入:  [B, C, H, W]
    输出:  [B, C, H, W]
    用 2 * sigmoid(a)，使初始通道权重大约在 1 附近，
    避免训练初期把特征整体压小。
    """
    def __init__(self, channels, reduction=8): #有64个通道，取压缩比例为8
        super().__init__()
        hidden_dim = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #定义全局平均池化
        self.fc1 = nn.Linear(channels, hidden_dim)  #定义两个全连接层
        self.fc2 = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        z = self.avg_pool(x).view(batch_size, channels)  #Squeeze阶段
        a = self.fc2(F.relu(self.fc1(z)))  #Excitation阶段

        # 原 SE: sigmoid(a), 初始约 0.5
        # 改进版: 2 * sigmoid(a), 初始约 1
        s = 2.0 * torch.sigmoid(a)
        s = s.view(batch_size, channels, 1, 1)

        return x * s
    

# 8. Basic CNN / SE-CNN 统一模型
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



# 9. 训练与测试函数
def count_parameters(model): #计算参数量，只计算需要梯度的参数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 抗噪措施：更换loss函数
class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy Loss, 简称 GCE.
    q -> 0 时接近普通交叉熵 CE；
    q -> 1 时更接近 MAE，对错误标签不那么敏感。
    """
    def __init__(self, q=0.7, eps=1e-7):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        # 取出模型对真实标签 targets 的预测概率 p_y
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = pt.clamp(min=self.eps, max=1.0)

        if self.q == 0:
            loss = -torch.log(pt)
        else:
            loss = (1.0 - pt.pow(self.q)) / self.q

        return loss.mean()


def build_criterion(loss_type="ce"):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "gce":
        return GeneralizedCrossEntropyLoss(q=GCE_Q)
    else:
        raise ValueError("loss_type must be 'ce' or 'gce'")
    

def train_one_epoch(model, train_loader, criterion, optimizer, device): #定义一个 epoch 的完整流程
    model.train()  #开启训练模式
    
    #初始化
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:  #遍历每个batch
        images = images.to(device)      
        labels = labels.to(device)

        optimizer.zero_grad()    #梯度清零

        logits = model(images)    #向前传播
        loss = criterion(logits, labels)   #计算loss

        loss.backward()     #向后传播
        optimizer.step()    #更新参数

        total_loss += loss.item() * images.size(0)   #计算累计loss

        preds = logits.argmax(dim=1)     #预测类别
        correct += (preds == labels).sum().item()   #正确数
        total += labels.size(0)    #总数

    avg_loss = total_loss / total   #平均loss
    acc = correct / total    #准确率

    return avg_loss, acc


@torch.no_grad()  #做对测试集的评估，不计算梯度，加速推理
def evaluate(model, data_loader, criterion, device):
    model.eval() #评估模式

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)    #向前传播
        loss = criterion(logits, labels)  #计算loss

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)  #预测类别
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

#计算混淆矩阵
@torch.no_grad()
def get_predictions(model, data_loader, device):
    model.eval()

    all_labels = []
    all_preds = []

    for images, labels in data_loader:
        images = images.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    return all_labels, all_preds


def compute_confusion_matrix(labels, preds, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for true_label, pred_label in zip(labels, preds):
        cm[true_label, pred_label] += 1

    return cm


def plot_confusion_matrix(cm, class_names, save_path, normalize=False, title="Confusion Matrix"):
    cm = cm.numpy().astype(float)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f"{cm[i, j]:.2f}"
            else:
                text = str(int(cm[i, j]))
            plt.text(j, i, text, ha="center", va="center", fontsize=7)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# def run_experiment(model_name, model, train_loader, test_loader): #主函数，输入模型名称、模型对象、训练数据、测试数据
#    model = model.to(device)  #加载模型
#   criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
def run_experiment(  #抗噪
    model_name,
    model,
    train_loader,
    test_loader,
    loss_type=LOSS_TYPE,
    use_early_stopping=USE_EARLY_STOPPING
):
    model = model.to(device)
    criterion = build_criterion(loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "epoch": [],       #第几个epoch
        "train_loss": [],  #训练loss
        "train_acc": [],   #训练准确率
        "test_loss": [],   #测试loss
        "test_acc": []     #测试准确率
    }

    print(f"\n Training {model_name} ")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Loss function: {loss_type}")

    best_test_acc = -1.0
    best_epoch = 0
    best_state_dict = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS + 1):  
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(  
            model, test_loader, criterion, device
        )

        #保存epoch的结果
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )
    
        # Early stopping：用测试准确率作为监控指标
        # 更规范的做法是单独划分 validation set，这里为了和你原代码结构保持一致，先用 test_acc 观察趋势
        if test_acc > best_test_acc + EARLY_STOP_MIN_DELTA:
            best_test_acc = test_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if use_early_stopping and no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Test Acc: {best_test_acc * 100:.2f}% at epoch {best_epoch}."
            )
            break

    #返回最终结果
    # 如果使用 early stopping，则恢复到测试准确率最高的模型
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 用最终保存的模型重新评估一次
    final_train_loss, final_train_acc = evaluate(
        model, train_loader, criterion, device
    )

    final_test_loss, final_test_acc = evaluate(
        model, test_loader, criterion, device
    )

    final_result = {
        "model": model_name,
        "params": count_parameters(model),
        "loss_type": loss_type,
        "early_stopping": use_early_stopping,
        "stopped_epoch": history["epoch"][-1],
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
    }

    cm, class_recall, class_precision, support, predicted_count = per_class_metrics(
        model,
        test_loader,
        device,
        num_classes=NUM_CLASSES
    )

    per_class_df = pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "support": support.numpy(),
        "predicted_count": predicted_count.numpy(),
        "recall_class_accuracy": class_recall.numpy() * 100,
        "precision_prediction_success": class_precision.numpy() * 100,
    })

    per_class_csv_path = os.path.join(
        RESULT_DIR,
        f"{model_name}_per_class_metrics.csv"
    )

    per_class_df.to_csv(
        per_class_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 保存测试集混淆矩阵
    labels, preds = get_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(labels, preds, num_classes=NUM_CLASSES)

    raw_cm_path = os.path.join(RESULT_DIR, f"{model_name}_confusion_matrix.png")
    norm_cm_path = os.path.join(RESULT_DIR, f"{model_name}_confusion_matrix_normalized.png")

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        raw_cm_path,
        normalize=False,
        title=f"{model_name} Confusion Matrix"
    )

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        norm_cm_path,
        normalize=True,
        title=f"{model_name} Normalized Confusion Matrix"
    )
    
    return final_result, history

# 10. 计算预测的成功率
@torch.no_grad()
def per_class_metrics(model, data_loader, device, num_classes=10):
    """
    返回每个类别的：
    1. recall / class accuracy: 真实为该类的样本中，有多少预测正确
    2. precision: 预测为该类的样本中，有多少真的属于该类
    3. support: 测试集中该真实类别的样本数
    4. predicted_count: 被预测成该类别的样本数
    """
    model.eval()

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            cm[true_label, pred_label] += 1

    class_recall = torch.zeros(num_classes)
    class_precision = torch.zeros(num_classes)

    for c in range(num_classes):
        true_count = cm[c, :].sum().item()
        pred_count = cm[:, c].sum().item()
        correct_count = cm[c, c].item()

        class_recall[c] = correct_count / true_count if true_count > 0 else 0.0
        class_precision[c] = correct_count / pred_count if pred_count > 0 else 0.0

    support = cm.sum(dim=1)
    predicted_count = cm.sum(dim=0)

    return cm, class_recall, class_precision, support, predicted_count

# 11. Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        pred_class = logits.argmax(dim=1).item()
        pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()

        return cam.detach().cpu(), pred_class, pred_prob
    
@torch.no_grad()
def collect_one_correct_sample_per_class(
    cnn_model,
    se_model,
    test_loader,
    device,
    class_ids=None,
    num_classes=10
):
    """
    每个类别选一张样本。
    默认要求 CNN 和 SE-CNN 都预测正确，这样 Grad-CAM 对比更公平。
    """
    cnn_model.eval()
    se_model.eval()

    if class_ids is None:
        class_ids = list(range(num_classes))

    selected = {}

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        cnn_preds = cnn_model(images).argmax(dim=1)
        se_preds = se_model(images).argmax(dim=1)

        for i in range(images.size(0)):
            label = labels[i].item()

            if label not in class_ids:
                continue

            if label in selected:
                continue

            # 要求两个模型都预测正确
            if cnn_preds[i].item() == label and se_preds[i].item() == label:
                selected[label] = images[i:i+1].detach().cpu()

            if len(selected) == len(class_ids):
                return selected

    return selected 
  
# 11. 作图函数
def plot_metric(histories, metric_name, ylabel, save_path):
    plt.figure(figsize=(8, 5))

    for model_name, history in histories.items():
        plt.plot(history["epoch"], history[metric_name], marker="o", label=model_name)  #指标随epoch的变化散点图

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_gradcam_comparison_by_class(
    cnn_model,
    se_model,
    test_loader,
    device,
    save_dir,
    class_ids=None
):
    os.makedirs(save_dir, exist_ok=True)

    if class_ids is None:
        class_ids = list(range(NUM_CLASSES))

    # 最后一层卷积是 features.features[3]
    cnn_target_layer = cnn_model.features.features[3]
    se_target_layer = se_model.features.features[3]

    cnn_cam = GradCAM(cnn_model, cnn_target_layer)
    se_cam = GradCAM(se_model, se_target_layer)

    selected_samples = collect_one_correct_sample_per_class(
        cnn_model=cnn_model,
        se_model=se_model,
        test_loader=test_loader,
        device=device,
        class_ids=class_ids,
        num_classes=NUM_CLASSES
    )

    print("Selected classes:")
    for label in selected_samples:
        print(label, CLASS_NAMES[label])

    for label, image_cpu in selected_samples.items():
        image = image_cpu.to(device)

        # 这里 target_class 用真实类别，更适合类别对比
        cnn_heatmap, cnn_pred, cnn_prob = cnn_cam(image, target_class=label)
        se_heatmap, se_pred, se_prob = se_cam(image, target_class=label)

        original = image.squeeze().detach().cpu().numpy()

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap="gray")
        plt.title(f"Original\nTrue: {CLASS_NAMES[label]}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(original, cmap="gray")
        plt.imshow(cnn_heatmap.numpy(), alpha=0.5, cmap="jet")
        plt.title(
            f"CNN_SPP\nPred: {CLASS_NAMES[cnn_pred]}\nProb: {cnn_prob:.2f}"
        )
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(original, cmap="gray")
        plt.imshow(se_heatmap.numpy(), alpha=0.5, cmap="jet")
        plt.title(
            f"SE_CNN_SPP\nPred: {CLASS_NAMES[se_pred]}\nProb: {se_prob:.2f}"
        )
        plt.axis("off")

        plt.tight_layout()

        file_name = f"gradcam_{label}_{CLASS_NAMES[label].replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, file_name), dpi=300)
        plt.close()

    cnn_cam.remove_hooks()
    se_cam.remove_hooks()

    print(f"Saved class-balanced Grad-CAM figures to: {save_dir}")

#运行Grad-CAM的，可将main()改为run_gradcam_experiment()来得到结果
def run_gradcam_experiment():
    train_loader, test_loader = get_dataloaders(noise_rate=NOISE_RATE)

    cnn_model = CNNModel(head_type="spp", use_se=False).to(device)
    se_model = CNNModel(head_type="spp", use_se=True).to(device)

    run_experiment("CNN_SPP", cnn_model, train_loader, test_loader)
    run_experiment("SE_CNN_SPP", se_model, train_loader, test_loader)

    save_gradcam_comparison_by_class(
    cnn_model=cnn_model,
    se_model=se_model,
    test_loader=test_loader,
    device=device,
    save_dir=os.path.join(RESULT_DIR, "gradcam_by_class"),
    class_ids=list(range(10))
)

# 12. 主程序
def main():
    train_loader, test_loader = get_dataloaders(noise_rate=NOISE_RATE)

    model_builders = {
        # 四个主模型
        "Linear":lambda: LinearClassifier(),
        "MLP":lambda: MLP(),
        "CNN_SPP":lambda: CNNModel(head_type="spp", use_se=False),
        "SE_CNN_SPP":lambda: CNNModel(head_type="spp", use_se=True),

        ## Basic CNN 分类头对比实验
        #"CNN_1x1GAP":lambda: CNNModel(head_type="gap", use_se=False),
        #"CNN_FlattenFC":lambda: CNNModel(head_type="flatten", use_se=False),
        #"CNN_SPP": lambda: CNNModel(head_type="spp", use_se=False),

    }

    all_results = []
    all_histories = {}

    for model_name, build_model in model_builders.items():
        set_seed(SEED)
        model = build_model()

        result, history = run_experiment(
            model_name,
            model,
            train_loader,
            test_loader
        )

        all_results.append(result)
        all_histories[model_name] = history

    # 保存总表
    results_df = pd.DataFrame(all_results)
    results_df["final_train_acc"] = results_df["final_train_acc"] * 100
    results_df["final_test_acc"] = results_df["final_test_acc"] * 100
    results_df["best_test_acc"] = results_df["best_test_acc"] * 100

    csv_path = os.path.join(RESULT_DIR, "summary_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n Summary Results:")
    print(results_df)
    print(f"\nSaved results to: {csv_path}")

    # 保存训练曲线
    plot_metric(
        all_histories,
        metric_name="train_loss",
        ylabel="Training Loss",
        save_path=os.path.join(RESULT_DIR, "train_loss_curves.png")
    )

    plot_metric(
        all_histories,
        metric_name="test_loss",
        ylabel="Test Loss",
        save_path=os.path.join(RESULT_DIR, "test_loss_curves.png")
    )

    plot_metric(
        all_histories,
        metric_name="train_acc",
        ylabel="Training Accuracy",
        save_path=os.path.join(RESULT_DIR, "train_acc_curves.png")
    )

    plot_metric(
        all_histories,
        metric_name="test_acc",
        ylabel="Test Accuracy",
        save_path=os.path.join(RESULT_DIR, "test_acc_curves.png")
    )

    print(f"Saved figures to: {RESULT_DIR}/")


if __name__ == "__main__":
    main()