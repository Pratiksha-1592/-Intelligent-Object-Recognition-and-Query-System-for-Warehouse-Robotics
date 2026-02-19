# ============================================================
# PART 2: Machine Learning Classifier - Warehouse Object Categories
# AI Research Intern - Technical Assessment
# Compatible with: Google Colab (CPU or GPU)
# Categories: FRAGILE | HEAVY | HAZARDOUS
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install & Import Dependencies
# ─────────────────────────────────────────────────────────────

!pip install torch torchvision scikit-learn matplotlib seaborn numpy Pillow -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.model_selection import train_test_split
import os, random, time, json
from collections import defaultdict

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Libraries loaded | Device: {DEVICE}")
print(f"   PyTorch: {torch.__version__}")

# ── Class Definitions ──
CLASSES     = ['FRAGILE', 'HEAVY', 'HAZARDOUS']
NUM_CLASSES = len(CLASSES)
CLASS_COLORS = {
    'FRAGILE':   '#E74C3C',   # Red
    'HEAVY':     '#3498DB',   # Blue
    'HAZARDOUS': '#F39C12',   # Orange
}


# ─────────────────────────────────────────────────────────────
# CELL 2: Synthetic Dataset Generator
# Creates labelled warehouse images for training
# ─────────────────────────────────────────────────────────────

class WarehouseSyntheticGenerator:
    """
    Generates synthetic 64x64 warehouse package images per class.

    FRAGILE  → Lighter colours, glass-jar shapes, red 'FRAGILE' stickers
    HEAVY    → Dark brown big boxes, metal drum shapes, weight labels
    HAZARDOUS → Yellow/orange barrels, biohazard markings, warning tape
    """

    IMG_SIZE = 64

    def __init__(self):
        self.rng = np.random.RandomState(SEED)

    def _base_canvas(self, bg_color):
        img = Image.new('RGB', (self.IMG_SIZE, self.IMG_SIZE), bg_color)
        draw = ImageDraw.Draw(img)
        return img, draw

    # ─── FRAGILE ────────────────────────────────────────────
    def make_fragile(self, idx):
        bg = (
            self.rng.randint(210, 240),
            self.rng.randint(210, 240),
            self.rng.randint(215, 245)
        )
        img, draw = self._base_canvas(bg)

        variant = idx % 4

        if variant == 0:
            # Cardboard box with FRAGILE sticker
            bx = self.rng.randint(6, 14)
            by = self.rng.randint(8, 16)
            bw = self.rng.randint(34, 46)
            bh = self.rng.randint(30, 42)
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(195,160,115), outline=(140,100,65), width=2)
            # tape lines
            draw.line([bx+bw//2, by, bx+bw//2, by+bh], fill=(175,140,95), width=2)
            draw.line([bx, by+bh//2, bx+bw, by+bh//2], fill=(175,140,95), width=2)
            # red sticker
            sx, sy = bx+4, by+bh//2-7
            draw.rectangle([sx, sy, sx+bw-8, sy+14], fill=(210,40,40))

        elif variant == 1:
            # Glass jar shape
            cx = self.IMG_SIZE // 2
            draw.ellipse([cx-10, 8, cx+10, 22], fill=(180,210,230), outline=(120,160,190), width=2)
            draw.rectangle([cx-13, 20, cx+13, 52], fill=(180,210,230), outline=(120,160,190), width=2)
            draw.ellipse([cx-13, 46, cx+13, 58], fill=(160,195,215), outline=(120,160,190), width=2)

        elif variant == 2:
            # Wrapped package with bow
            bx, by = 8, 10
            bw, bh = 46, 40
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(230,200,220), outline=(180,140,170), width=2)
            # ribbon
            draw.line([bx+bw//2, by, bx+bw//2, by+bh], fill=(200,80,80), width=3)
            draw.line([bx, by+bh//2, bx+bw, by+bh//2], fill=(200,80,80), width=3)
            # bow circles
            for dx in [-8, 8]:
                draw.ellipse([bx+bw//2+dx-6, by+bh//2-8, bx+bw//2+dx+6, by+bh//2+8],
                             fill=(220,100,100), outline=(180,60,60), width=1)

        else:
            # Bubble-wrapped item
            bx, by = 6, 8
            bw, bh = 50, 48
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(220,235,220), outline=(170,190,170), width=2)
            for bub_x in range(bx+4, bx+bw-4, 8):
                for bub_y in range(by+4, by+bh-4, 8):
                    draw.ellipse([bub_x, bub_y, bub_x+6, bub_y+6],
                                 fill=(200,220,245, 180), outline=(160,185,210), width=1)

        # Add noise
        arr = np.array(img).astype(np.float32)
        arr += self.rng.normal(0, self.rng.randint(3, 10), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # ─── HEAVY ──────────────────────────────────────────────
    def make_heavy(self, idx):
        bg = (
            self.rng.randint(55, 80),
            self.rng.randint(50, 75),
            self.rng.randint(50, 75)
        )
        img, draw = self._base_canvas(bg)

        variant = idx % 4

        if variant == 0:
            # Large dark wooden crate
            bx, by = 4, 6
            bw, bh = 54, 50
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(95, 65, 35), outline=(55, 35, 15), width=3)
            # wood slats
            for y in range(by+10, by+bh, 12):
                draw.line([bx, y, bx+bw, y], fill=(75, 48, 25), width=2)
            # metal corners
            for cx2, cy2 in [(bx, by), (bx+bw, by), (bx, by+bh), (bx+bw, by+bh)]:
                draw.rectangle([cx2-3, cy2-3, cx2+3, cy2+3], fill=(160,160,160))

        elif variant == 1:
            # Steel drum / barrel
            cx = self.IMG_SIZE // 2
            draw.ellipse([cx-18, 6, cx+18, 22], fill=(110,110,120), outline=(70,70,80), width=2)
            draw.rectangle([cx-18, 14, cx+18, 54], fill=(100,105,115), outline=(65,68,78), width=2)
            draw.ellipse([cx-18, 46, cx+18, 60], fill=(85,90,100), outline=(60,65,75), width=2)
            # hoops
            for y in [24, 36, 48]:
                draw.line([cx-18, y, cx+18, y], fill=(140,140,150), width=2)

        elif variant == 2:
            # Pallet with boxes stacked
            draw.rectangle([4, 48, 60, 58], fill=(130,95,55), outline=(90,60,25), width=2)
            for x in range(10, 58, 10):
                draw.line([x, 48, x, 58], fill=(105,72,38), width=1)
            # box stack
            draw.rectangle([8, 20, 56, 50], fill=(115,80,45), outline=(75,48,20), width=2)
            draw.rectangle([14, 6, 50, 22], fill=(125,88,50), outline=(85,55,25), width=2)

        else:
            # Concrete block / heavy slab
            bx, by = 5, 14
            bw, bh = 52, 36
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(120,118,115), outline=(80,78,75), width=2)
            # texture dots
            for _ in range(40):
                dx = self.rng.randint(bx+2, bx+bw-2)
                dy = self.rng.randint(by+2, by+bh-2)
                draw.ellipse([dx-1, dy-1, dx+1, dy+1], fill=(95,93,90))

        # Add noise
        arr = np.array(img).astype(np.float32)
        arr += self.rng.normal(0, self.rng.randint(4, 12), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # ─── HAZARDOUS ──────────────────────────────────────────
    def make_hazardous(self, idx):
        bg = (
            self.rng.randint(35, 55),
            self.rng.randint(35, 55),
            self.rng.randint(35, 55)
        )
        img, draw = self._base_canvas(bg)

        variant = idx % 4

        if variant == 0:
            # Yellow hazmat drum
            cx = self.IMG_SIZE // 2
            draw.ellipse([cx-16, 6, cx+16, 20], fill=(210,175,0), outline=(160,130,0), width=2)
            draw.rectangle([cx-16, 13, cx+16, 52], fill=(210,175,0), outline=(160,130,0), width=2)
            draw.ellipse([cx-16, 44, cx+16, 58], fill=(185,155,0), outline=(145,118,0), width=2)
            # hazard symbol
            draw.rectangle([cx-10, 22, cx+10, 40], fill=(0,0,0))
            draw.text((cx-5, 24), "!", fill=(255,220,0))

        elif variant == 1:
            # Orange warning container
            bx, by = 6, 8
            bw, bh = 50, 46
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(210,100,15), outline=(160,65,5), width=2)
            # warning stripes
            stripe_w = 8
            for i in range(0, bw+bh, stripe_w*2):
                draw.polygon([
                    (bx+i, by), (bx+i+stripe_w, by),
                    (bx+i+stripe_w-bh, by+bh), (bx+i-bh, by+bh)
                ], fill=(25,25,25))
            draw.rectangle([bx, by, bx+bw, by+bh], outline=(140,55,0), width=2)

        elif variant == 2:
            # Biohazard canister (green)
            cx = self.IMG_SIZE // 2
            draw.ellipse([cx-15, 6, cx+15, 18], fill=(60,160,60), outline=(35,110,35), width=2)
            draw.rectangle([cx-14, 12, cx+14, 52], fill=(55,150,55), outline=(30,100,30), width=2)
            draw.ellipse([cx-14, 44, cx+14, 56], fill=(45,130,45), outline=(25,85,25), width=2)
            # biohazard rings
            draw.ellipse([cx-10, 22, cx+10, 42], outline=(20,80,20), width=2)
            for angle_pt in [(cx, 22), (cx-9, 38), (cx+9, 38)]:
                draw.ellipse([angle_pt[0]-4, angle_pt[1]-4,
                              angle_pt[0]+4, angle_pt[1]+4],
                             fill=(40,120,40))

        else:
            # Red flammable box
            bx, by = 5, 8
            bw, bh = 52, 46
            draw.rectangle([bx, by, bx+bw, by+bh], fill=(185,30,30), outline=(130,15,15), width=2)
            # flame symbol outline
            pts = [(cx:=bx+bw//2, by+8),
                   (cx+10, by+22), (cx+5, by+22),
                   (cx+8, by+38), (cx-8, by+28),
                   (cx-4, by+28), (cx-10, by+16)]
            draw.polygon(pts, fill=(250,170,30), outline=(220,120,0))

        # Add noise
        arr = np.array(img).astype(np.float32)
        arr += self.rng.normal(0, self.rng.randint(4, 12), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def generate_dataset(self, samples_per_class=300):
        """Generate full dataset: images + labels."""
        gen_funcs = [self.make_fragile, self.make_heavy, self.make_hazardous]
        images, labels = [], []

        print("🏗️  Generating synthetic warehouse dataset...")
        for class_idx, (label, fn) in enumerate(zip(CLASSES, gen_funcs)):
            for i in range(samples_per_class):
                img = fn(i)
                images.append(img)
                labels.append(class_idx)
            print(f"   ✅ {label}: {samples_per_class} images generated")

        return images, labels


# Generate the dataset
generator = WarehouseSyntheticGenerator()
all_images, all_labels = generator.generate_dataset(samples_per_class=300)
print(f"\n📊 Total dataset: {len(all_images)} images | {NUM_CLASSES} classes")

# Preview sample images
fig, axes = plt.subplots(3, 8, figsize=(18, 7))
fig.suptitle("Synthetic Warehouse Dataset — Sample Images per Class",
             fontsize=14, fontweight='bold')

for class_idx, class_name in enumerate(CLASSES):
    class_imgs = [img for img, lbl in zip(all_images, all_labels) if lbl == class_idx]
    for j in range(8):
        axes[class_idx, j].imshow(class_imgs[j * 5])
        axes[class_idx, j].axis('off')
        if j == 0:
            axes[class_idx, j].set_ylabel(class_name, fontsize=11,
                                           fontweight='bold',
                                           color=CLASS_COLORS[class_name],
                                           rotation=0, labelpad=55)

plt.tight_layout()
plt.show()
print("✅ Dataset preview shown!")


# ─────────────────────────────────────────────────────────────
# CELL 3: Dataset Class & Transforms
# ─────────────────────────────────────────────────────────────

class WarehouseDataset(Dataset):
    """
    PyTorch Dataset wrapper for the synthetic warehouse images.
    Applies train/val transforms with data augmentation for training.
    """

    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ──
IMG_RESIZE = 64

train_transform = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),   # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Train / Validation / Test split: 70 / 15 / 15 ──
indices = list(range(len(all_images)))
train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=all_labels, random_state=SEED)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.50, stratify=[all_labels[i] for i in temp_idx], random_state=SEED)

train_imgs  = [all_images[i] for i in train_idx]
train_lbls  = [all_labels[i] for i in train_idx]
val_imgs    = [all_images[i] for i in val_idx]
val_lbls    = [all_labels[i] for i in val_idx]
test_imgs   = [all_images[i] for i in test_idx]
test_lbls   = [all_labels[i] for i in test_idx]

train_ds = WarehouseDataset(train_imgs, train_lbls, train_transform)
val_ds   = WarehouseDataset(val_imgs,   val_lbls,   val_transform)
test_ds  = WarehouseDataset(test_imgs,  test_lbls,  val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

print(f"✅ Dataset split complete:")
print(f"   Train : {len(train_ds)} samples")
print(f"   Val   : {len(val_ds)} samples")
print(f"   Test  : {len(test_ds)} samples")


# ─────────────────────────────────────────────────────────────
# CELL 4: Model Architecture (Transfer Learning — MobileNetV2)
# ─────────────────────────────────────────────────────────────

class WarehouseClassifier(nn.Module):
    """
    Transfer Learning Model: MobileNetV2 backbone + custom classifier head.

    Architecture:
    - MobileNetV2 pretrained on ImageNet (frozen early layers)
    - Replace final classifier with:
        Linear(1280 → 256) → BatchNorm → ReLU → Dropout(0.4)
        Linear(256 → 64)   → ReLU → Dropout(0.2)
        Linear(64  → 3)    → output logits

    MobileNetV2 was chosen for:
    - Lightweight (3.4M params) — fast on Colab CPU
    - Strong feature extraction even with small datasets
    - Depthwise separable convolutions → efficient
    """

    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()

        # ── Load pretrained MobileNetV2 ──
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # ── Freeze early layers (keep pretrained features) ──
        if freeze_backbone:
            for i, (name, param) in enumerate(backbone.named_parameters()):
                if 'features.0' in name or 'features.1' in name or 'features.2' in name:
                    param.requires_grad = False  # Freeze first 3 blocks

        # ── Backbone (all convolutional layers) ──
        self.backbone = backbone.features   # Output: [B, 1280, 2, 2] for 64x64 input

        # ── Global Average Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)  # → [B, 1280, 1, 1]

        # ── Custom Classifier Head ──
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = WarehouseClassifier(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)

# ── Count parameters ──
total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model: MobileNetV2 + Custom Head")
print(f"   Total params    : {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")
print(f"   Frozen params   : {total_params - trainable_params:,}")


# ─────────────────────────────────────────────────────────────
# CELL 5: Training Loop
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Training Configuration ──
NUM_EPOCHS    = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # Label smoothing helps generalisation
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing: smoothly decays LR to 0
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

# ── Training History ──
history = defaultdict(list)

print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
print(f"   Optimizer  : AdamW  (lr={LR}, wd={WEIGHT_DECAY})")
print(f"   Scheduler  : CosineAnnealingLR")
print(f"   Loss       : CrossEntropy (label_smoothing=0.1)")
print(f"   Batch size : {train_loader.batch_size}")
print("-" * 60)

best_val_acc  = 0.0
best_model_state = None

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    elapsed = time.time() - t0
    flag = " ← best" if val_acc > best_val_acc else ""

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  |  "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  "
          f"({elapsed:.1f}s){flag}")

# Restore best model
model.load_state_dict(best_model_state)
print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")


# ─────────────────────────────────────────────────────────────
# CELL 6: Training Curves
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part 2: Training History — Warehouse Object Classifier",
             fontsize=14, fontweight='bold', color='navy')

epochs_x = range(1, NUM_EPOCHS + 1)

# Loss
axes[0].plot(epochs_x, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
axes[0].plot(epochs_x, history['val_loss'],   'r-o', markersize=3, label='Val Loss')
axes[0].set_title('Loss Curves', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(epochs_x, [a*100 for a in history['train_acc']], 'b-o', markersize=3, label='Train Acc')
axes[1].plot(epochs_x, [a*100 for a in history['val_acc']],   'r-o', markersize=3, label='Val Acc')
axes[1].set_title('Accuracy Curves', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Learning Rate
axes[2].plot(epochs_x, history['lr'], 'g-o', markersize=3, label='LR')
axes[2].set_title('Learning Rate Schedule', fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('LR')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ Training curves plotted!")


# ─────────────────────────────────────────────────────────────
# CELL 7: Evaluation on Test Set + Performance Report
# ─────────────────────────────────────────────────────────────

print("🔍 Evaluating on test set...")
test_loss, test_acc, test_preds, test_true = evaluate(model, test_loader, criterion, DEVICE)

# ── Metrics ──
accuracy  = accuracy_score(test_true, test_preds)
precision = precision_score(test_true, test_preds, average='weighted', zero_division=0)
recall    = recall_score(test_true, test_preds, average='weighted', zero_division=0)
f1        = f1_score(test_true, test_preds, average='weighted', zero_division=0)

print("\n" + "="*60)
print("  PERFORMANCE REPORT — TEST SET")
print("="*60)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%  (weighted)")
print(f"  Recall    : {recall*100:.2f}%  (weighted)")
print(f"  F1-Score  : {f1*100:.2f}%  (weighted)")
print(f"  Test Loss : {test_loss:.4f}")
print("="*60)

print("\nPer-Class Report:")
print(classification_report(test_true, test_preds,
                             target_names=CLASSES,
                             digits=4))


# ─────────────────────────────────────────────────────────────
# CELL 8: Confusion Matrix + Per-Class Bar Chart
# ─────────────────────────────────────────────────────────────

cm = confusion_matrix(test_true, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Row-normalised

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Part 2: Model Evaluation — Confusion Matrix & Per-Class Accuracy",
             fontsize=14, fontweight='bold', color='navy')

# ── Confusion Matrix ──
sns.heatmap(cm_norm, annot=True, fmt='.2%',
            xticklabels=CLASSES, yticklabels=CLASSES,
            cmap='Blues', linewidths=0.5, linecolor='grey',
            ax=axes[0], annot_kws={'size': 13})

# Overlay raw counts
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        axes[0].text(j + 0.5, i + 0.72,
                     f"({cm[i, j]})",
                     ha='center', va='center',
                     fontsize=9, color='grey')

axes[0].set_title('Confusion Matrix\n(normalised, raw count in brackets)',
                  fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontweight='bold')

# ── Per-Class Bar Chart ──
per_class_acc = cm_norm.diagonal()
bars = axes[1].bar(CLASSES, per_class_acc * 100,
                   color=[CLASS_COLORS[c] for c in CLASSES],
                   edgecolor='black', linewidth=1.2, width=0.5)

for bar, acc in zip(bars, per_class_acc):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{acc*100:.1f}%",
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

axes[1].set_title('Per-Class Accuracy', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
axes[1].set_ylim(0, 110)
axes[1].axhline(y=accuracy*100, color='navy', linestyle='--',
                linewidth=1.5, label=f'Overall: {accuracy*100:.1f}%')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ Confusion matrix and per-class accuracy plotted!")


# ─────────────────────────────────────────────────────────────
# CELL 9: Inference Demo on New Images
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(model, img_pil, transform, device):
    """
    Run inference on a single PIL image.
    Returns: predicted class, confidence scores, all probabilities.
    """
    model.eval()
    tensor = transform(img_pil).unsqueeze(0).to(device)   # Add batch dim
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(np.argmax(probs))
    return pred, probs[pred], probs


def show_inference_grid(model, generator, transform, device, n_per_class=4):
    """
    Generate fresh test images and display inference results.
    Shows prediction confidence bars per image.
    """
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class,
                             figsize=(n_per_class * 3.5, NUM_CLASSES * 3.5))
    fig.suptitle("Part 2: Inference Demo — Fresh Unseen Images",
                 fontsize=14, fontweight='bold', color='navy')

    gen_funcs = [generator.make_fragile, generator.make_heavy, generator.make_hazardous]

    for class_idx, (class_name, gen_fn) in enumerate(zip(CLASSES, gen_funcs)):
        for j in range(n_per_class):
            test_img  = gen_fn(idx=j + 100)   # Use high idx → fresh variants
            pred_cls, confidence, all_probs = predict_single(model, test_img,
                                                              transform, device)
            pred_name = CLASSES[pred_cls]
            correct   = (pred_cls == class_idx)

            ax = axes[class_idx, j]
            ax.imshow(test_img)

            # Border colour = green (correct) / red (wrong)
            border_color = '#27AE60' if correct else '#E74C3C'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            title = f"True: {class_name}\nPred: {pred_name}\n{confidence*100:.1f}%"
            ax.set_title(title, fontsize=8.5,
                         color='green' if correct else 'red',
                         fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


show_inference_grid(model, generator, val_transform, DEVICE, n_per_class=5)
print("✅ Inference demo complete!")


# ─────────────────────────────────────────────────────────────
# CELL 10: Confidence Histogram & Summary Dashboard
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_all_probs(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_true.extend(labels.numpy())
    return np.array(all_probs), np.array(all_true)

test_probs, test_true_np = get_all_probs(model, test_loader, DEVICE)
max_confs = test_probs.max(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part 2: Model Confidence Analysis",
             fontsize=14, fontweight='bold', color='navy')

# ── Confidence distribution ──
axes[0].hist(max_confs * 100, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(max_confs.mean() * 100, color='red', linestyle='--',
                label=f'Mean: {max_confs.mean()*100:.1f}%')
axes[0].set_title('Prediction Confidence Distribution', fontweight='bold')
axes[0].set_xlabel('Max Confidence (%)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ── Confidence by class ──
for class_idx, class_name in enumerate(CLASSES):
    mask = test_true_np == class_idx
    axes[1].hist(max_confs[mask] * 100, bins=20, alpha=0.65,
                 label=class_name, color=CLASS_COLORS[class_name], edgecolor='white')
axes[1].set_title('Confidence by Class', fontweight='bold')
axes[1].set_xlabel('Max Confidence (%)')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(alpha=0.3)

# ── Summary metrics table ──
metrics = {
    'Accuracy':     f"{accuracy*100:.2f}%",
    'Precision':    f"{precision*100:.2f}%",
    'Recall':       f"{recall*100:.2f}%",
    'F1-Score':     f"{f1*100:.2f}%",
    'Mean Conf':    f"{max_confs.mean()*100:.2f}%",
    'Val Accuracy': f"{best_val_acc*100:.2f}%",
}
axes[2].axis('off')
rows = list(metrics.items())
tbl = axes[2].table(cellText=rows,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center', loc='center',
                     colWidths=[0.55, 0.45])
tbl.auto_set_font_size(False)
tbl.set_fontsize(13)
tbl.scale(1.2, 2.5)
for j in range(2):
    tbl[0, j].set_facecolor('#2C3E50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    color = '#EBF5FB' if i % 2 == 0 else 'white'
    for j in range(2):
        tbl[i, j].set_facecolor(color)
axes[2].set_title('Summary Metrics', fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
print("✅ Analysis dashboard complete!")


# ─────────────────────────────────────────────────────────────
# CELL 11: Save Model Weights + Performance Report JSON
# ─────────────────────────────────────────────────────────────

os.makedirs('/content/results', exist_ok=True)

# ── Save model weights ──
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': CLASSES,
    'best_val_acc': best_val_acc,
    'test_acc': accuracy,
    'architecture': 'MobileNetV2 + Custom Head',
    'input_size': IMG_RESIZE,
    'num_classes': NUM_CLASSES,
}, '/content/results/warehouse_classifier.pth')

# ── Save performance report ──
report = {
    'architecture': 'MobileNetV2 Transfer Learning',
    'classes': CLASSES,
    'dataset_size': len(all_images),
    'train_val_test_split': '70/15/15',
    'epochs': NUM_EPOCHS,
    'best_val_accuracy': round(best_val_acc, 4),
    'test_metrics': {
        'accuracy':  round(accuracy,  4),
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1_score':  round(f1,        4),
        'test_loss': round(test_loss, 4),
    },
    'per_class_accuracy': {
        CLASSES[i]: round(float(cm_norm[i, i]), 4) for i in range(NUM_CLASSES)
    },
    'confusion_matrix': cm.tolist(),
    'training_history': {
        'final_train_acc': round(history['train_acc'][-1], 4),
        'final_val_acc':   round(history['val_acc'][-1],   4),
        'final_train_loss': round(history['train_loss'][-1], 4),
        'final_val_loss':   round(history['val_loss'][-1],   4),
    }
}

with open('/content/results/performance_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("✅ Saved:")
print("   /content/results/warehouse_classifier.pth  (model weights)")
print("   /content/results/performance_report.json   (metrics report)")

# ── Download ──
from google.colab import files
import zipfile

with zipfile.ZipFile('/content/Part2_ML_Results.zip', 'w') as zf:
    zf.write('/content/results/warehouse_classifier.pth', 'warehouse_classifier.pth')
    zf.write('/content/results/performance_report.json',  'performance_report.json')

files.download('/content/Part2_ML_Results.zip')
print("📦 Part2_ML_Results.zip downloaded!")


# ─────────────────────────────────────────────────────────────
# LIMITATIONS DISCUSSION (150-200 words)
# ─────────────────────────────────────────────────────────────
"""
PART 2 – LIMITATIONS (150-200 words)
═══════════════════════════════════════════════════════════════

1. SYNTHETIC DATA BIAS
   The classifier was trained entirely on procedurally generated
   images using programmatic drawing primitives. Real warehouse
   images exhibit far greater variation: motion blur, occlusion,
   irregular lighting, wear/tear on packaging, and overlapping
   objects. Performance on real images would likely be lower until
   fine-tuned on even a small real-world labelled sample.

2. SMALL IMAGE RESOLUTION (64×64)
   Fine-grained visual features such as small warning text or
   subtle label colours are lost at this resolution. Increasing
   to 224×224 (standard ImageNet resolution) would improve
   discrimination, at the cost of longer training time.

3. CLASS OVERLAP
   HEAVY and HAZARDOUS categories share similar container shapes
   (drums, barrels). The model may confuse dark metallic drums
   with hazardous yellow drums under poor lighting. Adding a
   dedicated texture branch or multi-label output would help.

4. NO UNCERTAINTY ESTIMATION
   The current softmax output is known to produce overconfident
   predictions. Replacing with a Bayesian head or Monte Carlo
   Dropout would yield calibrated confidence — critical for
   safety-sensitive warehouse robotics.
═══════════════════════════════════════════════════════════════
"""

print("\n" + "="*55)
print("  ✅ PART 2: MACHINE LEARNING MODULE COMPLETE!")
print("="*55)
print("\nNext step → Run Part 3: RAG System")
