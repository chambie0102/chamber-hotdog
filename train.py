"""Hot Dog / Not Hot Dog classifier — Chamber GPU training job.

Uses Food-101 dataset (hotdog class vs balanced sample of other classes).
All hyperparams via environment variables for iteration without rebuild.
"""
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import vit_b_16, ViT_B_16_Weights
from collections import defaultdict

# ── Env Vars ──────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "3e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.0"))
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "false").lower() == "true"
AUGMENTATION = os.environ.get("AUGMENTATION", "none")
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chamber-hotdog")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jasonong-chamberai")
SEED = int(os.environ.get("SEED", "42"))

# ── Reproducibility ───────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── W&B Init ──────────────────────────────────────────────
import wandb
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
    "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "epochs": EPOCHS,
    "dropout": DROPOUT, "class_weights": USE_CLASS_WEIGHTS,
    "augmentation": AUGMENTATION, "warmup": WARMUP_EPOCHS,
    "model": "vit_b_16", "dataset": "food-101-hotdog-binary",
})

# ── Device ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Transforms ────────────────────────────────────────────
base_transform = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.Lambda(lambda x: x.convert("RGB")),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

aug_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.Lambda(lambda x: x.convert("RGB")),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_transform = aug_transform if AUGMENTATION != "none" else base_transform
val_transform = base_transform


# ── Dataset: Food-101 → Binary (hotdog vs not) ───────────
class HotDogBinaryDataset(Dataset):
    """Wraps a HuggingFace Food-101 split into hotdog (1) vs not-hotdog (0)."""

    def __init__(self, hf_dataset, transform, balance=True):
        self.transform = transform
        self.samples = []  # list of (image, label)

        # Food-101 class index for hot_dog
        label_names = hf_dataset.features["label"].names
        hotdog_idx = label_names.index("hot_dog")

        hotdog_items = []
        other_items = []

        for item in hf_dataset:
            if item["label"] == hotdog_idx:
                hotdog_items.append((item["image"], 1))
            else:
                other_items.append((item["image"], 0))

        print(f"  Raw: {len(hotdog_items)} hotdog, {len(other_items)} not-hotdog")

        if balance:
            # Balance: sample same number of not-hotdog as hotdog
            random.shuffle(other_items)
            other_items = other_items[:len(hotdog_items)]
            print(f"  Balanced: {len(hotdog_items)} hotdog, {len(other_items)} not-hotdog")

        self.samples = hotdog_items + other_items
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = self.transform(image)
        return image, label


print("Loading Food-101 dataset...")
from datasets import load_dataset
food101 = load_dataset("ethz/food-101")

print("Building training set...")
train_ds = HotDogBinaryDataset(food101["train"], train_transform, balance=True)
print("Building validation set...")
val_ds = HotDogBinaryDataset(food101["validation"], val_transform, balance=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

# ── Class Weights ─────────────────────────────────────────
if USE_CLASS_WEIGHTS:
    label_counts = defaultdict(int)
    for _, label in train_ds.samples:
        label_counts[label] += 1
    total = sum(label_counts.values())
    weights = torch.tensor([total / (2 * label_counts[i]) for i in range(2)],
                           dtype=torch.float32).to(device)
    print(f"Class weights: {weights.tolist()}")
else:
    weights = None

# ── Model ─────────────────────────────────────────────────
print("Loading ViT-B/16 pretrained model...")
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
if DROPOUT > 0:
    model.heads = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(768, 2))
else:
    model.heads = nn.Linear(768, 2)
model = model.to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {param_count:,}")

# ── Loss + Optimizer ──────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# ── Warmup + Cosine Scheduler ────────────────────────────
if WARMUP_EPOCHS > 0:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training Loop ─────────────────────────────────────────
best_acc = 0.0
print(f"\n{'='*60}")
print(f"Starting training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                  f"loss={loss.item():.4f}")

    train_acc = 100.0 * correct / total

    # LR scheduling
    if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    # ── Validation ────────────────────────────────────────
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_acc = 100.0 * val_correct / val_total

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    class_names = {0: "not_hotdog", 1: "hotdog"}
    per_class = {class_names[c]: 100.0 * class_correct[c] / class_total[c]
                 for c in class_total}

    # Log to W&B
    log = {
        "epoch": epoch + 1,
        "train_loss": running_loss / len(train_loader),
        "train_accuracy": train_acc,
        "val_loss": val_loss / len(val_loader),
        "val_accuracy": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    }
    for name, acc in per_class.items():
        log[f"{name}_accuracy"] = acc
    wandb.log(log)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    for name, acc in per_class.items():
        print(f"  {name}: {acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        print(f"  💾 New best: {best_acc:.2f}%")

print(f"\n{'='*60}")
print(f"Training complete! Best val accuracy: {best_acc:.2f}%")
print(f"Per-class final: {per_class}")
print(f"{'='*60}")

wandb.log({"best_val_accuracy": best_acc})
wandb.finish()
