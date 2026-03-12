"""Hot Dog / Not Hot Dog — Inference Script.

Usage:
    python inference.py <image_path_or_url>
    python inference.py hotdog.jpg
    python inference.py https://example.com/food.jpg

Loads the best model from W&B and classifies the image.
"""
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import requests
from io import BytesIO

# ── Config ────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pth")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chamber-hotdog")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jasonong-chamberai")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "NOT HOT DOG 🚫", 1: "HOT DOG 🌭"}


def load_model(model_path=None):
    """Load the trained model. Downloads from W&B if no local path."""
    model = vit_b_16(weights=None)
    model.heads = nn.Linear(768, 2)

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Downloading best model from W&B...")
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", order="-created_at")
        for run in runs:
            if run.state == "finished":
                try:
                    artifact_path = run.file("best_model.pth").download(replace=True)
                    state_dict = torch.load(artifact_path.name, map_location=DEVICE)
                    model.load_state_dict(state_dict)
                    print(f"Loaded from W&B run: {run.name}")
                    break
                except Exception as e:
                    print(f"Could not load from {run.name}: {e}")
                    continue

    model = model.to(DEVICE)
    model.eval()
    return model


def load_image(path_or_url):
    """Load image from file path or URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        print(f"Downloading image from {path_or_url}")
        response = requests.get(path_or_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        print(f"Loading image from {path_or_url}")
        image = Image.open(path_or_url)
    return image.convert("RGB")


def predict(model, image):
    """Run inference on a single image."""
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    label = CLASS_NAMES[predicted.item()]
    conf = confidence.item() * 100

    return label, conf, probabilities[0].cpu().tolist()


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path_or_url>")
        print("  python inference.py hotdog.jpg")
        print("  python inference.py https://example.com/food.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model(MODEL_PATH)
    image = load_image(image_path)
    label, confidence, probs = predict(model, image)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*40}")
    print(f"  Not hot dog: {probs[0]*100:.1f}%")
    print(f"  Hot dog:     {probs[1]*100:.1f}%")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
