"""Model architectures and registry for the leaf-classifier API.

These class definitions are intentionally standalone copies of the ones in
01-06 notebooks (not imports) — the notebooks are the training/experimentation
source of truth; this module is what production inference loads. If you change
an architecture in a notebook, mirror the change here too.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

CLASSES     = sorted(d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir())
NUM_CLASSES = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE      = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def preprocess(pil_image):
    """PIL RGB image -> (1, 3, 224, 224) tensor, ready for model(...)."""
    array = np.array(pil_image.convert("RGB"))
    tensor = eval_transform(image=array)["image"]
    return tensor.unsqueeze(0)


# ── Custom 6-Conv CNN (from 02_train_cnn.ipynb) ──────────────────────────────
class Custom6CNN(nn.Module):
    def __init__(self, num_classes, p_drop=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,    32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,   64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool          = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout       = nn.Dropout(p_drop)
        self.fc1           = nn.Linear(1024, 1024)
        self.fc2           = nn.Linear(1024, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x   = self.adaptive_pool(x).view(x.size(0), -1)
        emb = F.relu(self.fc1(x))
        logits = self.fc2(self.dropout(emb))
        if return_embedding:
            return logits, emb
        return logits


def build_custom_cnn(num_classes):
    return Custom6CNN(num_classes)


# ── Generic classifier-head swap (ConvNeXt-Tiny / DenseNet121 / MobileNetV3-Small) ──
def replace_classifier(base_model, attr_name, num_classes):
    module = getattr(base_model, attr_name)
    if isinstance(module, nn.Linear):
        setattr(base_model, attr_name, nn.Linear(module.in_features, num_classes))
    elif isinstance(module, nn.Sequential):
        for idx in reversed(range(len(module))):
            if isinstance(module[idx], nn.Linear):
                module[idx] = nn.Linear(module[idx].in_features, num_classes)
                break
    else:
        raise TypeError(f"Unexpected classifier type: {type(module)}")
    return base_model


def build_convnext_tiny(num_classes):
    # weights=None: at inference time we load our own fine-tuned checkpoint,
    # no need to also fetch ImageNet weights first.
    m = models.convnext_tiny(weights=None)
    return replace_classifier(m, "classifier", num_classes)


def build_densenet121(num_classes):
    m = models.densenet121(weights=None)
    return replace_classifier(m, "classifier", num_classes)


def build_mobilenetv3_small(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    return replace_classifier(m, "classifier", num_classes)


# ── EfficientNet-B0 + CBAM (from 05_train_efficientnet_b0_cbam.ipynb) ────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes, p_drop=0.2):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features   = backbone.features
        self.cbam       = CBAM(channels=1280)
        self.avgpool    = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_drop, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_efficientnet_b0_cbam(num_classes):
    return EfficientNetB0_CBAM(num_classes)


# ── Registry ──────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "custom_cnn": {
        "label": "Custom 6-Conv CNN",
        "build_fn": build_custom_cnn,
        "checkpoint_path": ROOT / "cnn_results" / "best_custom_6cnn_model.pth",
    },
    "custom_cnn_tuned": {
        "label": "Custom 6-Conv CNN (Optuna-Tuned)",
        "build_fn": build_custom_cnn,
        "checkpoint_path": ROOT / "optuna_hy_tunining" / "results" / "best_tuned_custom6cnn.pth",
    },
    "convnext_tiny": {
        "label": "ConvNeXt-Tiny",
        "build_fn": build_convnext_tiny,
        "checkpoint_path": ROOT / "convnext_tiny_results" / "best_convnext_tiny.pth",
    },
    "densenet121": {
        "label": "DenseNet121",
        "build_fn": build_densenet121,
        "checkpoint_path": ROOT / "densenet121_results" / "best_densenet121.pth",
    },
    "efficientnet_b0_cbam": {
        "label": "EfficientNet-B0 + CBAM",
        "build_fn": build_efficientnet_b0_cbam,
        "checkpoint_path": ROOT / "efficientnet_b0_cbam_results" / "best_efficientnet_b0_cbam.pth",
    },
    "mobilenetv3_small": {
        "label": "MobileNetV3-Small",
        "build_fn": build_mobilenetv3_small,
        "checkpoint_path": ROOT / "mobilenetv3_small_results" / "best_mobilenetv3_small.pth",
    },
}

_MODEL_CACHE = {}


def is_available(model_key):
    return MODEL_REGISTRY[model_key]["checkpoint_path"].exists()


def get_model(model_key):
    """Lazily build + load + cache a model. Raises FileNotFoundError if untrained."""
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    cfg = MODEL_REGISTRY[model_key]
    ckpt_path = cfg["checkpoint_path"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint for '{model_key}' at {ckpt_path}")

    model = cfg["build_fn"](NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[model_key] = model
    return model


@torch.no_grad()
def predict_image(model_key, pil_image):
    """Returns dict: {"predicted_class": str, "confidence": float, "probabilities": {cls: prob}}"""
    model  = get_model(model_key)
    tensor = preprocess(pil_image).to(DEVICE)

    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    prob_map = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    predicted = max(prob_map, key=prob_map.get)

    return {
        "predicted_class": predicted,
        "confidence": prob_map[predicted],
        "probabilities": prob_map,
    }
