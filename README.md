# Leaf Species Classifier

Classifies leaf photos into 8 species (Apple, Berry, Fig, Guava, Orange, Palm,
Persimmon, Tomato) using five different CNN architectures, with a web app to
try predictions interactively.

## Dataset

Source images: `archive(2)/New_data/` — 3,588 field photos of leaves, one
folder per class.

| Class | Images |
|---|---|
| Orange | 547 |
| Persimmon | 527 |
| Guava | 520 |
| Apple | 519 |
| Fig | 508 |
| Palm | 468 |
| Berry | 340 |
| Tomato | 159 |

Two things about this dataset shaped the preprocessing:

- **Class imbalance** — Tomato has less than a third of the images of the
  largest class. Handled with a class-weighted loss (`data/class_weights.json`,
  computed from the train split only).
- **Near-duplicate crops** — most "images" are actually multiple crops from the
  same original photograph (filenames like `IMG_<date>_<time>_1.jpg`). A random
  train/val/test split would leak crops of the same photo across splits, so the
  split is **grouped by source photo ID**, not by individual file.

See `01_preprocessing.ipynb` for the full analysis (brightness/blur checks, exact
duplicate detection, split logic) and the reasoning behind each decision.

## Project structure

```
Task04/
├── archive(2)/New_data/          raw dataset (not in git — see .gitignore)
├── data/                         train/val/test split + class_weights.json (generated)
├── 01_preprocessing.ipynb        EDA, grouped split, class weights, augmentation pipeline
├── 02_train_cnn.ipynb            Custom 6-Conv CNN (trained from scratch)
├── 03_train_convnext_tiny.ipynb  ConvNeXt-Tiny (pretrained, fine-tuned)
├── 04_train_densenet121.ipynb    DenseNet121 (pretrained, fine-tuned)
├── 05_train_efficientnet_b0_cbam.ipynb   EfficientNet-B0 + CBAM attention
├── 06_train_mobilenetv3_small.ipynb      MobileNetV3-Small (pretrained, fine-tuned)
├── *_results/                    per-model checkpoint, plots, reports, embeddings (generated)
├── webapp/
│   ├── backend/                  FastAPI inference server
│   └── frontend/                 React (Vite) upload + predict UI
├── requirements.txt
└── .venv/                        Python virtual environment (not in git)
```

## Setup

```bash
cd Task04
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires an NVIDIA GPU + CUDA-capable PyTorch for training in reasonable time
(all 5 notebooks assume `torch.cuda.is_available()`).

## Running the pipeline

1. **`01_preprocessing.ipynb`** — run first. Analyzes the raw dataset, builds
   the grouped train/val/test split into `data/`, and computes class weights.
2. **`02`–`06`** — train each model independently (any order). Each notebook is
   self-contained: same data pipeline, same augmentation, own architecture. Each
   saves to its own `*_results/` folder:
   - best checkpoint (`.pth`)
   - training/validation accuracy & loss curves
   - confusion matrix, classification report, per-class precision/recall/F1
   - ROC curve and precision-recall curve (one-vs-rest, per class)
   - `train/val/test_embs.npy` + `*_y.npy` — penultimate-layer feature vectors,
     saved for a future mRMR + SVM hybrid stage (not yet implemented)

## Results

Test-set accuracy (543 held-out images, grouped split):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom 6-Conv CNN | 96.3% | 0.959 |
| MobileNetV3-Small | 99.8% | 0.997 |
| ConvNeXt-Tiny | 100% | 1.000 |
| DenseNet121 | 100% | 1.000 |
| EfficientNet-B0 + CBAM | 100% | 1.000 |

**Read the 100% numbers with caution, not as a finish line.** The grouped split
prevents exact photo-crop leakage across train/test, but each class still comes
from a fairly small number of distinct photo *sessions* — pretrained ImageNet
backbones are powerful enough that they may be partly keying off session-specific
cues (background, lighting, camera angle) rather than pure leaf morphology. The
real test is accuracy on leaves photographed on a different day; that hasn't
been done yet. Treat these numbers as "this pipeline works end-to-end," not as a
generalization guarantee.

## Web application

Upload a leaf photo, pick a model, get a prediction with a per-class confidence
chart. The backend auto-detects which of the 5 models have a trained checkpoint,
so it degrades gracefully if you haven't trained all of them.

```bash
# Terminal 1 — backend (FastAPI, port 8000)
cd Task04
source .venv/bin/activate
uvicorn webapp.backend.app.main:app --reload --port 8000

# Terminal 2 — frontend (React/Vite)
cd Task04/webapp/frontend
npm install   # first time only
npm run dev
```

Then open the URL Vite prints (typically `http://localhost:5173`).

### Testing the web application

With the backend running (`http://localhost:8000`), verify it end-to-end from
the command line before (or instead of) clicking through the UI:

```bash
# 1. Backend is up and which device it's running inference on
curl http://localhost:8000/health

# 2. Which models have a trained checkpoint available
curl http://localhost:8000/models

# 3. Predict on a sample image (swap the path for any leaf photo)
curl -X POST http://localhost:8000/predict \
  -F "model_key=custom_cnn" \
  -F "file=@data/test/Apple/IMG_20240710_161157.jpg"

# 4. Grad-CAM (+ CBAM attention, for efficientnet_b0_cbam only) for the same image
curl -X POST http://localhost:8000/explain \
  -F "model_key=custom_cnn" \
  -F "file=@data/test/Apple/IMG_20240710_161157.jpg"
```

`model_key` is one of: `custom_cnn`, `custom_cnn_tuned`, `convnext_tiny`,
`densenet121`, `efficientnet_b0_cbam`, `mobilenetv3_small` — matches the `key`
field from step 2's `/models` response. `/predict` returns the predicted class,
confidence, and per-class probabilities; `/explain` returns the same fields'
worth of image data as base64 data-URIs (`gradcam`, and `cbam_attention` when
not applicable to that model).

For the frontend, exercise it manually in the browser (upload a photo, switch
between models, toggle "AI Validation" to check Grad-CAM/CBAM render), and run
its lint/build checks:

```bash
cd webapp/frontend
npm run lint    # oxlint
npm run build   # production build — catches type/import errors dev mode won't
```

## Planned next step

Use the saved `*_embs.npy` feature vectors for an mRMR + SVM hybrid

