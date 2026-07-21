import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .gradcam import explain as compute_explanation
from .models import CLASSES, DEVICE, MODEL_REGISTRY, is_available, predict_image

app = FastAPI(title="Leaf Classifier API")

app.add_middleware(
    CORSMiddleware,
    # Matches any localhost/127.0.0.1 port — Vite bumps to 5174, 5175, etc.
    # whenever the default port is already taken, so pinning one exact origin
    # kept breaking. This is dev-only; a deployed frontend would use a fixed origin.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASSES}


@app.get("/models")
def list_models():
    return [
        {"key": key, "label": cfg["label"], "available": is_available(key)}
        for key, cfg in MODEL_REGISTRY.items()
    ]


def _load_upload(model_key: str, raw: bytes) -> Image.Image:
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'.")
    if not is_available(model_key):
        raise HTTPException(
            status_code=409,
            detail=f"'{MODEL_REGISTRY[model_key]['label']}' has not been trained yet — no checkpoint found.",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return image


@app.post("/predict")
async def predict(model_key: str = Form(...), file: UploadFile = File(...)):
    image = _load_upload(model_key, await file.read())
    try:
        result = predict_image(model_key, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return {"model": model_key, **result}


@app.post("/explain")
async def explain(model_key: str = Form(...), file: UploadFile = File(...)):
    """Grad-CAM (+ CBAM attention map, for that one model) for the uploaded image."""
    image = _load_upload(model_key, await file.read())
    try:
        result = compute_explanation(model_key, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    return {"model": model_key, **result}
