"""
Explainable CNN for Multi-Class Brain Tumor Classification from MRI
Grad-CAM + SHAP + LIME dashboard (research / education only - NOT a medical device).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [APP_DIR / "models", APP_DIR / "results", APP_DIR]

MODEL_FILE = "resnet50_finetuned.keras"
RESNET_RESULTS_FILE = "resnet50_test_results.json"
COMPARISON_FILE = "model_comparison_test_results.json"
HISTORY_FILE = "custom_cnn_history.json"
BASELINE_FILE = "custom_cnn_baseline.keras"

# Keras image_dataset_from_directory / flow_from_directory sort class folders
# alphabetically -> this is the DEFAULT assumption and must be confirmed.
DEFAULT_ORDER = ["glioma", "meningioma", "notumor", "pituitary"]
CANON = set(DEFAULT_ORDER)
DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}

PREPROC_LABELS = {
    "resnet": "ResNet50 preprocess_input (BGR, mean-subtracted)",
    "scale01": "Rescale to 0-1 (divide by 255)",
    "raw": "Raw 0-255 (model has its own preprocessing layer)",
}

GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

C_INK, C_PANEL, C_LINE = "#0d1a22", "#14252f", "#26404d"
C_MIST, C_MUTED = "#e7eff3", "#8fa6b2"
C_CYAN, C_AMBER, C_CORAL, C_BLUE = "#5fd6cf", "#f3b562", "#ef7b6c", "#5aa9ff"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap');
:root{--ink:#0d1a22;--panel:#14252f;--line:#26404d;--mist:#e7eff3;--muted:#8fa6b2;--cyan:#5fd6cf;--amber:#f3b562;--coral:#ef7b6c;}
html, body, .stApp, [class*="css"] {font-family:'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;}
.stApp{background:radial-gradient(1100px 480px at 88% -8%, rgba(95,214,207,.11), transparent 62%), var(--ink);}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','DM Sans',sans-serif !important;letter-spacing:-.01em;}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem;}
section[data-testid="stSidebar"]{background:#0a141b;border-right:1px solid var(--line);}

.hero{padding:.4rem 0 1.4rem 0;}
.hero h1{font-size:clamp(2.1rem,4.6vw,3.6rem);line-height:1.04;margin:0 0 .7rem 0;font-weight:700;color:var(--mist);}
.hero p{max-width:640px;color:var(--muted);font-size:1.08rem;line-height:1.55;margin:0 0 1.1rem 0;}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;}
.chip{border:1px solid var(--line);background:rgba(20,37,47,.7);color:var(--mist);padding:.28rem .8rem;border-radius:999px;font-size:.86rem;}
.chip b{color:var(--cyan);font-weight:600;}

.note{border-left:3px solid var(--amber);background:rgba(243,181,98,.07);padding:.75rem 1rem;border-radius:0 10px 10px 0;color:#f1dcb8;font-size:.92rem;line-height:1.5;margin:.4rem 0 1.4rem 0;}
.note.info{border-left-color:var(--cyan);background:rgba(95,214,207,.07);color:#cfeeed;}
.note.warn{border-left-color:var(--coral);background:rgba(239,123,108,.08);color:#f6d2cc;}

.pred{background:linear-gradient(180deg,#16303c,#14252f);border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.4rem;}
.pred .lab{color:var(--muted);font-size:.9rem;}
.pred .cls{font-family:'Bricolage Grotesque',sans-serif;font-size:2.5rem;font-weight:700;color:var(--mist);line-height:1.1;margin:.1rem 0 .35rem 0;}
.pred .sc{color:var(--cyan);font-weight:600;font-size:1.05rem;}
.pred .sub{color:var(--muted);font-size:.84rem;margin-top:.3rem;line-height:1.45;}

.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:.4rem 0 1rem 0;}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;}
.metric span{display:block;color:var(--muted);font-size:.82rem;margin-bottom:.15rem;}
.metric b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.55rem;color:var(--mist);font-weight:600;}

div[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line);}
div[data-testid="stImageCaption"]{color:var(--muted);}
button[data-baseweb="tab"]{font-weight:500;}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"]{color:#06222a !important;font-weight:600;border:0;}
.stButton button, button[data-testid="stBaseButton-secondary"]{border-radius:10px;}
div[data-testid="stFileUploader"] section{border:1px dashed var(--line);background:rgba(20,37,47,.55);border-radius:14px;}
.legend{display:flex;gap:1.1rem;flex-wrap:wrap;color:var(--muted);font-size:.86rem;margin:.2rem 0 .6rem 0;}
.legend i{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;margin-right:.4rem;vertical-align:-1px;}
.small{color:var(--muted);font-size:.86rem;line-height:1.5;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(s: str) -> None:
    """Render HTML collapsed to one line (blank/indented lines break Markdown)."""
    st.markdown(re.sub(r"\n\s*", "", s), unsafe_allow_html=True)


def note(text: str, kind: str = "") -> None:
    html(f'<div class="note {kind}">{text}</div>')


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def find_file(name: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False)
def read_json(path_str: str, mtime: float):
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_file(name: str):
    p = find_file(name)
    if p is None:
        return None, None
    try:
        return read_json(str(p), p.stat().st_mtime), p
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"{type(e).__name__}: {e}"}, p


# --------------------------------------------------------------------------- #
# Class-order helpers
# --------------------------------------------------------------------------- #
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def parse_order(text: str):
    parts = [norm_key(p) for p in text.split(",") if p.strip()]
    return parts if sorted(parts) == sorted(CANON) else None


def find_class_list(obj):
    """Look for evidence of the class order inside a loaded JSON object."""
    if isinstance(obj, dict):
        if len(obj) == 4 and all(isinstance(v, int) for v in obj.values()):
            keys = [norm_key(k) for k in obj]
            if set(keys) == CANON and sorted(obj.values()) == [0, 1, 2, 3]:
                return [norm_key(k) for k, _ in sorted(obj.items(), key=lambda kv: kv[1])]
        for v in obj.values():
            r = find_class_list(v)
            if r:
                return r
    elif isinstance(obj, list):
        if len(obj) == 4 and all(isinstance(v, str) for v in obj):
            keys = [norm_key(v) for v in obj]
            if set(keys) == CANON:
                return keys
        for v in obj:
            r = find_class_list(v)
            if r:
                return r
    return None


# --------------------------------------------------------------------------- #
# Model loading / inspection
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading ResNet50 model (first run only)...")
def load_model(path_str: str, mtime: float):
    import tensorflow as tf

    return tf.keras.models.load_model(path_str, compile=False)


def iter_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from iter_layers(layer)


def _out_shape(layer):
    try:
        s = layer.output.shape
    except Exception:  # noqa: BLE001
        try:
            s = layer.output_shape
        except Exception:  # noqa: BLE001
            return None
    if isinstance(s, list):
        s = s[0]
    return tuple(s)


def last_4d_layer(model):
    for layer in reversed(model.layers):
        s = _out_shape(layer)
        if s is not None and len(s) == 4:
            return layer
    return None


def detect_preprocessing(model):
    """Heuristic only: look for preprocessing layers inside the saved model."""
    for layer in iter_layers(model):
        cname, lname = layer.__class__.__name__.lower(), layer.name.lower()
        if cname in ("rescaling", "normalization") or "preprocess" in lname or "rescal" in lname:
            return "raw", f"found layer '{layer.name}' ({layer.__class__.__name__}) inside the model"
        if cname == "lambda":
            return "raw", f"found Lambda layer '{layer.name}' (may be preprocessing - check!)"
    return "resnet", "no preprocessing layer found inside the model; assuming ResNet50 preprocess_input"


def model_info(model) -> dict:
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    oshape = model.output_shape
    if isinstance(oshape, list):
        oshape = oshape[0]
    h = int(ishape[1] or 224)
    w = int(ishape[2] or 224)
    tl = last_4d_layer(model)
    return {
        "input_shape": tuple(ishape),
        "output_shape": tuple(oshape),
        "size": (h, w),
        "channels": ishape[3] if len(ishape) > 3 else None,
        "params": int(model.count_params()),
        "n_layers": len(model.layers),
        "target_layer": tl.name if tl is not None else None,
        "final_layer": f"{model.layers[-1].name} ({model.layers[-1].__class__.__name__})",
    }


# --------------------------------------------------------------------------- #
# Pre-processing and prediction
# --------------------------------------------------------------------------- #
def preprocess(batch, mode: str):
    x = np.array(batch, dtype=np.float32, copy=True)
    if mode == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input(x)
    if mode == "scale01":
        return x / 255.0
    return x


def to_scores(out) -> np.ndarray:
    """Return softmax-like scores. If the model already outputs probabilities keep them."""
    out = np.atleast_2d(np.asarray(out, dtype=np.float64))
    if np.all(out >= 0) and np.all(out <= 1.0001) and np.allclose(out.sum(1), 1.0, atol=1e-3):
        return out
    e = np.exp(out - out.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def predict_scores(model, batch, mode: str, batch_size: int = 32) -> np.ndarray:
    x = preprocess(batch, mode)
    return to_scores(model.predict(x, batch_size=batch_size, verbose=0))


def load_image(uploaded, size_hw):
    data = uploaded.getvalue()
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("The file is larger than 10 MB. Please upload a smaller MRI image.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("This file could not be read as an image. Upload a valid JPG, JPEG or PNG.")
    img = ImageOps.exif_transpose(img)
    if min(img.size) < 64:
        raise ValueError(f"Image is too small ({img.size[0]}x{img.size[1]} px). Minimum is 64 px per side.")
    if img.mode in ("I;16", "I", "F"):
        a = np.asarray(img, dtype=np.float64)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-9) * 255
        img = Image.fromarray(a.astype(np.uint8))
    img = img.convert("RGB")
    h, w = size_hw
    return np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Map utilities
# --------------------------------------------------------------------------- #
def norm01(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def resize_map(m: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    return np.asarray(Image.fromarray(np.asarray(m, dtype=np.float32)).resize((w, h), Image.BILINEAR))


def colorize(cam01: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    cm = matplotlib.colormaps[cmap]
    return (cm(cam01)[..., :3] * 255).astype(np.uint8)


def overlay_heat(img: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = colorize(cam01).astype(np.float32)
    return np.clip((1 - alpha) * img.astype(np.float32) + alpha * heat, 0, 255).astype(np.uint8)


def hex_rgb(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def overlay_signed(img: np.ndarray, m: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """Warm = supports the class, cool = argues against it."""
    m = np.asarray(m, dtype=np.float32)
    scale = np.percentile(np.abs(m), 99.5)
    n = np.clip(m / scale, -1, 1) if scale > 1e-12 else np.zeros_like(m)
    gray = np.repeat(img.astype(np.float32).mean(-1, keepdims=True), 3, -1) * 0.8
    color = np.where(n[..., None] >= 0, hex_rgb(C_CORAL), hex_rgb(C_BLUE))
    a = (np.abs(n) * strength)[..., None]
    return np.clip(gray * (1 - a) + color * a, 0, 255).astype(np.uint8)


def describe_map(m: np.ndarray, top_pct: int = 15):
    m = np.maximum(np.asarray(m, dtype=np.float32), 0)
    if m.max() <= 0:
        return None
    thr = max(np.percentile(m, 100 - top_pct), 1e-12)
    ys, xs = np.nonzero(m >= thr)
    cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
    v = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "central"
    h = "left" if cx < 0.4 else "right" if cx > 0.6 else "central"
    place = "central region" if v == h == "central" else "-".join(p for p in (v, h) if p != "central") + " area"
    return {"focus": place, "area_above_half_max_pct": round(float((m >= 0.5 * m.max()).mean() * 100), 1)}


def top_iou(a: np.ndarray, b: np.ndarray, pct: int = 15) -> float:
    a, b = np.maximum(a, 0), np.maximum(b, 0)
    if a.max() <= 0 or b.max() <= 0:
        return float("nan")
    ta = (a >= np.percentile(a, 100 - pct)) & (a > 0)
    tb = (b >= np.percentile(b, 100 - pct)) & (b > 0)
    u = (ta | tb).sum()
    return float((ta & tb).sum() / u) if u else float("nan")


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def _softmax_tail(model):
    """If the model ends in softmax, return how to read the logits instead
    (softmax saturates and can zero-out gradients at ~100% confidence)."""
    import tensorflow as tf

    last = model.layers[-1]
    act = getattr(last, "activation", None)
    is_softmax = getattr(act, "__name__", "") == "softmax"
    if isinstance(last, tf.keras.layers.Dense) and is_softmax:
        return "dense", last
    if isinstance(last, tf.keras.layers.Softmax) or (
        isinstance(last, tf.keras.layers.Activation) and is_softmax
    ):
        return "act", last
    return None, None


def gradcam(model, x_pre: np.ndarray, class_idx: int, layer) -> tuple[np.ndarray, dict]:
    """Returns (cam in [0,1] at feature-map resolution, info)."""
    import tensorflow as tf

    x = tf.convert_to_tensor(x_pre)
    kind, last = _softmax_tail(model)
    info = {"layer": layer.name, "score_source": "model output"}
    try:
        outs = [layer.output, model.output]
        if kind:
            outs.append(last.input)
        gm = tf.keras.Model(model.inputs, outs)
        with tf.GradientTape() as tape:
            res = gm(x, training=False)
            conv, preds = res[0], res[1]
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            if kind == "dense":
                logits = tf.matmul(res[2], tf.convert_to_tensor(last.kernel))
                if last.bias is not None:
                    logits = logits + tf.convert_to_tensor(last.bias)
                score = logits[:, class_idx]
                info["score_source"] = "pre-softmax logit"
            elif kind == "act":
                score = res[2][:, class_idx]
                info["score_source"] = "pre-softmax logit"
            else:
                score = preds[:, class_idx]
        grads = tape.gradient(score, conv)
    except Exception:  # noqa: BLE001  (fallback: manual forward pass for odd topologies)
        with tf.GradientTape() as tape:
            h, conv = x, None
            for lyr in model.layers:
                if isinstance(lyr, tf.keras.layers.InputLayer):
                    continue
                h = lyr(h, training=False)
                if lyr is layer:
                    conv = h
            score = h[:, class_idx]
        grads = tape.gradient(score, conv)
        info["score_source"] = "model output (fallback path)"
    if grads is None:
        raise RuntimeError("Gradients could not be computed for the selected layer.")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    if cam.max() <= 0:
        info["warning"] = "Grad-CAM is all zeros for this class (no positive evidence in this layer)."
        return np.zeros_like(cam), info
    return norm01(cam), info


# --------------------------------------------------------------------------- #
# SHAP (model-agnostic Partition explainer with an image masker)
# --------------------------------------------------------------------------- #
def run_shap(model, img: np.ndarray, mode: str, max_evals: int, batch_size: int = 32) -> dict:
    import shap

    def f(x):
        return predict_scores(model, x, mode, batch_size=batch_size)

    masker = shap.maskers.Image("blur(64,64)", img.shape)
    explainer = shap.Explainer(f, masker)
    sv = explainer(img[None].astype(np.float32), max_evals=int(max_evals), batch_size=batch_size)
    vals = np.asarray(sv.values[0])  # (H, W, C, n_classes)
    base = np.asarray(sv.base_values[0])
    return {"values": vals, "base": base, "max_evals": int(max_evals)}


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
def run_lime(model, img: np.ndarray, mode: str, class_idx: int, num_samples: int, kernel_size: int, top_k: int) -> dict:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    from skimage.segmentation import mark_boundaries

    def clf(images):
        return predict_scores(model, np.asarray(images), mode, batch_size=32)

    seg_fn = SegmentationAlgorithm("quickshift", kernel_size=int(kernel_size), max_dist=200, ratio=0.2, random_seed=42)
    explainer = lime_image.LimeImageExplainer(random_state=42)
    exp = explainer.explain_instance(
        img,
        clf,
        labels=(int(class_idx),),
        top_labels=None,
        hide_color=0,
        num_samples=int(num_samples),
        batch_size=32,
        segmentation_fn=seg_fn,
        random_seed=42
        
    )
    segs = exp.segments
    weights = sorted(exp.local_exp[int(class_idx)], key=lambda t: -abs(t[1]))[: int(top_k)]
    wmap = np.zeros(segs.shape, dtype=np.float32)
    lab = np.zeros(segs.shape, dtype=np.int32)
    for sid, w in weights:
        wmap[segs == sid] = w
        lab[segs == sid] = int(sid) + 1
    vis = overlay_signed(img, wmap, strength=0.7)
    vis = (mark_boundaries(vis / 255.0, lab, color=(0.93, 0.97, 1.0), mode="inner") * 255).astype(np.uint8)
    score = getattr(exp, "score", None)
    if isinstance(score, dict):
        score = score.get(int(class_idx))
    return {
        "vis": vis,
        "wmap": wmap,
        "n_segments": int(segs.max() + 1),
        "local_r2": float(score) if score is not None else None,
        "num_samples": int(num_samples),
    }


# --------------------------------------------------------------------------- #
# Groq (optional)
# --------------------------------------------------------------------------- #
def get_groq_client():
    """Returns (client | None, status). Status never contains the key."""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:  # noqa: BLE001  (no secrets file / no key)
        return None, "missing"
    if not isinstance(key, str) or not key.strip() or "PASTE_YOUR" in key:
        return None, "placeholder"
    try:
        from groq import Groq
    except ImportError:
        return None, "package"
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:  # noqa: BLE001
        return None, "invalid"
    return client, "ok"


GROQ_STATUS_TEXT = {
    "ok": "Groq key found. Text explanations are available.",
    "missing": "No Groq key found. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and add your key. Everything except the text explanation still works.",
    "placeholder": "secrets.toml still contains the placeholder. Replace it with your Groq key to enable text explanations.",
    "package": "The groq package is not installed. Run: pip install groq",
    "invalid": "The Groq client could not be created from the saved key.",
}


def groq_explain(client, model_name: str, payload: dict) -> tuple[str | None, str | None, str | None]:
    """Returns (text, used_model, error). Error strings never include the key."""
    import groq

    system = (
        "You help students and researchers interpret the output of a brain-MRI tumor classifier "
        "for a research demo. You receive numeric summaries only, not the image. Write 150-220 words "
        "in plain English with short paragraphs: (1) what the model predicted and how large the margin is, "
        "(2) what each explainability method highlights and whether they roughly agree, (3) limitations. "
        "Do not diagnose, do not state that a tumor is present or absent, do not claim the heatmaps show "
        "tumor boundaries, and remind the reader that scores are not calibrated probabilities and a "
        "radiologist must interpret any real scan."
    )
    user = "Model run summary (JSON):\n" + json.dumps(payload, indent=2)
    candidates = [model_name] + [m for m in GROQ_MODELS if m != model_name]
    last_err = None
    for m in candidates:
        try:
            r = client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_completion_tokens=1800,
            )
            text = (r.choices[0].message.content or "").strip()
            if not text:
                last_err = "The model returned an empty answer. Try again or pick another Groq model."
                continue
            return text, m, None
        except groq.AuthenticationError:
            return None, None, "Groq rejected the API key (invalid or revoked). Create a new key at console.groq.com and update secrets.toml."
        except groq.PermissionDeniedError:
            return None, None, "This Groq key is not allowed to use the model. Check your Groq account permissions."
        except groq.RateLimitError:
            return None, None, "Groq rate limit reached. Wait a minute and try again."
        except groq.NotFoundError:
            last_err = f"Model '{m}' is not available for this key. Trying the next one..."
            continue
        except groq.APIConnectionError:
            return None, None, "Could not reach Groq. Check your internet connection."
        except groq.APIStatusError as e:
            last_err = f"Groq returned an error (HTTP {getattr(e, 'status_code', '?')})."
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"Unexpected error while calling Groq ({type(e).__name__})."
            continue
    return None, None, last_err or "No Groq model could be used."


# --------------------------------------------------------------------------- #
# Evaluation JSON helpers
# --------------------------------------------------------------------------- #
def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten_scalars(obj, prefix="", depth=0, maxdepth=2):
    if isinstance(obj, dict):
        if "precision" in obj and depth > 0:  # per-class/avg rows go to the report table
            return
        for k, v in obj.items():
            p = f"{prefix} {k}".strip() if prefix else str(k)
            if is_num(v):
                yield p, v
            elif isinstance(v, dict) and depth < maxdepth:
                yield from flatten_scalars(v, p, depth + 1, maxdepth)


def fmt_metric(name: str, v) -> str:
    n = name.lower()
    if "loss" in n:
        return f"{v:.4f}"
    if isinstance(v, int) and v > 1:
        return f"{v:,}"
    if 0 <= v <= 1 and any(t in n for t in ("acc", "prec", "rec", "f1", "auc", "score", "sens", "spec")):
        return f"{v * 100:.2f}%"
    return f"{v:.4g}"


def find_report(obj):
    if isinstance(obj, dict):
        rows = {k: v for k, v in obj.items() if isinstance(v, dict) and "precision" in v}
        if len(rows) >= 2:
            return rows
        for v in obj.values():
            r = find_report(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_report(v)
            if r:
                return r
    return None


def find_confusion(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "confusion" in str(k).lower() and isinstance(v, list) and v and isinstance(v[0], list):
                return np.array(v)
        for v in obj.values():
            r = find_confusion(v)
            if r is not None:
                return r
    return None


def comparison_frame(obj) -> pd.DataFrame | None:
    rows = []
    if isinstance(obj, dict):
        dict_vals = {k: v for k, v in obj.items() if isinstance(v, dict)}
        if len(dict_vals) >= 2:
            for k, v in dict_vals.items():
                row = {"model": k}
                row.update({n: x for n, x in flatten_scalars(v, maxdepth=1)})
                rows.append(row)
    elif isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        for i, r in enumerate(obj):
            row = {"model": str(r.get("model", r.get("name", f"model {i + 1}")))}
            row.update({n: x for n, x in flatten_scalars(r, maxdepth=1)})
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df if df.shape[1] > 1 else None


def metric_cards(pairs) -> None:
    cards = "".join(f'<div class="metric"><span>{n}</span><b>{fmt_metric(n, v)}</b></div>' for n, v in pairs)
    html(f'<div class="mgrid">{cards}</div>')


def plotly_base(fig, height=300):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_MIST, family="DM Sans, sans-serif"),
        legend=dict(orientation="h", y=1.15, x=0),
    )
    fig.update_xaxes(gridcolor=C_LINE, zeroline=False)
    fig.update_yaxes(gridcolor=C_LINE, zeroline=False)
    return fig


def render_evaluation() -> None:
    note(
        "These are <b>dataset-level test metrics</b> read from your JSON files. They describe how the model did "
        "on the held-out test set. They say nothing certain about any single uploaded image.",
        "info",
    )

    # ---- ResNet50 test results --------------------------------------------
    st.subheader("ResNet50 fine-tuned: test set")
    data, p = load_json_file(RESNET_RESULTS_FILE)
    if data is None:
        st.info(f"`{RESNET_RESULTS_FILE}` not found in `models/` or `results/`.")
    elif "__error__" in data:
        st.error(f"Could not read `{p.name}`: {data['__error__']}")
    else:
        heads = [(n, v) for n, v in flatten_scalars(data) if any(t in n.lower() for t in ("acc", "prec", "rec", "f1", "loss", "auc", "kappa", "mcc"))]
        if heads:
            metric_cards(heads[:12])
        rep = find_report(data)
        if rep:
            df = pd.DataFrame(rep).T
            for c in df.columns:
                if c != "support":
                    df[c] = df[c].astype(float)
            st.dataframe(
                df.style.format({c: "{:.4f}" for c in df.columns if c != "support"}, na_rep=""),
                use_container_width=True,
            )
            per_class = df[[c for c in df.columns if c in ("precision", "recall", "f1-score", "f1_score", "f1")]]
            per_class = per_class[~per_class.index.str.contains("avg|accuracy", case=False)]
            if len(per_class) >= 2:
                fig = go.Figure()
                colors = [C_CYAN, C_AMBER, C_CORAL]
                for i, col in enumerate(per_class.columns):
                    fig.add_bar(name=col, x=per_class.index, y=per_class[col], marker_color=colors[i % 3])
                fig.update_layout(barmode="group", yaxis_range=[0, 1.05], title="Per-class metrics")
                st.plotly_chart(plotly_base(fig, 330))
        cm = find_confusion(data)
        if cm is not None and cm.ndim == 2:
            labels = [DISPLAY.get(c, c) for c in DEFAULT_ORDER] if cm.shape[0] == 4 else [str(i) for i in range(cm.shape[0])]
            fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#14252f"], [1, C_CYAN]], text=cm, texttemplate="%{text}", showscale=False))
            fig.update_layout(title="Confusion matrix (rows: true, columns: predicted; label order as in the file's classes)", yaxis_autorange="reversed")
            st.plotly_chart(plotly_base(fig, 380))
        with st.expander("Raw JSON"):
            st.json(data)

    # ---- Model comparison ---------------------------------------------------
    st.subheader("Model comparison")
    data, p = load_json_file(COMPARISON_FILE)
    if data is None:
        st.info(f"`{COMPARISON_FILE}` not found.")
    elif "__error__" in data:
        st.error(f"Could not read `{p.name}`: {data['__error__']}")
    else:
        df = comparison_frame(data)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
            num_cols = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c]) and any(t in c.lower() for t in ("acc", "prec", "rec", "f1"))]
            if num_cols:
                fig = go.Figure()
                palette = [C_CYAN, C_AMBER, C_CORAL, C_BLUE, "#b794f6"]
                for i, c in enumerate(num_cols):
                    fig.add_bar(name=c, x=df["model"], y=df[c], marker_color=palette[i % 5])
                fig.update_layout(barmode="group", title="Test metrics by model")
                st.plotly_chart(plotly_base(fig, 340))
        else:
            st.caption("The file's structure was not recognised as a model table, so only the raw content is shown.")
        with st.expander("Raw JSON", expanded=df is None):
            st.json(data)

    # ---- Custom CNN history -----------------------------------------------
    st.subheader("Custom CNN training history")
    data, p = load_json_file(HISTORY_FILE)
    if data is None:
        st.info(f"`{HISTORY_FILE}` not found.")
    elif "__error__" in data:
        st.error(f"Could not read `{p.name}`: {data['__error__']}")
    else:
        hist = data.get("history", data) if isinstance(data, dict) else {}
        series = {k: v for k, v in hist.items() if isinstance(v, list) and v and all(is_num(x) for x in v)} if isinstance(hist, dict) else {}
        if series:
            epochs = list(range(1, max(len(v) for v in series.values()) + 1))
            c1, c2 = st.columns(2)
            for col, key_part, title in ((c1, "loss", "Loss"), (c2, "acc", "Accuracy")):
                fig = go.Figure()
                palette = [C_CYAN, C_AMBER, C_CORAL, C_BLUE]
                ks = [k for k in series if key_part in k.lower()]
                for i, k in enumerate(ks):
                    fig.add_scatter(x=epochs[: len(series[k])], y=series[k], name=k, mode="lines", line=dict(color=palette[i % 4], width=2.5))
                fig.update_layout(title=f"{title} per epoch", xaxis_title="Epoch")
                with col:
                    if ks:
                        st.plotly_chart(plotly_base(fig, 300))
            facts = []
            for k, v in series.items():
                if "val" in k.lower() and "loss" in k.lower():
                    facts.append((f"best {k}", min(v)))
                elif "val" in k.lower() and "acc" in k.lower():
                    facts.append((f"best {k}", max(v)))
            facts.append(("epochs logged", len(epochs)))
            metric_cards(facts)
        with st.expander("Raw JSON"):
            st.json(data)

    if find_file(BASELINE_FILE):
        st.caption(f"`{BASELINE_FILE}` is present. It is not used in this dashboard (only the ResNet50 model is explained).")


# --------------------------------------------------------------------------- #
# UI blocks
# --------------------------------------------------------------------------- #
def render_hero() -> None:
    chips = "".join(f'<span class="chip"><b>{DISPLAY[c]}</b></span>' for c in DEFAULT_ORDER)
    html(
        f"""
        <div class="hero">
          <h1>What did the network look at?</h1>
          <p>Upload a brain MRI and compare Grad-CAM, SHAP and LIME explanations from a fine-tuned ResNet50 that sorts scans into four classes.</p>
          <div class="chips">{chips}</div>
        </div>
        """
    )
    note(
        "<b>Research and education only.</b> This is not a medical device and must not be used for diagnosis or treatment decisions. "
        "Scores are not calibrated probabilities, explanation maps do not prove tumor boundaries, and AI-written text can be wrong."
    )


def score_chart(scores: np.ndarray, order: list[str], pidx: int):
    labels = [DISPLAY[c] for c in order]
    fig = go.Figure(
        go.Bar(
            x=scores * 100,
            y=labels,
            orientation="h",
            marker_color=[C_CYAN if i == pidx else "#3a5666" for i in range(len(order))],
            text=[f"{s * 100:.1f}%" for s in scores],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(xaxis_range=[0, 112], yaxis_autorange="reversed", showlegend=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return plotly_base(fig, 240)


def legend_signed() -> None:
    html(
        f'<div class="legend"><span><i style="background:{C_CORAL}"></i>supports the selected class</span>'
        f'<span><i style="background:{C_BLUE}"></i>argues against it</span></div>'
    )


def sidebar(model_ready: bool, model, info, auto_mode, auto_reason):
    sb = st.sidebar
    sb.markdown("### Control panel")
    mode = "resnet"
    order = DEFAULT_ORDER
    if model_ready:
        sb.markdown("**Model**")
        sb.caption(f"Input {info['input_shape']}  |  Output {info['output_shape']}")
        sb.caption(f"{info['params']:,} parameters  |  {info['n_layers']} top-level layers")
        sb.caption(f"Grad-CAM layer: `{info['target_layer']}`")
        with sb.expander("Preprocessing (verify!)", expanded=False):
            keys = list(PREPROC_LABELS)
            mode = st.selectbox(
                "Input scaling",
                keys,
                index=keys.index(auto_mode),
                format_func=lambda k: PREPROC_LABELS[k],
                help="Must match what you used in training. Auto-detected default shown first.",
            )
            st.caption(f"Auto-detect: {auto_reason}. This is a heuristic, not proof.")
        # class order evidence from JSON files
        evidence, src = None, None
        for fname in (RESNET_RESULTS_FILE, COMPARISON_FILE):
            d, _ = load_json_file(fname)
            if d and "__error__" not in d:
                evidence = find_class_list(d)
                if evidence:
                    src = fname
                    break
        default_txt = ", ".join(evidence or DEFAULT_ORDER)
        with sb.expander("Class order (verify!)", expanded=False):
            txt = st.text_input("Output index 0 to 3", value=default_txt)
            parsed = parse_order(txt)
            if parsed is None:
                st.error("Enter exactly: glioma, meningioma, notumor, pituitary in the order of your model's output indices.")
            else:
                order = parsed
            if evidence:
                st.caption(f"Default taken from class list found in `{src}`.")
            else:
                st.caption("No class list found in the JSON files. Default is alphabetical folder order. Confirm with `train_ds.class_names` in Colab.")
    else:
        sb.info("Model not loaded yet.")

    sb.markdown("**Explainability settings**")
    shap_evals = sb.select_slider("SHAP evaluations", [100, 200, 300, 500, 800, 1200], value=300, help="Higher = finer maps, slower.")
    lime_samples = sb.select_slider("LIME samples", [200, 400, 800, 1200, 2000], value=800, help="Perturbed images evaluated by the model.")
    lime_kernel = sb.slider("LIME superpixel size", 2, 8, 4, help="Larger = fewer, bigger superpixels.")
    lime_topk = sb.slider("LIME regions shown", 3, 15, 6)

    sb.markdown("**Text explanation (Groq)**")
    _, gstatus = get_groq_client()
    (sb.success if gstatus == "ok" else sb.warning)(GROQ_STATUS_TEXT[gstatus])
    gmodel = sb.selectbox("Groq model", GROQ_MODELS, index=0)
    return mode, order, dict(shap_evals=shap_evals, lime_samples=lime_samples, lime_kernel=lime_kernel, lime_topk=lime_topk, groq_model=gmodel)


def render_diagnostics(model, info, order, mode) -> None:
    with st.expander("Check preprocessing and class order with a labelled image"):
        st.caption(
            "Upload an MRI whose true class you know (ideally from your test set). The table shows the model's top class under each "
            "preprocessing option. The option that gets known images right is the one used in training."
        )
        up = st.file_uploader("Labelled MRI", type=["jpg", "jpeg", "png"], key="diag_up")
        truth = st.selectbox("True class", order, format_func=lambda c: DISPLAY[c], key="diag_truth")
        if up is not None:
            try:
                arr = load_image(up, info["size"])
            except ValueError as e:
                st.error(str(e))
                return
            rows = []
            for m, label in PREPROC_LABELS.items():
                s = predict_scores(model, arr[None], m)[0]
                rows.append({"Preprocessing": label, "Top output index": int(s.argmax()), "Top class (with current order)": DISPLAY[order[int(s.argmax())]], "Top score": f"{s.max() * 100:.1f}%", "Matches?": "yes" if order[int(s.argmax())] == truth else "no"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("One image is weak evidence. Try 3-4 images from different classes.")


# --------------------------------------------------------------------------- #
# Analysis page
# --------------------------------------------------------------------------- #
def render_analysis(model, info, order, mode, cfg) -> None:
    S = st.session_state
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader("MRI image")
        up = st.file_uploader("Upload a brain MRI (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up is None:
            note("Choose a brain MRI slice to begin. Axial, coronal or sagittal slices from the same dataset as training work best.", "info")
            return
        try:
            img = load_image(up, info["size"])
        except ValueError as e:
            st.error(str(e))
            return
        st.image(img, caption=f"Model input ({info['size'][1]}x{info['size'][0]} px)", use_container_width=True)
        file_hash = hashlib.sha1(up.getvalue()).hexdigest()[:12]
        sig = f"{file_hash}|{mode}|{','.join(order)}"
        if S.get("sig") != sig:
            S.sig, S.pred, S.xai, S.ai = sig, None, {}, None
        go_btn = st.button("Analyze MRI", type="primary", use_container_width=True)

    if go_btn:
        try:
            with st.spinner("Running the model and Grad-CAM..."):
                scores = predict_scores(model, img[None], mode)[0]
                if scores.shape[0] != len(order):
                    raise ValueError(f"Model outputs {scores.shape[0]} values but 4 classes are configured.")
                S.pred = {"scores": scores, "pidx": int(scores.argmax()), "img": img}
                S.xai = {}
                S.ai = None
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed ({type(e).__name__}): {e}")
            return

    pred = S.get("pred")
    with right:
        st.subheader("Prediction")
        if pred is None:
            note("Press <b>Analyze MRI</b> to get the class scores.", "info")
            return
        scores, pidx = pred["scores"], pred["pidx"]
        srt = np.sort(scores)[::-1]
        margin = (srt[0] - srt[1]) * 100
        html(
            f"""<div class="pred"><div class="lab">Model's top class</div>
            <div class="cls">{DISPLAY[order[pidx]]}</div>
            <div class="sc">{scores[pidx] * 100:.1f}% model score</div>
            <div class="sub">{margin:.1f} points ahead of the runner-up ({DISPLAY[order[int(np.argsort(scores)[-2])]]}).
            Softmax outputs are not calibrated probabilities and apply only to this one image.</div></div>"""
        )
        st.plotly_chart(score_chart(scores, order, pidx))
        if scores[pidx] < 0.6 or margin < 20:
            note("The scores are close together, so this prediction is uncertain. Treat the explanations with extra caution.", "warn")

    # ----------------------------------------------------------------- explanations
    st.divider()
    st.subheader("Explanations")
    cls = st.selectbox(
        "Class to explain",
        list(range(len(order))),
        index=pidx,
        format_func=lambda i: DISPLAY[order[i]] + ("  (top class)" if i == pidx else ""),
        key=f"cls_{S.sig}",
    )
    img = pred["img"]
    xai = S.xai

    t_cam, t_shap, t_lime, t_cmp, t_ai = st.tabs(["Grad-CAM", "SHAP", "LIME", "Compare", "AI explanation"])

    # Grad-CAM -------------------------------------------------------------
    with t_cam:
        key = ("cam", cls)
        if key not in xai:
            try:
                layer = last_4d_layer(model)
                if layer is None:
                    raise RuntimeError("No convolutional (4-D) layer found in the model.")
                with st.spinner("Computing Grad-CAM..."):
                    cam, ginfo = gradcam(model, preprocess(img[None], mode), cls, layer)
                xai[key] = {"cam": resize_map(cam, img.shape[:2]).clip(0, 1), "info": ginfo, "grid": cam.shape}
            except Exception as e:  # noqa: BLE001
                st.error(f"Grad-CAM failed ({type(e).__name__}): {e}")
        if key in xai:
            r = xai[key]
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Original MRI", use_container_width=True)
            c2.image(colorize(r["cam"]), caption="Grad-CAM heatmap", use_container_width=True)
            c3.image(overlay_heat(img, r["cam"]), caption="Overlay on MRI", use_container_width=True)
            if r["info"].get("warning"):
                st.warning(r["info"]["warning"])
            st.markdown(
                f"Grad-CAM weights the last convolutional feature maps (`{r['info']['layer']}`, {r['grid'][0]}x{r['grid'][1]} grid) by how strongly "
                f"each one raises the **{DISPLAY[order[cls]]}** score ({r['info']['score_source']}). Warm colours mark regions that increased that score. "
                "The map is coarse because it is upsampled from the feature grid, and it shows model attention, not tumor boundaries."
            )

    # SHAP -----------------------------------------------------------------
    with t_shap:
        st.markdown(
            "SHAP (Partition explainer with an image masker) hides blocks of the image with a blurred version and measures how the "
            "model's scores change, then shares credit among regions with Shapley values. It runs on your real model and preprocessing; "
            "no background dataset is needed with this masker."
        )
        if st.button("Compute SHAP", key="b_shap"):
            try:
                with st.spinner(f"Computing SHAP ({cfg['shap_evals']} model evaluations per image)... this can take a few minutes on CPU."):
                    xai[("shap", "all")] = run_shap(model, img, mode, cfg["shap_evals"])
            except Exception as e:  # noqa: BLE001
                st.error(f"SHAP failed ({type(e).__name__}): {e}. Try fewer evaluations in the sidebar, or close other apps to free memory.")
        if ("shap", "all") in xai:
            r = xai[("shap", "all")]
            m = r["values"][..., cls].sum(-1)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Original MRI", use_container_width=True)
            c2.image(overlay_signed(img, m), caption=f"SHAP attribution for {DISPLAY[order[cls]]}", use_container_width=True)
            legend_signed()
            total = float(m.sum() + r["base"][cls])
            st.caption(
                f"Sanity check: base value {r['base'][cls]:.3f} + sum of attributions {m.sum():.3f} = {total:.3f}; "
                f"this should be close to the model's score for this class ({pred['scores'][cls]:.3f}). Computed with {r['max_evals']} evaluations."
            )
            st.markdown(
                "Red areas pushed the score for this class up; blue areas pushed it down. Attributions are block-level estimates "
                "and depend on the masking method, so they do not outline a tumor."
            )

    # LIME -----------------------------------------------------------------
    with t_lime:
        st.markdown(
            "LIME splits the MRI into superpixels, switches random subsets off, asks the real model for each variant, and fits a small "
            "linear model to see which superpixels matter most for the selected class."
        )
        if st.button("Compute LIME", key="b_lime"):
            try:
                with st.spinner(f"Computing LIME ({cfg['lime_samples']} perturbed images)..."):
                    xai[("lime", cls)] = run_lime(model, img, mode, cls, cfg["lime_samples"], cfg["lime_kernel"], cfg["lime_topk"])
            except Exception as e:  # noqa: BLE001
                st.error(f"LIME failed ({type(e).__name__}): {e}. Try fewer samples in the sidebar.")
        if ("lime", cls) in xai:
            r = xai[("lime", cls)]
            c1, c2 = st.columns(2)
            c1.image(img, caption="Original MRI", use_container_width=True)
            c2.image(r["vis"], caption=f"LIME top {cfg['lime_topk']} superpixels for {DISPLAY[order[cls]]}", use_container_width=True)
            legend_signed()
            fit = f" Local surrogate R2 = {r['local_r2']:.2f} (closer to 1 means a more faithful local fit)." if r["local_r2"] is not None else ""
            st.caption(f"{r['n_segments']} superpixels, {r['num_samples']} samples, fixed random seed.{fit}")
            st.markdown("LIME is stochastic and sensitive to the segmentation, so rerunning with other settings can change the picture.")

    # Compare --------------------------------------------------------------
    with t_cmp:
        maps, panels = {}, []
        if ("cam", cls) in xai:
            c = xai[("cam", cls)]["cam"]
            maps["Grad-CAM"] = c
            panels.append(("Grad-CAM", overlay_heat(img, c)))
        if ("shap", "all") in xai:
            m = xai[("shap", "all")]["values"][..., cls].sum(-1)
            maps["SHAP"] = np.maximum(m, 0)
            panels.append(("SHAP", overlay_signed(img, m)))
        if ("lime", cls) in xai:
            maps["LIME"] = np.maximum(xai[("lime", cls)]["wmap"], 0)
            panels.append(("LIME", xai[("lime", cls)]["vis"]))
        if len(panels) < 2:
            note("Compute at least two methods (Grad-CAM appears automatically; compute SHAP and/or LIME in their tabs) for the same class to compare them here.", "info")
        cols = st.columns(len(panels) + 1)
        cols[0].image(img, caption="Original MRI", use_container_width=True)
        for col, (name, im) in zip(cols[1:], panels):
            col.image(im, caption=name, use_container_width=True)
        if len(maps) >= 2:
            names = list(maps)
            rows = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    rows.append({"Pair": f"{names[i]} vs {names[j]}", "Overlap of top-15% regions (IoU)": round(top_iou(maps[names[i]], maps[names[j]]), 3)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("IoU is computed from the three maps above using only their positive evidence. 0 = no overlap, 1 = identical. Low overlap is common and does not by itself say which method is right.")
        st.markdown(
            "**Reading them together.** Grad-CAM is gradient-based and coarse; SHAP and LIME are perturbation-based and model-agnostic but "
            "approximate and sensitive to how the image is masked or segmented. They answer slightly different questions, so they can disagree. "
            "None of them establishes a clinical diagnosis or tumor boundary."
        )

    # AI explanation -------------------------------------------------------
    with t_ai:
        client, gstatus = get_groq_client()
        if client is None:
            note(GROQ_STATUS_TEXT[gstatus] + " Prediction, Grad-CAM, SHAP and LIME keep working without it.", "warn")
        st.caption("Only numbers (scores and a rough description of where each map is strongest) are sent to Groq. The image is never uploaded.")
        if st.button("Generate AI Explanation", key="b_ai", disabled=client is None, type="primary"):
            payload = {
                "predicted_class": DISPLAY[order[pidx]],
                "class_scores_percent": {DISPLAY[c]: round(float(s) * 100, 1) for c, s in zip(order, scores)},
                "class_explained": DISPLAY[order[cls]],
                "note_on_regions": "Left/right are image-left/right, not radiological convention.",
                "methods": {},
            }
            if ("cam", cls) in xai:
                payload["methods"]["Grad-CAM"] = describe_map(xai[("cam", cls)]["cam"])
            if ("shap", "all") in xai:
                payload["methods"]["SHAP"] = describe_map(xai[("shap", "all")]["values"][..., cls].sum(-1))
            if ("lime", cls) in xai:
                payload["methods"]["LIME"] = describe_map(xai[("lime", cls)]["wmap"])
            if len(maps) >= 2:
                payload["pairwise_top15_iou"] = {r["Pair"]: r["Overlap of top-15% regions (IoU)"] for r in rows}
            with st.spinner("Asking Groq..."):
                text, used, err = groq_explain(client, cfg["groq_model"], payload)
            S.ai = {"text": text, "model": used, "error": err}
        if S.get("ai"):
            if S.ai["error"]:
                st.error(S.ai["error"])
            else:
                st.markdown(S.ai["text"])
                st.caption(f"Generated by `{S.ai['model']}` on Groq. AI-written text can be inaccurate and is not medical advice.")


def render_about() -> None:
    st.subheader("How this works")
    st.markdown(
        """
- **Model:** `resnet50_finetuned.keras`, loaded once and cached. Input size, output size and the Grad-CAM layer are read from the model itself.
- **Grad-CAM:** gradients of the class score (pre-softmax logit when the model ends in softmax) with respect to the last convolutional feature maps.
- **SHAP:** `shap.Explainer` with an image masker (Partition explainer). It is model-agnostic, so it does not depend on Keras internals, and it calls the same prediction function as the rest of the app.
- **LIME:** `lime.lime_image` with quickshift superpixels and the same prediction function.
- **Text:** optional Groq explanation of numeric summaries, produced only on request.
"""
    )
    st.subheader("Limitations")
    st.markdown(
        """
- Preprocessing and class order **cannot be proven** from the model file alone. Use the sidebar and the labelled-image check, and compare with your training notebook.
- Scores are softmax outputs, not calibrated probabilities. A high score can still be wrong.
- Explanation methods are approximations. They can disagree, and they do not show tumor boundaries.
- SHAP and LIME run many forward passes, so they are slow on CPU. Lower the settings in the sidebar if memory is tight.
- The model was trained on one dataset and may fail on scans from other scanners, sequences or populations.
- Not a medical device. Real scans must be interpreted by qualified clinicians.
"""
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Explainable Brain Tumor MRI", page_icon="🧠", layout="wide")
    inject_css()
    render_hero()

    model, info, auto_mode, auto_reason, ready = None, None, "resnet", "", False
    mpath = find_file(MODEL_FILE)
    if mpath is None:
        st.error(f"`{MODEL_FILE}` was not found. Put it in the `models/` folder next to app.py, then refresh.")
    else:
        try:
            model = load_model(str(mpath), mpath.stat().st_mtime)
            info = model_info(model)
            auto_mode, auto_reason = detect_preprocessing(model)
            if info["output_shape"][-1] != 4:
                st.error(f"The model has {info['output_shape'][-1]} outputs, but 4 classes are expected.")
            elif info["channels"] not in (3, None):
                st.error(f"The model expects {info['channels']} input channels, but this app feeds RGB (3).")
            else:
                ready = True
        except Exception as e:  # noqa: BLE001
            st.error(
                f"The model could not be loaded ({type(e).__name__}). This usually means your local TensorFlow/Keras version "
                f"differs from the one used to save the model in Colab. Details: {str(e)[:400]}"
            )

    mode, order, cfg = sidebar(ready, model, info, auto_mode, auto_reason)

    tab_a, tab_e, tab_i = st.tabs(["Analyze", "Evaluation results", "About and limits"])
    with tab_a:
        if ready:
            note(
                f"Preprocessing in use: <b>{PREPROC_LABELS[mode]}</b>. Class order: <b>{', '.join(DISPLAY[c] for c in order)}</b>. "
                "Both are assumptions until you confirm them against your training code (see sidebar).",
                "info",
            )
            render_diagnostics(model, info, order, mode)
            render_analysis(model, info, order, mode, cfg)
        else:
            st.info("Load a valid model to enable image analysis.")
    with tab_e:
        render_evaluation()
    with tab_i:
        render_about()

    st.divider()
    st.caption("For research and education only. Not for clinical use. Explanations are approximations and do not prove tumor boundaries.")


if __name__ == "__main__":
    main()
