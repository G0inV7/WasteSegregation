# ==============================================================================
# rpi_inference.py — EfficientEdge WasteNet  |  Raspberry Pi 4B Deployment
# ==============================================================================
# Headless mode: no display required. Press ENTER to capture and classify.
# Each result is saved as a timestamped JPEG in results/.
# Optional per-capture layer heatmaps are saved under results/heatmaps/<ts>/.
#
# Prerequisites (on the RPi):
#   pip install tflite-runtime opencv-python-headless numpy
#
# Files needed in the same directory as this script:
#   waste_net_student_int8.tflite   ← download from Kaggle output
#
# Run:
#   python rpi_inference.py
#
# Controls (type in the terminal, then press ENTER):
#   <ENTER>          Capture → classify → save result image (+ heatmaps if on)
#   h  <ENTER>       Toggle heatmap saving on/off
#   s  <ENTER>       Show session summary
#   q  <ENTER>       Quit
# ==============================================================================

import cv2
import math
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/waste_net_student_int8_FirstWork.tflite"
IMG_SIZE        = 160
THRESHOLD       = 0.6869    # Replace with Youden's J value from training run
NUM_THREADS     = 4
CAMERA_INDEX    = 0
CAM_WIDTH       = 1920
CAM_HEIGHT      = 1080
RESULTS_DIR     = "results"
HEATMAPS_DIR    = os.path.join(RESULTS_DIR, "heatmaps")

# Set to False to skip heatmaps by default (toggle at runtime with 'h')
SAVE_HEATMAPS_DEFAULT = True

# Layers to visualise — matched against TFLite tensor names by substring.
# Each entry:  (match_substring, folder_name, max_filters_to_show)
HEATMAP_LAYERS = [
    ("stem_conv",           "stem_conv",           16),
    ("stem_relu",           "stem_relu",            16),
    ("spatial_attn_stage2", "spatial_attn_stage2",  16),
    ("se_stage3",           "se_stage3",            16),
    ("spatial_attn_neck",   "spatial_attn_neck",    16),
]

# Heatmap grid layout
GRID_COLS       = 4          # filters per row in the saved grid image
CELL_PX         = 160        # each filter cell scaled to this size
CELL_PAD        = 4          # padding between cells (pixels)
COLORMAP        = cv2.COLORMAP_VIRIDIS

# Overlay colours (BGR)
COLOUR_ORGANIC    = (94,  197,  34)
COLOUR_RECYCLABLE = (246, 130,  59)
COLOUR_INFO       = (255, 255, 255)
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
Path(RESULTS_DIR).mkdir(exist_ok=True)
Path(HEATMAPS_DIR).mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  +  MAP LAYER NAMES → TENSOR INDICES
# ──────────────────────────────────────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model file not found: {path}")
    print(f"Loading model: {path}")
    interp = tflite.Interpreter(model_path=path, num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input  tensor : {inp['shape']}  dtype={inp['dtype'].__name__}")
    print(f"  Output tensor : {out['shape']}  dtype={out['dtype'].__name__}")
    print(f"  Threads       : {NUM_THREADS}")
    print(f"  Threshold     : {THRESHOLD}")
    return interp, inp, out


def build_layer_index(interp, layer_specs):
    """
    Scan all TFLite tensors and match each spec's substring to a tensor name.
    Returns a list of (folder_name, tensor_idx, max_filters, quant_params)
    for every spec that was successfully matched.

    INT8 quantised tensors are dequantised via:
        float_val = (int8_val - zero_point) * scale
    Tensors with scale == 0 are already float32 — no dequantisation needed.
    """
    all_tensors = interp.get_tensor_details()
    matched = []
    for (substr, folder, max_f) in layer_specs:
        hit = None
        for t in all_tensors:
            if substr in t["name"]:
                hit = t
                break
        if hit is None:
            print(f"  [HEATMAP] WARNING: no tensor matched '{substr}' — skipped")
            continue
        scale, zero_point = hit["quantization"]
        matched.append((folder, hit["index"], max_f, scale, zero_point))
        shape_str = "x".join(str(d) for d in hit["shape"]) if hit["shape"] is not None else "?"
        print(f"  [HEATMAP] '{substr}' -> tensor #{hit['index']} "
              f"name='{hit['name']}' shape=({shape_str})")
    return matched


interpreter, inp_detail, out_detail = load_model(MODEL_PATH)
layer_index = build_layer_index(interpreter, HEATMAP_LAYERS)


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Centre-crop 16:9 -> 1:1, resize to IMG_SIZE, BGR->RGB, add batch dim.
    Model has Rescaling(1/255) internally so raw uint8 float values are fine.
    """
    h, w    = bgr_frame.shape[:2]
    crop    = min(h, w)
    y0      = (h - crop) // 2
    x0      = (w - crop) // 2
    square  = bgr_frame[y0:y0 + crop, x0:x0 + crop]
    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[np.newaxis]   # (1, 160, 160, 3)


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (returns raw tensor values for heatmaps too)
# ──────────────────────────────────────────────────────────────────────────────
def predict(blob: np.ndarray):
    """
    Run one forward pass.
    Returns (label, probability, inference_ms).
    Intermediate tensors are accessible via interpreter.get_tensor() after this.
    """
    interpreter.set_tensor(inp_detail["index"], blob)
    t0 = time.perf_counter()
    interpreter.invoke()
    inf_ms = (time.perf_counter() - t0) * 1000
    prob   = float(interpreter.get_tensor(out_detail["index"])[0, 0])
    label  = "Recyclable" if prob >= THRESHOLD else "Organic"
    return label, prob, inf_ms


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def dequantize(tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Convert INT8 tensor to float32. No-op when scale == 0 (already float)."""
    if scale == 0:
        return tensor.astype(np.float32)
    return (tensor.astype(np.float32) - zero_point) * scale


def make_filter_grid(feature_map: np.ndarray, max_filters: int) -> np.ndarray:
    """
    feature_map : (1, H, W, C)  float32
    Returns a BGR grid image showing up to max_filters channels,
    each normalised independently and colourised with COLORMAP_VIRIDIS.
    """
    fmap   = feature_map[0]              # (H, W, C)
    H, W, C = fmap.shape
    n      = min(C, max_filters)
    cols   = GRID_COLS
    rows   = math.ceil(n / cols)

    cell   = CELL_PX
    pad    = CELL_PAD
    canvas = np.zeros(
        (rows * (cell + pad) + pad,
         cols * (cell + pad) + pad, 3),
        dtype=np.uint8
    )

    for i in range(n):
        ch = fmap[:, :, i].astype(np.float32)

        # Per-channel min-max normalisation -> 0..255
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            ch = (ch - mn) / (mx - mn) * 255.0
        else:
            ch = np.zeros_like(ch)

        ch_u8  = ch.astype(np.uint8)
        ch_rsz = cv2.resize(ch_u8, (cell, cell),
                            interpolation=cv2.INTER_NEAREST)
        coloured = cv2.applyColorMap(ch_rsz, COLORMAP)  # (cell, cell, 3) BGR

        r = i // cols
        c = i  % cols
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = coloured

    return canvas


def save_heatmaps(ts_folder: str):
    """
    Extract intermediate tensors (already populated by the last invoke()),
    generate filter grids, and save into per-layer subfolders.

    Folder layout:
        results/heatmaps/<timestamp>_<label>/
            stem_conv/
                grid.jpg
            stem_relu/
                grid.jpg
            ...
    """
    base = ts_folder
    Path(base).mkdir(parents=True, exist_ok=True)

    saved = []
    for (folder, t_idx, max_f, scale, zero_point) in layer_index:
        try:
            raw = interpreter.get_tensor(t_idx)           # (1, H, W, C) or None
        except Exception as e:
            print(f"  [HEATMAP] Could not read tensor for '{folder}': {e}")
            continue

        if raw is None or raw.ndim != 4:
            print(f"  [HEATMAP] Unexpected tensor shape for '{folder}' "
                  f"(shape={getattr(raw,'shape','?')}) — skipped")
            continue

        fmap  = dequantize(raw, scale, zero_point)
        grid  = make_filter_grid(fmap, max_f)

        layer_dir = os.path.join(base, folder)
        Path(layer_dir).mkdir(exist_ok=True)
        out_path  = os.path.join(layer_dir, "grid.jpg")
        cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(f"{folder}/grid.jpg")

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# RESULT IMAGE DRAWING
# ──────────────────────────────────────────────────────────────────────────────
def draw_result(frame: np.ndarray, label: str, prob: float,
                inf_ms: float, timestamp: str,
                heatmaps_saved: bool) -> np.ndarray:
    out    = frame.copy()
    h, w   = out.shape[:2]
    colour = COLOUR_RECYCLABLE if label == "Recyclable" else COLOUR_ORGANIC
    conf   = prob if label == "Recyclable" else (1 - prob)

    # Top banner
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, f"{label}   {conf*100:.1f}%",
                (24, 70), FONT, 2.2, colour, 4, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 24, 88, 500, 22
    cv2.rectangle(out, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    cv2.rectangle(out, (bar_x, bar_y),
                  (bar_x + int(bar_w * conf), bar_y + bar_h), colour, -1)

    # Bottom strip
    strip_y  = h - 46
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, strip_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, out, 0.45, 0, out)
    hm_tag = "  [heatmaps saved]" if heatmaps_saved else ""
    bottom  = (f"Inference: {inf_ms:.1f} ms   |   "
               f"Threshold: {THRESHOLD:.4f}   |   {timestamp}{hm_tag}")
    cv2.putText(out, bottom,
                (24, h - 14), FONT, 0.75, COLOUR_INFO, 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SAVE RESULT IMAGE
# ──────────────────────────────────────────────────────────────────────────────
def save_result(annotated: np.ndarray, label: str, timestamp: str) -> str:
    safe_ts  = timestamp.replace(":", "").replace(" ", "_").replace("-", "")
    filename = f"{safe_ts}_{label}.jpg"
    filepath = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return filepath, safe_ts


# ──────────────────────────────────────────────────────────────────────────────
# SESSION LOG
# ──────────────────────────────────────────────────────────────────────────────
session_log = []

def log_result(label, prob, inf_ms, timestamp, filepath):
    session_log.append(dict(
        timestamp=timestamp, label=label,
        prob=prob, inf_ms=inf_ms, file=filepath))


def print_summary():
    if not session_log:
        print("\n  No predictions made yet.\n")
        return
    total   = len(session_log)
    organic = sum(1 for r in session_log if r["label"] == "Organic")
    recycl  = total - organic
    avg_ms  = sum(r["inf_ms"] for r in session_log) / total
    print("\n" + "-" * 62)
    print(f"  SESSION SUMMARY  ({total} prediction{'s' if total != 1 else ''})")
    print("-" * 62)
    print(f"  Organic     : {organic:>3}  ({organic/total*100:.1f}%)")
    print(f"  Recyclable  : {recycl:>3}  ({recycl/total*100:.1f}%)")
    print(f"  Avg latency : {avg_ms:.1f} ms")
    print("-" * 62)
    for r in session_log[-10:]:
        conf = r["prob"] if r["label"] == "Recyclable" else 1 - r["prob"]
        print(f"    {r['timestamp']}  {r['label']:<12}  "
              f"{conf*100:5.1f}%  {r['inf_ms']:6.1f} ms")
    if total > 10:
        print(f"    ... ({total - 10} earlier entries not shown)")
    print(f"  Results folder  : {os.path.abspath(RESULTS_DIR)}/")
    print(f"  Heatmaps folder : {os.path.abspath(HEATMAPS_DIR)}/")
    print("-" * 62 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    save_heatmaps_flag = SAVE_HEATMAPS_DEFAULT

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera opened: {actual_w}x{actual_h}")
    print(f"Results    -> {os.path.abspath(RESULTS_DIR)}/")
    print(f"Heatmaps   -> {os.path.abspath(HEATMAPS_DIR)}/")

    # Print matched heatmap layers
    if layer_index:
        print(f"\nHeatmap layers ({len(layer_index)} matched):")
        for (folder, t_idx, max_f, *_) in layer_index:
            print(f"  [{folder}]  tensor #{t_idx}  up to {max_f} filters")
    else:
        print("\n  [HEATMAP] No layers matched — heatmaps will be empty.")

    hm_status = "ON" if save_heatmaps_flag else "OFF"
    print(f"\nHeatmap saving: {hm_status}")
    print("\n" + "=" * 52)
    print("  Controls (type command, then press ENTER):")
    print("    <ENTER>       Capture -> classify -> save")
    print("    h  <ENTER>    Toggle heatmap saving on/off")
    print("    s  <ENTER>    Session summary")
    print("    q  <ENTER>    Quit")
    print("=" * 52 + "\n")

    # Interpreter warmup
    print("Warming up interpreter...", end=" ", flush=True)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    interpreter.set_tensor(inp_detail["index"], dummy)
    interpreter.invoke()
    print("ready.\n")

    while True:
        try:
            hm_tag = "[heatmaps ON]" if save_heatmaps_flag else "[heatmaps OFF]"
            cmd = input(f"ENTER=capture | h=toggle heatmaps {hm_tag} | s=summary | q=quit : "
                        ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if cmd == "q":
            print("Quitting.")
            break

        if cmd == "s":
            print_summary()
            continue

        if cmd == "h":
            save_heatmaps_flag = not save_heatmaps_flag
            state = "ON" if save_heatmaps_flag else "OFF"
            print(f"  Heatmap saving turned {state}.\n")
            continue

        # ── Capture + classify ────────────────────────────────────────────────
        frame = None
        for _ in range(3):          # discard frames for auto-exposure settle
            ret, frame = cap.read()
            if not ret:
                break

        if not ret or frame is None:
            print("  [WARNING] Failed to read from camera — try again.\n")
            continue

        timestamp           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blob                = preprocess(frame)
        label, prob, inf_ms = predict(blob)
        conf                = prob if label == "Recyclable" else 1 - prob

        # ── Heatmaps (tensors still valid right after invoke()) ───────────────
        hm_paths = []
        if save_heatmaps_flag and layer_index:
            safe_ts  = (timestamp.replace(":", "")
                                 .replace(" ", "_")
                                 .replace("-", ""))
            ts_folder = os.path.join(HEATMAPS_DIR,
                                     f"{safe_ts}_{label}")
            hm_paths  = save_heatmaps(ts_folder)

        # ── Save annotated result image ───────────────────────────────────────
        annotated         = draw_result(frame, label, prob, inf_ms,
                                        timestamp, bool(hm_paths))
        filepath, safe_ts = save_result(annotated, label, timestamp)
        log_result(label, prob, inf_ms, timestamp, filepath)

        # ── Console feedback ──────────────────────────────────────────────────
        bar_len = 30
        filled  = int(bar_len * conf)
        bar     = "#" * filled + "-" * (bar_len - filled)
        symbol  = "[R]" if label == "Recyclable" else "[O]"
        print(f"\n  {symbol}  {label:<12}  [{bar}]  {conf*100:5.1f}%"
              f"   {inf_ms:.1f} ms")
        print(f"  Image   -> {filepath}")
        if hm_paths:
            hm_base = os.path.join(HEATMAPS_DIR, f"{safe_ts}_{label}")
            print(f"  Heatmaps-> {hm_base}/")
            for p in hm_paths:
                print(f"             {p}")
        print()

    cap.release()
    print_summary()
    print("Camera released. Exiting.")


if __name__ == "__main__":
    main()
