# ==============================================================================
# rpi_inference.py — EfficientEdge WasteNet  |  Raspberry Pi 4B Deployment
# ==============================================================================
# Headless / no-display mode.  Press ENTER to capture and classify.
#
# Fixes in this version
# ─────────────────────
# 1. Buffer drain  : OpenCV keeps a ~4-frame internal buffer.  After user
#    input the buffer holds stale frames from before ENTER was pressed.
#    We now drain the buffer with rapid cap.grab() calls before the real
#    capture, so the saved image reflects what the camera sees RIGHT NOW.
#
# 2. Dual-crop ensemble  : Two crops of the same frame are classified
#    independently and then fused.
#      • Crop A (FULL)   — standard 1:1 centre crop, sees full object +
#                          surrounding packaging context.
#      • Crop B (TIGHT)  — inner 55 % of the frame, zooms into the centre
#                          of the object.
#    Fusion rule
#      Both agree            → use that label (high confidence)
#      Disagree              → RECYCLABLE wins
#        Rationale: the only common disagreement pattern is a recyclable
#        item whose wrapper carries a printed food image.  Crop B sees the
#        food picture and fires Organic; Crop A sees the plastic/foil edges
#        and correctly fires Recyclable.  Defaulting to Recyclable on
#        disagreement eliminates this false-positive class.
#
# Controls (type in terminal, then ENTER):
#   <ENTER>     Capture → classify → save
#   f <ENTER>   Flag last prediction as wrong → give correct label
#   h <ENTER>   Toggle heatmap saving on/off
#   s <ENTER>   Session summary
#   q <ENTER>   Quit
#
# Prerequisites:
#   pip install numpy opencv-python-headless
#   sudo apt install python3-tflite-runtime
# ==============================================================================

import cv2
import json
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
MODEL_PATH   = "models/waste_net_student_int8_FirstWork.tflite"
IMG_SIZE     = 160
THRESHOLD    = 0.6869     # Replace with Youden's J from training run
NUM_THREADS  = 4
CAMERA_INDEX = 0
CAM_WIDTH    = 1920
CAM_HEIGHT   = 1080

RESULTS_DIR  = "results"
HEATMAPS_DIR = os.path.join(RESULTS_DIR, "heatmaps")
FEEDBACK_DIR = "feedback"
FEEDBACK_LOG = os.path.join(FEEDBACK_DIR, "feedback_log.json")

SAVE_HEATMAPS_DEFAULT = True

# ── Dual-crop settings ────────────────────────────────────────────────────────
# TIGHT_CROP_RATIO : fraction of the 1:1 square used for the tight crop.
#   0.55  → use inner 55 % → effective 2× zoom on a 1920×1080 source
#   Increase toward 1.0 to reduce zoom; decrease toward 0.3 for more zoom.
TIGHT_CROP_RATIO = 0.55

# ── White background masking (applied to the FULL crop only) ──────────────────
# Pixels brighter than WHITE_BG_THRESHOLD in all channels are replaced with
# WHITE_BG_FILL (neutral gray) before the full-crop inference pass.
#
# Why only the full crop?
#   The full (zoomed-out) crop sees large regions of white table/plate/paper
#   surrounding the object. The model was trained on clean dataset images
#   where the object fills the frame, so a dominant white surround gets
#   misread as white paper packaging -> false Recyclable.
#   Replacing white with neutral gray removes that spurious signal without
#   affecting the object itself.
#
#   The tight crop is NOT masked: it is already zoomed into the object and
#   contains very little background, so masking is unnecessary there.
#
# Tuning WHITE_BG_THRESHOLD:
#   230 -- conservative, only catches near-pure-white backgrounds (default)
#   200 -- more aggressive, also masks light-grey surfaces
#   Lower this if organic items are still being called Recyclable.
#   Raise it if naturally white organic items (garlic, onion) get masked.
WHITE_BG_THRESHOLD = 230                # 0-255; pixels above this are masked
WHITE_BG_FILL      = (128, 128, 128)    # BGR neutral gray replacement

# ── Buffer drain settings ─────────────────────────────────────────────────────
# How many frames to grab-and-discard before the real capture.
# 10 frames at 30 fps ≈ 330 ms — enough to clear the buffer and let
# auto-exposure settle on the current scene.
DRAIN_FRAMES = 10

# ── Heatmap layers ────────────────────────────────────────────────────────────
HEATMAP_LAYERS = [
    ("stem_conv",           "stem_conv",           16),
    ("stem_relu",           "stem_relu",            16),
    ("spatial_attn_stage2", "spatial_attn_stage2",  16),
    ("se_stage3",           "se_stage3",            16),
    ("spatial_attn_neck",   "spatial_attn_neck",    16),
]
GRID_COLS = 4
CELL_PX   = 160
CELL_PAD  = 4
COLORMAP  = cv2.COLORMAP_VIRIDIS

# Overlay colours (BGR)
COLOUR_ORGANIC    = (94,  197,  34)
COLOUR_RECYCLABLE = (246, 130,  59)
COLOUR_ENSEMBLE   = (0,   220, 255)   # amber — shown when crops disagreed
COLOUR_INFO       = (255, 255, 255)
COLOUR_FEEDBACK   = (0,   80,  255)   # red — feedback indicator
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────────────────────────────────────────
# SETUP DIRECTORIES
# ──────────────────────────────────────────────────────────────────────────────
for d in [RESULTS_DIR, HEATMAPS_DIR,
          os.path.join(FEEDBACK_DIR, "Organic"),
          os.path.join(FEEDBACK_DIR, "Recyclable")]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + MAP HEATMAP LAYER NAMES → TENSOR INDICES
# ──────────────────────────────────────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model file not found: {path}")
    print(f"Loading model: {path}")
    interp = tflite.Interpreter(model_path=path, num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input  : {inp['shape']}  dtype={inp['dtype'].__name__}")
    print(f"  Output : {out['shape']}  dtype={out['dtype'].__name__}")
    print(f"  Threads: {NUM_THREADS}   Threshold: {THRESHOLD}")
    return interp, inp, out


def build_layer_index(interp, specs):
    all_tensors = interp.get_tensor_details()
    matched = []
    for (substr, folder, max_f) in specs:
        hit = next((t for t in all_tensors if substr in t["name"]), None)
        if hit is None:
            print(f"  [HEATMAP] WARNING: no tensor matched '{substr}'")
            continue
        scale, zp = hit["quantization"]
        shape_str = "x".join(str(d) for d in hit["shape"])
        print(f"  [HEATMAP] '{folder}' -> tensor #{hit['index']} ({shape_str})")
        matched.append((folder, hit["index"], max_f, scale, zp))
    return matched


interpreter, inp_detail, out_detail = load_model(MODEL_PATH)
layer_index = build_layer_index(interpreter, HEATMAP_LAYERS)


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — two crop variants
# ──────────────────────────────────────────────────────────────────────────────
def _square_crop(bgr_frame: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """
    Return a centre-cropped square from bgr_frame.
    ratio=1.0 → largest possible square (standard behaviour).
    ratio=0.55 → inner 55 % of that square (tight / zoomed crop).
    Then resize to IMG_SIZE × IMG_SIZE and convert to RGB float32 batch.
    """
    h, w  = bgr_frame.shape[:2]
    side  = int(min(h, w) * ratio)
    cy, cx = h // 2, w // 2
    y0 = cy - side // 2
    x0 = cx - side // 2
    square  = bgr_frame[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[np.newaxis]   # (1, 160, 160, 3)


def _mask_white_background(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Replace near-white pixels with neutral gray (WHITE_BG_FILL).
    Applied only to the full crop before inference.

    A pixel is considered 'white background' when ALL three BGR channels
    exceed WHITE_BG_THRESHOLD. This avoids masking colourful objects that
    happen to have one bright channel (e.g. a yellow banana won't be masked
    because its B and G channels are well below 230).
    """
    b, g, r = cv2.split(bgr_frame)
    mask = (b.astype(np.uint16) + g.astype(np.uint16) + r.astype(np.uint16)
            > WHITE_BG_THRESHOLD * 3).astype(np.uint8)  # 1 where white
    result = bgr_frame.copy()
    result[mask == 1] = WHITE_BG_FILL
    return result


def preprocess_full(bgr_frame):
    masked = _mask_white_background(bgr_frame)
    return _square_crop(masked, ratio=1.0)


def preprocess_tight(bgr_frame):
    return _square_crop(bgr_frame, ratio=TIGHT_CROP_RATIO)


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def _run(blob: np.ndarray):
    """Single forward pass. Returns (label, prob)."""
    interpreter.set_tensor(inp_detail["index"], blob)
    interpreter.invoke()
    prob  = float(interpreter.get_tensor(out_detail["index"])[0, 0])
    label = "Recyclable" if prob >= THRESHOLD else "Organic"
    return label, prob


def predict_dual(bgr_frame: np.ndarray):
    """
    Run two forward passes and fuse results.

    Pass A — full crop  (ratio 1.0, white background MASKED)
        Sees the whole object + packaging context.
        White background replaced with neutral gray so the model does not
        mistake table/plate/paper surround for white paper packaging.

    Pass B — tight crop (ratio TIGHT_CROP_RATIO, no masking)
        Inner 55% of the frame, ~2x zoom onto the object centre.
        No masking needed — the object already fills this crop.

    Fusion
    ------
    Both agree       -> average probabilities, use that label.
    Disagree         -> Recyclable wins.
        Remaining disagreement case after masking:
        Recyclable item whose wrapper has a printed food image.
        Tight crop zooms into the food image -> fires Organic.
        Full crop (masked) still sees plastic/foil edges -> fires Recyclable.
        Recyclable tie-break is correct here.

        The previous false-positive (organic item on white background) is
        resolved upstream by the masking step — both crops now agree Organic
        so the tie-break is never reached for that scenario.
    """
    t0 = time.perf_counter()

    label_full,  prob_full  = _run(preprocess_full(bgr_frame))
    label_tight, prob_tight = _run(preprocess_tight(bgr_frame))

    inf_ms = (time.perf_counter() - t0) * 1000

    if label_full == label_tight:
        # Both agree — average the probabilities for a smoother confidence
        avg_prob  = (prob_full + prob_tight) / 2.0
        label     = "Recyclable" if avg_prob >= THRESHOLD else "Organic"
        conf      = avg_prob if label == "Recyclable" else 1 - avg_prob
        disagreed = False
    else:
        # Disagreement → Recyclable wins (prevents food-on-wrapper FP)
        label     = "Recyclable"
        # Confidence is the full-crop probability for Recyclable
        # (full crop sees packaging context, more reliable signal)
        conf      = prob_full if label_full == "Recyclable" else 1 - prob_full
        disagreed = True

    return label, conf, inf_ms, label_full, label_tight, prob_full, prob_tight, disagreed


# ──────────────────────────────────────────────────────────────────────────────
# BUFFER DRAIN  (fix for predictions being 1 frame behind)
# ──────────────────────────────────────────────────────────────────────────────
def drain_and_capture(cap: cv2.VideoCapture):
    """
    Grab DRAIN_FRAMES frames as fast as possible to flush the camera's
    internal buffer, then do one final cap.read() which reflects the
    actual live scene at the moment the user pressed ENTER.

    cap.grab() decodes nothing — it is extremely fast (~1 ms/frame).
    cap.retrieve() decodes only the last grabbed frame into memory.
    """
    for _ in range(DRAIN_FRAMES):
        cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        # Fallback: try a regular read
        ret, frame = cap.read()
    return ret, frame


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def dequantize(tensor, scale, zero_point):
    if scale == 0:
        return tensor.astype(np.float32)
    return (tensor.astype(np.float32) - zero_point) * scale


def make_filter_grid(fmap4d, max_filters):
    fmap    = fmap4d[0]
    H, W, C = fmap.shape
    n       = min(C, max_filters)
    rows    = math.ceil(n / GRID_COLS)
    cell, pad = CELL_PX, CELL_PAD
    canvas  = np.zeros(
        (rows * (cell + pad) + pad, GRID_COLS * (cell + pad) + pad, 3),
        dtype=np.uint8)
    for i in range(n):
        ch = fmap[:, :, i].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        ch = ((ch - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
             else np.zeros_like(ch, dtype=np.uint8)
        col_img = cv2.applyColorMap(
            cv2.resize(ch, (cell, cell), interpolation=cv2.INTER_NEAREST),
            COLORMAP)
        r, c = divmod(i, GRID_COLS)
        y, x = pad + r * (cell + pad), pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = col_img
    return canvas


def save_heatmaps(ts_folder: str):
    Path(ts_folder).mkdir(parents=True, exist_ok=True)
    saved = []
    for (folder, t_idx, max_f, scale, zp) in layer_index:
        try:
            raw = interpreter.get_tensor(t_idx)
        except Exception as e:
            print(f"  [HEATMAP] Cannot read '{folder}': {e}")
            continue
        if raw is None or raw.ndim != 4:
            continue
        grid      = make_filter_grid(dequantize(raw, scale, zp), max_f)
        layer_dir = os.path.join(ts_folder, folder)
        Path(layer_dir).mkdir(exist_ok=True)
        path = os.path.join(layer_dir, "grid.jpg")
        cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(f"{folder}/grid.jpg")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# RESULT IMAGE DRAWING
# ──────────────────────────────────────────────────────────────────────────────
def draw_result(frame, label, conf, inf_ms, timestamp,
                label_full, label_tight, prob_full, prob_tight,
                disagreed, heatmaps_saved=False,
                is_feedback=False, corrected_label=None):

    out    = frame.copy()
    h, w   = out.shape[:2]
    colour = COLOUR_ENSEMBLE if disagreed else (
             COLOUR_RECYCLABLE if label == "Recyclable" else COLOUR_ORGANIC)

    # ── Top banner ────────────────────────────────────────────────────────────
    banner_h = 185
    overlay  = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Main label
    cv2.putText(out, f"{label}   {conf*100:.1f}%",
                (24, 68), FONT, 2.2, colour, 4, cv2.LINE_AA)

    # Confidence bar
    cv2.rectangle(out, (24, 84), (524, 106), (70, 70, 70), -1)
    cv2.rectangle(out, (24, 84),
                  (24 + int(500 * conf), 106), colour, -1)

    # Per-crop breakdown
    conf_f = prob_full  if label_full  == "Recyclable" else 1 - prob_full
    conf_t = prob_tight if label_tight == "Recyclable" else 1 - prob_tight
    breakdown = (f"Full: {label_full} {conf_f*100:.0f}%   "
                 f"Tight: {label_tight} {conf_t*100:.0f}%")
    if disagreed:
        breakdown += "   [DISAGREED -> Recyclable]"
    cv2.putText(out, breakdown,
                (24, 136), FONT, 0.72, COLOUR_INFO, 1, cv2.LINE_AA)

    # Feedback tag
    if is_feedback and corrected_label:
        cv2.putText(out, f"FEEDBACK: correct = {corrected_label}",
                    (24, 172), FONT, 0.85, COLOUR_FEEDBACK, 2, cv2.LINE_AA)

    # ── Bottom strip ──────────────────────────────────────────────────────────
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, h - 46), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, out, 0.45, 0, out)
    hm_tag = "  [heatmaps]" if heatmaps_saved else ""
    bottom = (f"Inference: {inf_ms:.1f} ms  |  "
              f"Threshold: {THRESHOLD:.4f}  |  {timestamp}{hm_tag}")
    cv2.putText(out, bottom, (24, h - 14), FONT, 0.75,
                COLOUR_INFO, 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SAVE RESULT IMAGE
# ──────────────────────────────────────────────────────────────────────────────
def save_result_image(annotated, label, timestamp):
    safe_ts  = timestamp.replace(":", "").replace(" ", "_").replace("-", "")
    filepath = os.path.join(RESULTS_DIR, f"{safe_ts}_{label}.jpg")
    cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return filepath, safe_ts


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK
# ──────────────────────────────────────────────────────────────────────────────
def save_feedback(raw_frame, correct_label, pred_label, prob_full, timestamp):
    safe_ts  = timestamp.replace(":", "").replace(" ", "_").replace("-", "")
    filepath = os.path.join(FEEDBACK_DIR, correct_label, f"{safe_ts}.jpg")
    cv2.imwrite(filepath, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    entry = dict(timestamp=timestamp, predicted=pred_label,
                 correct=correct_label, prob_full=round(prob_full, 4),
                 image=filepath)
    log = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)
    return filepath


def count_feedback():
    org  = len(list(Path(os.path.join(FEEDBACK_DIR, "Organic")).glob("*.jpg")))
    rec  = len(list(Path(os.path.join(FEEDBACK_DIR, "Recyclable")).glob("*.jpg")))
    return org, rec


# ──────────────────────────────────────────────────────────────────────────────
# SESSION LOG + SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
session_log = []

def log_prediction(label, conf, inf_ms, timestamp, filepath, disagreed):
    session_log.append(dict(timestamp=timestamp, label=label, conf=conf,
                             inf_ms=inf_ms, file=filepath, disagreed=disagreed))


def print_summary():
    total = len(session_log)
    fb_org, fb_rec = count_feedback()
    fb_total = fb_org + fb_rec
    print("\n" + "-" * 64)
    print("  SESSION SUMMARY")
    print("-" * 64)
    if total:
        organic    = sum(1 for r in session_log if r["label"] == "Organic")
        recycl     = total - organic
        avg_ms     = sum(r["inf_ms"] for r in session_log) / total
        disagreed  = sum(1 for r in session_log if r["disagreed"])
        print(f"  Predictions  : {total}  (crops disagreed: {disagreed})")
        print(f"    Organic    : {organic}  ({organic/total*100:.1f}%)")
        print(f"    Recyclable : {recycl}  ({recycl/total*100:.1f}%)")
        print(f"  Avg latency  : {avg_ms:.1f} ms  (2 passes per capture)")
    else:
        print("  No predictions made yet.")
    print("-" * 64)
    print(f"  Feedback (all time): Organic={fb_org}  Recyclable={fb_rec}  "
          f"Total={fb_total}")
    if fb_total > 0:
        needed = max(0, 50 - fb_total)
        if needed == 0:
            print(f"  Enough samples to fine-tune the model.")
            print(f"  scp -r pi@<ip>:{os.path.abspath(FEEDBACK_DIR)}/ .")
        else:
            print(f"  Collect ~{needed} more before retraining ({fb_total}/50).")
    if session_log:
        print("-" * 64)
        print("  Recent predictions:")
        for r in session_log[-8:]:
            flag = " *" if r["disagreed"] else "  "
            print(f"    {r['timestamp']}  {r['label']:<12}  "
                  f"{r['conf']*100:5.1f}%  {r['inf_ms']:6.1f} ms{flag}")
        if any(r["disagreed"] for r in session_log):
            print("  * = crops disagreed, Recyclable tie-break applied")
    print(f"\n  Results  -> {os.path.abspath(RESULTS_DIR)}/")
    print(f"  Feedback -> {os.path.abspath(FEEDBACK_DIR)}/")
    print("-" * 64 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    save_heatmaps_flag = SAVE_HEATMAPS_DEFAULT

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    # Minimise internal buffer so drain_and_capture works efficiently
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fb_org, fb_rec = count_feedback()

    print(f"\nCamera: {actual_w}x{actual_h}")
    print(f"Buffer drain  : {DRAIN_FRAMES} frames before each capture")
    print(f"Dual-crop     : Full (1.0) + Tight ({TIGHT_CROP_RATIO})")
    print(f"Results  -> {os.path.abspath(RESULTS_DIR)}/")
    print(f"Feedback -> {os.path.abspath(FEEDBACK_DIR)}/  "
          f"(Organic: {fb_org}  Recyclable: {fb_rec})")
    if layer_index:
        print(f"Heatmaps : {len(layer_index)} layers matched")
    hm_status = "ON" if save_heatmaps_flag else "OFF"
    print(f"Heatmap saving: {hm_status}")
    print("\n" + "=" * 58)
    print("  Controls (type command, then ENTER):")
    print("    <ENTER>     Capture -> classify -> save")
    print("    f <ENTER>   Flag last prediction as WRONG")
    print("    h <ENTER>   Toggle heatmap saving")
    print("    s <ENTER>   Session summary")
    print("    q <ENTER>   Quit")
    print("=" * 58 + "\n")

    # Interpreter warmup
    print("Warming up interpreter...", end=" ", flush=True)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    interpreter.set_tensor(inp_detail["index"], dummy)
    interpreter.invoke()
    print("ready.\n")

    last = dict(frame=None, label=None, prob_full=None,
                timestamp=None, flagged=False)

    while True:
        try:
            hm_tag = "ON" if save_heatmaps_flag else "OFF"
            cmd = input(
                f"ENTER=capture | f=feedback | h=heatmaps[{hm_tag}] "
                f"| s=summary | q=quit : "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if cmd == "q":
            break

        if cmd == "s":
            print_summary()
            continue

        if cmd == "h":
            save_heatmaps_flag = not save_heatmaps_flag
            print(f"  Heatmap saving: {'ON' if save_heatmaps_flag else 'OFF'}\n")
            continue

        # ── Feedback ──────────────────────────────────────────────────────────
        if cmd == "f":
            if last["frame"] is None:
                print("  No prediction yet — capture something first.\n")
                continue
            if last["flagged"]:
                print("  Feedback already given for this capture.\n")
                continue
            print(f"\n  Last prediction: {last['label']}")
            print("  Correct label?  o=Organic   r=Recyclable")
            try:
                ans = input("  Your answer: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("  Feedback cancelled.\n")
                continue
            if ans in ("o", "organic"):
                correct = "Organic"
            elif ans in ("r", "recyclable"):
                correct = "Recyclable"
            else:
                print("  Unrecognised — enter 'o' or 'r'. Skipped.\n")
                continue
            if correct == last["label"]:
                print(f"  Matches prediction ({correct}). Nothing saved.\n")
                continue
            fb_path = save_feedback(last["frame"], correct,
                                    last["label"], last["prob_full"],
                                    last["timestamp"])
            last["flagged"] = True
            fb_org, fb_rec = count_feedback()
            print(f"  Saved -> {fb_path}")
            print(f"  Total feedback: Organic={fb_org}  Recyclable={fb_rec}")
            needed = max(0, 50 - fb_org - fb_rec)
            if needed == 0:
                print("  Enough samples to fine-tune the model.")
            else:
                print(f"  Collect ~{needed} more before retraining.")
            print()
            continue

        # ── Capture + classify ────────────────────────────────────────────────
        print("  Draining buffer...", end=" ", flush=True)
        ret, frame = drain_and_capture(cap)
        if not ret or frame is None:
            print("FAILED — try again.\n")
            continue
        print("captured.")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        (label, conf, inf_ms,
         label_full, label_tight,
         prob_full, prob_tight,
         disagreed) = predict_dual(frame)

        # Store raw frame before annotation (for feedback)
        last.update(frame=frame.copy(), label=label,
                    prob_full=prob_full, timestamp=timestamp, flagged=False)

        # Heatmaps — tensors valid right after last invoke()
        hm_paths = []
        if save_heatmaps_flag and layer_index:
            safe_ts   = (timestamp.replace(":", "").replace(" ", "_")
                                  .replace("-", ""))
            ts_folder = os.path.join(HEATMAPS_DIR, f"{safe_ts}_{label}")
            hm_paths  = save_heatmaps(ts_folder)

        annotated        = draw_result(
            frame, label, conf, inf_ms, timestamp,
            label_full, label_tight, prob_full, prob_tight,
            disagreed, bool(hm_paths))
        filepath, safe_ts = save_result_image(annotated, label, timestamp)
        log_prediction(label, conf, inf_ms, timestamp, filepath, disagreed)

        # Console output
        bar    = "#" * int(30 * conf) + "-" * (30 - int(30 * conf))
        symbol = "[R]" if label == "Recyclable" else "[O]"
        flag   = "  ** CROPS DISAGREED — Recyclable tie-break **" if disagreed else ""
        print(f"\n  {symbol}  {label:<12}  [{bar}]  {conf*100:5.1f}%"
              f"   {inf_ms:.1f} ms{flag}")
        print(f"       Full : {label_full:<12} "
              f"({(prob_full if label_full=='Recyclable' else 1-prob_full)*100:.1f}%)")
        print(f"      Tight : {label_tight:<12} "
              f"({(prob_tight if label_tight=='Recyclable' else 1-prob_tight)*100:.1f}%)")
        print(f"  Image -> {filepath}")
        if hm_paths:
            hm_base = os.path.join(HEATMAPS_DIR, f"{safe_ts}_{label}")
            print(f"  Heatmaps -> {hm_base}/")
        print(f"  (Type 'f' + ENTER if this prediction is wrong)\n")

    cap.release()
    print_summary()
    print("Camera released. Exiting.")


if __name__ == "__main__":
    main()
