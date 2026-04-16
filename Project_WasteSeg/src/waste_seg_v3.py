# ==============================================================================
# rpi_inference.py — EfficientEdge WasteNet  |  Raspberry Pi 4B Deployment
# ==============================================================================
# Headless / no-display.  Physical push button triggers capture + classify.
# Stepper motor: NEMA23 driven by TB6600 driver directly from RPi GPIO.
#
# TB6600 wiring (common-cathode — RPi 3.3 V GPIOs sink current)
# ──────────────────────────────────────────────────────────────
#   TB6600 PUL+  <──  5 V  (pin 2 or 4)
#   TB6600 PUL-  <──  GPIO 23  (pin 16)   STEP signal
#   TB6600 DIR+  <──  5 V  (pin 2 or 4)
#   TB6600 DIR-  <──  GPIO 24  (pin 18)   DIR signal
#   TB6600 ENA+  <──  5 V  (pin 2 or 4)
#   TB6600 ENA-  <──  GPIO 25  (pin 22)   ENABLE  (LOW = driver ON)
#   TB6600 GND   <──  RPi GND  (pin 6)
#   TB6600 VCC   <──  Separate 24 V / 36 V PSU  (NOT from RPi)
#   TB6600 A+/A- and B+/B-  <──  NEMA23 motor coil pairs
#
#   NOTE: TB6600 optocouplers need ~10–15 mA.  With PUL+/DIR+/ENA+ tied to
#   5 V and GPIO sinking to GND the current is ~(5-0.6)/330 Ω ≈ 13 mA —
#   within GPIO sink limit (16 mA max on RPi 4B).
#   If your TB6600 board has its own 330 Ω resistors on PUL-/DIR-/ENA- you
#   do NOT need external resistors.  Check your board silkscreen.
#
# Push button wiring
# ──────────────────
#   One leg → GPIO 17  (pin 11)
#   Other   → GND      (pin 9)
#   No external resistor needed — internal pull-up enabled in software.
#
# Motor behaviour (matches validated Arduino sketch)
# ──────────────────────────────────────────────────
#   Organic    → Clockwise         500 steps, 5 s pause, reset
#   Recyclable → Counter-Clockwise 500 steps, 5 s pause, reset
#
# Auto-start on boot
# ──────────────────
#   sudo cp wastenet.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable wastenet
#   sudo systemctl start  wastenet
#   journalctl -u wastenet -f
#
# Safe shutdown
# ─────────────
#   sudo shutdown -h now   ← always preferred over hard power cut
#   The script catches SIGTERM and cleans up GPIO + camera before exit.
#
# SSH maintenance commands (while service is running)
#   f  → flag last prediction wrong
#   h  → toggle heatmap saving (off by default — see note below)
#   s  → session summary
#   q  → quit
#
# Prerequisites:
#   pip install numpy opencv-python-headless RPi.GPIO
#   sudo apt install python3-tflite-runtime
# ==============================================================================

import cv2
import json
import numpy as np
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO not found — button and stepper disabled.")

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/waste_net_student_int8.tflite"
IMG_SIZE    = 160
THRESHOLD   = 0.6869    # Replace with Youden's J from training run
NUM_THREADS = 4

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAM_WIDTH    = 1920
CAM_HEIGHT   = 1080
DRAIN_FRAMES = 10       # frames discarded before real capture (buffer flush)

# ── GPIO — push button ────────────────────────────────────────────────────────
BUTTON_PIN    = 17      # BCM  |  physical pin 11  |  other leg to GND
BUTTON_BOUNCE = 400     # debounce ms

# ── GPIO — TB6600 stepper driver (BCM numbers) ────────────────────────────────
STEP_PIN   = 23         # physical pin 16  →  TB6600 PUL-
DIR_PIN    = 24         # physical pin 18  →  TB6600 DIR-
ENABLE_PIN = 25         # physical pin 22  →  TB6600 ENA-  (LOW = driver ON)

# Motor parameters — verified with Arduino sketch:
#   stepsPerMove = 500, stepDelay = 1000 µs
STEPS_PER_MOVE = 500
STEP_DELAY_US  = 1000           # µs per half-cycle  (full cycle = 2 ms)
STEP_PAUSE_S   = 5.0            # seconds to hold at position before reset
STEPPER_ENABLED = True          # False = simulate without moving motor

# Direction: swap HIGH/LOW here if motor turns the wrong way
DIR_CLOCKWISE         = 1       # GPIO.HIGH — Organic
DIR_COUNTER_CLOCKWISE = 0       # GPIO.LOW  — Recyclable

# ── Dual-crop + white-background masking ──────────────────────────────────────
TIGHT_CROP_RATIO   = 0.55
WHITE_BG_THRESHOLD = 230        # lower → mask more aggressively
WHITE_BG_FILL      = (128, 128, 128)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
FEEDBACK_DIR = "feedback"
FEEDBACK_LOG = os.path.join(FEEDBACK_DIR, "feedback_log.json")

# ── Overlay colours (BGR) ─────────────────────────────────────────────────────
COLOUR_ORGANIC    = (94,  197,  34)
COLOUR_RECYCLABLE = (246, 130,  59)
COLOUR_ENSEMBLE   = (0,   220, 255)
COLOUR_INFO       = (255, 255, 255)
COLOUR_FEEDBACK   = (0,   80,  255)
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ──────────────────────────────────────────────────────────────────────────────
for _d in [RESULTS_DIR,
           os.path.join(FEEDBACK_DIR, "Organic"),
           os.path.join(FEEDBACK_DIR, "Recyclable")]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# GPIO SETUP
# ──────────────────────────────────────────────────────────────────────────────
def setup_gpio(cmd_queue: queue.Queue):
    if not GPIO_AVAILABLE:
        return

    # Always clean up first — prevents "Failed to add edge detection" error
    # which occurs when a previous run left GPIO pins configured.
    GPIO.cleanup()

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Stepper driver pins — initialise ENABLE high (driver OFF) until needed
    GPIO.setup(STEP_PIN,   GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_PIN,    GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENABLE_PIN, GPIO.OUT, initial=GPIO.HIGH)

    # Push button — internal pull-up, fires on falling edge (press pulls LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(
        BUTTON_PIN,
        GPIO.FALLING,
        callback=lambda ch: cmd_queue.put("capture"),
        bouncetime=BUTTON_BOUNCE,
    )
    print(f"  GPIO ready | STEP=GPIO{STEP_PIN}  DIR=GPIO{DIR_PIN}  "
          f"ENABLE=GPIO{ENABLE_PIN}  BUTTON=GPIO{BUTTON_PIN}")


def cleanup_gpio():
    if GPIO_AVAILABLE:
        try:
            GPIO.output(ENABLE_PIN, GPIO.HIGH)  # disable driver
        except Exception:
            pass
        GPIO.cleanup()
        print("  GPIO cleaned up.")


# ──────────────────────────────────────────────────────────────────────────────
# STEPPER MOTOR
# ──────────────────────────────────────────────────────────────────────────────
_stepper_lock = threading.Lock()


def _pulse_motor(direction: int, steps: int):
    """
    Enable TB6600 driver, pulse STEP pin `steps` times, then disable.
    Timing mirrors the validated Arduino sketch (stepDelay = 1000 µs).
    """
    if not GPIO_AVAILABLE or not STEPPER_ENABLED:
        label_str = "CW" if direction == DIR_CLOCKWISE else "CCW"
        print(f"  [MOTOR SIM] {label_str}  {steps} steps")
        time.sleep(steps * STEP_DELAY_US * 2 / 1_000_000)
        return

    delay_s = STEP_DELAY_US / 1_000_000

    GPIO.output(DIR_PIN,    direction)
    GPIO.output(ENABLE_PIN, GPIO.LOW)       # TB6600 ENA- LOW = driver enabled
    time.sleep(0.005)                       # 5 ms settle

    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay_s)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay_s)

    GPIO.output(ENABLE_PIN, GPIO.HIGH)      # disable driver (reduces heat)


def actuate_motor(label: str):
    """
    Actuates in a background daemon thread so the main loop stays responsive.
    Organic    → CW  500 steps → pause 5 s → CCW 500 steps (reset)
    Recyclable → CCW 500 steps → pause 5 s → CW  500 steps (reset)
    """
    def _run():
        with _stepper_lock:
            if label == "Organic":
                print("  [MOTOR] Organic     -> CW")
                _pulse_motor(DIR_CLOCKWISE, STEPS_PER_MOVE)
                time.sleep(STEP_PAUSE_S)
                print("  [MOTOR] Resetting   -> CCW")
                _pulse_motor(DIR_COUNTER_CLOCKWISE, STEPS_PER_MOVE)
            else:
                print("  [MOTOR] Recyclable  -> CCW")
                _pulse_motor(DIR_COUNTER_CLOCKWISE, STEPS_PER_MOVE)
                time.sleep(STEP_PAUSE_S)
                print("  [MOTOR] Resetting   -> CW")
                _pulse_motor(DIR_CLOCKWISE, STEPS_PER_MOVE)
            print("  [MOTOR] Done.")

    threading.Thread(target=_run, daemon=True, name="stepper").start()


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model not found: {path}")
    print(f"Loading model: {path}")
    interp = tflite.Interpreter(model_path=path, num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input  : {inp['shape']}  dtype={inp['dtype'].__name__}")
    print(f"  Output : {out['shape']}  dtype={out['dtype'].__name__}")
    print(f"  Threads: {NUM_THREADS}   Threshold: {THRESHOLD}")
    return interp, inp, out


interpreter, inp_detail, out_detail = load_model(MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — dual crop
# ──────────────────────────────────────────────────────────────────────────────
def _mask_white_bg(bgr: np.ndarray) -> np.ndarray:
    """
    Replace near-white pixels (all 3 channels > threshold) with neutral gray.
    Prevents white table/plate surround from being read as paper packaging.
    Only applied to the full (zoomed-out) crop.
    """
    b, g, r = cv2.split(bgr)
    mask = ((b.astype(np.uint16) + g.astype(np.uint16) + r.astype(np.uint16))
            > WHITE_BG_THRESHOLD * 3).astype(np.uint8)
    result = bgr.copy()
    result[mask == 1] = WHITE_BG_FILL
    return result


def _crop_to_blob(bgr: np.ndarray, ratio: float) -> np.ndarray:
    h, w   = bgr.shape[:2]
    side   = int(min(h, w) * ratio)
    cy, cx = h // 2, w // 2
    y0, x0 = cy - side // 2, cx - side // 2
    patch  = bgr[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis]


def preprocess_full(bgr):  return _crop_to_blob(_mask_white_bg(bgr), 1.0)
def preprocess_tight(bgr): return _crop_to_blob(bgr, TIGHT_CROP_RATIO)


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE — dual crop + fusion
# ──────────────────────────────────────────────────────────────────────────────
def _run(blob):
    interpreter.set_tensor(inp_detail["index"], blob)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(out_detail["index"])[0, 0])
    return ("Recyclable" if prob >= THRESHOLD else "Organic"), prob


def predict_dual(bgr_frame):
    """
    Full crop (bg-masked) + tight crop.
    Agreement   → average probs.
    Disagreement → Recyclable wins (food-image-on-wrapper tie-break).
    White-bg masking resolves the organic-on-white-background false positive
    upstream so the tie-break is only reached for the wrapper case.
    """
    t0 = time.perf_counter()
    label_full,  prob_full  = _run(preprocess_full(bgr_frame))
    label_tight, prob_tight = _run(preprocess_tight(bgr_frame))
    inf_ms = (time.perf_counter() - t0) * 1000

    if label_full == label_tight:
        avg   = (prob_full + prob_tight) / 2.0
        label = "Recyclable" if avg >= THRESHOLD else "Organic"
        conf  = avg if label == "Recyclable" else 1 - avg
        disagreed = False
    else:
        label     = "Recyclable"
        conf      = prob_full if label_full == "Recyclable" else 1 - prob_full
        disagreed = True

    return (label, conf, inf_ms,
            label_full, label_tight, prob_full, prob_tight, disagreed)


# ──────────────────────────────────────────────────────────────────────────────
# BUFFER DRAIN  (fixes 1-frame-behind bug)
# ──────────────────────────────────────────────────────────────────────────────
def drain_and_capture(cap):
    for _ in range(DRAIN_FRAMES):
        cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        ret, frame = cap.read()
    return ret, frame


# ──────────────────────────────────────────────────────────────────────────────
# RESULT IMAGE DRAWING
# ──────────────────────────────────────────────────────────────────────────────
def draw_result(frame, label, conf, inf_ms, timestamp,
                label_full, label_tight, prob_full, prob_tight,
                disagreed, is_feedback=False, corrected_label=None):
    out    = frame.copy()
    h, w   = out.shape[:2]
    colour = COLOUR_ENSEMBLE if disagreed else (
             COLOUR_RECYCLABLE if label == "Recyclable" else COLOUR_ORGANIC)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 175), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    cv2.putText(out, f"{label}   {conf*100:.1f}%",
                (24, 68), FONT, 2.2, colour, 4, cv2.LINE_AA)

    cv2.rectangle(out, (24, 84), (524, 106), (70, 70, 70), -1)
    cv2.rectangle(out, (24, 84),
                  (24 + int(500 * conf), 106), colour, -1)

    cf = prob_full  if label_full  == "Recyclable" else 1 - prob_full
    ct = prob_tight if label_tight == "Recyclable" else 1 - prob_tight
    line2 = (f"Full: {label_full} {cf*100:.0f}%   "
             f"Tight: {label_tight} {ct*100:.0f}%")
    if disagreed:
        line2 += "   [disagree -> Recyclable]"
    cv2.putText(out, line2, (24, 132), FONT, 0.72, COLOUR_INFO, 1, cv2.LINE_AA)

    if is_feedback and corrected_label:
        cv2.putText(out, f"FEEDBACK: correct = {corrected_label}",
                    (24, 165), FONT, 0.80, COLOUR_FEEDBACK, 2, cv2.LINE_AA)

    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, out, 0.45, 0, out)
    cv2.putText(out,
                f"Inference: {inf_ms:.1f} ms  |  Threshold: {THRESHOLD:.4f}"
                f"  |  {timestamp}",
                (24, h - 14), FONT, 0.72, COLOUR_INFO, 1, cv2.LINE_AA)
    return out


def save_result_image(annotated, label, timestamp):
    safe = timestamp.replace(":", "").replace(" ", "_").replace("-", "")
    path = os.path.join(RESULTS_DIR, f"{safe}_{label}.jpg")
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path, safe


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK
# ──────────────────────────────────────────────────────────────────────────────
def save_feedback(raw_frame, correct_label, pred_label, prob_full, timestamp):
    safe = timestamp.replace(":", "").replace(" ", "_").replace("-", "")
    path = os.path.join(FEEDBACK_DIR, correct_label, f"{safe}.jpg")
    cv2.imwrite(path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG) as f: log = json.load(f)
        except Exception: log = []
    log.append(dict(timestamp=timestamp, predicted=pred_label,
                    correct=correct_label, prob_full=round(prob_full, 4),
                    image=path))
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)
    return path


def count_feedback():
    org = len(list(Path(os.path.join(FEEDBACK_DIR, "Organic")).glob("*.jpg")))
    rec = len(list(Path(os.path.join(FEEDBACK_DIR, "Recyclable")).glob("*.jpg")))
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
    fb_o, fb_r = count_feedback()
    print("\n" + "-" * 62)
    print("  SESSION SUMMARY")
    print("-" * 62)
    if total:
        org  = sum(1 for r in session_log if r["label"] == "Organic")
        rec  = total - org
        avg  = sum(r["inf_ms"] for r in session_log) / total
        disc = sum(1 for r in session_log if r["disagreed"])
        print(f"  Predictions  : {total}  (tie-breaks: {disc})")
        print(f"    Organic    : {org}  ({org/total*100:.1f}%)")
        print(f"    Recyclable : {rec}  ({rec/total*100:.1f}%)")
        print(f"  Avg latency  : {avg:.1f} ms")
    else:
        print("  No predictions yet.")
    print("-" * 62)
    print(f"  Feedback: Organic={fb_o}  Recyclable={fb_r}  "
          f"Total={fb_o+fb_r}")
    if fb_o + fb_r > 0:
        needed = max(0, 50 - fb_o - fb_r)
        if needed == 0:
            print(f"  Ready to retrain.")
            print(f"  scp -r pi@<ip>:{os.path.abspath(FEEDBACK_DIR)}/ .")
        else:
            print(f"  Collect ~{needed} more before retraining.")
    if session_log:
        print("-" * 62)
        for r in session_log[-8:]:
            flag = " *" if r["disagreed"] else "  "
            print(f"    {r['timestamp']}  {r['label']:<12}"
                  f"  {r['conf']*100:5.1f}%  {r['inf_ms']:6.1f} ms{flag}")
    print(f"\n  Results  -> {os.path.abspath(RESULTS_DIR)}/")
    print(f"  Feedback -> {os.path.abspath(FEEDBACK_DIR)}/")
    print("-" * 62 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# KEYBOARD THREAD (SSH maintenance)
# ──────────────────────────────────────────────────────────────────────────────
def keyboard_thread(cmd_queue: queue.Queue):
    while True:
        try:
            cmd = input().strip().lower()
            cmd_queue.put(cmd)
        except (EOFError, OSError):
            break


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    shutdown_event = threading.Event()
    cmd_queue      = queue.Queue()

    def _handle_signal(signum, frame):
        print(f"\n[SIGNAL] {signum} received — shutting down...")
        shutdown_event.set()
        cmd_queue.put("q")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    # GPIO (GPIO.cleanup() is called inside setup_gpio before init)
    setup_gpio(cmd_queue)

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cleanup_gpio()
        sys.exit(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fb_o, fb_r = count_feedback()

    print(f"\nCamera   : {actual_w}x{actual_h}")
    print(f"Motor    : STEP=GPIO{STEP_PIN}  DIR=GPIO{DIR_PIN}  "
          f"ENABLE=GPIO{ENABLE_PIN}  "
          f"({'active' if STEPPER_ENABLED else 'SIMULATED'})")
    print(f"Button   : GPIO{BUTTON_PIN}")
    print(f"Results  : {os.path.abspath(RESULTS_DIR)}/")
    print(f"Feedback : {os.path.abspath(FEEDBACK_DIR)}/  "
          f"(O:{fb_o}  R:{fb_r})")
    print("\n" + "=" * 54)
    print("  READY — press button to capture and sort")
    print("  SSH: f=feedback  s=summary  q=quit")
    print("=" * 54 + "\n")

    # Warmup
    print("Warming up...", end=" ", flush=True)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    interpreter.set_tensor(inp_detail["index"], dummy)
    interpreter.invoke()
    print("ready.\n")

    threading.Thread(target=keyboard_thread, args=(cmd_queue,),
                     daemon=True, name="keyboard").start()

    last = dict(frame=None, label=None, prob_full=None,
                timestamp=None, flagged=False, awaiting_feedback=False)

    while not shutdown_event.is_set():
        try:
            cmd = cmd_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if cmd == "q":
            break

        if cmd == "s":
            print_summary()
            continue

        # ── Feedback answer (o / r entered after 'f') ─────────────────────────
        if last["awaiting_feedback"]:
            last["awaiting_feedback"] = False
            if cmd in ("o", "organic"):
                correct = "Organic"
            elif cmd in ("r", "recyclable"):
                correct = "Recyclable"
            else:
                print("  Unrecognised — enter 'o' or 'r'. Skipped.\n")
                continue
            if correct == last["label"]:
                print(f"  Matches prediction ({correct}). Nothing saved.\n")
            elif not last["flagged"]:
                fp = save_feedback(last["frame"], correct,
                                   last["label"], last["prob_full"],
                                   last["timestamp"])
                last["flagged"] = True
                fb_o, fb_r = count_feedback()
                print(f"  Saved -> {fp}")
                print(f"  Feedback total: Organic={fb_o}  Recyclable={fb_r}")
                needed = max(0, 50 - fb_o - fb_r)
                print(f"  {'Ready to retrain.' if needed == 0 else f'~{needed} more needed.'}\n")
            continue

        # ── Flag last prediction ───────────────────────────────────────────────
        if cmd == "f":
            if last["frame"] is None:
                print("  No prediction yet.\n")
            elif last["flagged"]:
                print("  Already flagged.\n")
            else:
                print(f"  Last: {last['label']}  |  Correct?  o=Organic  r=Recyclable")
                last["awaiting_feedback"] = True
            continue

        # ── Capture (button or bare ENTER) ────────────────────────────────────
        if cmd not in ("capture", ""):
            continue

        if _stepper_lock.locked():
            print("  [SKIP] Motor still moving.\n")
            continue

        print("  Capturing...", end=" ", flush=True)
        ret, frame = drain_and_capture(cap)
        if not ret or frame is None:
            print("FAILED — try again.\n")
            continue
        print("ok.")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (label, conf, inf_ms,
         lf, lt, pf, pt, disagreed) = predict_dual(frame)

        last.update(frame=frame.copy(), label=label, prob_full=pf,
                    timestamp=timestamp, flagged=False,
                    awaiting_feedback=False)

        annotated        = draw_result(frame, label, conf, inf_ms, timestamp,
                                       lf, lt, pf, pt, disagreed)
        filepath, safe_ts = save_result_image(annotated, label, timestamp)
        log_prediction(label, conf, inf_ms, timestamp, filepath, disagreed)

        actuate_motor(label)

        bar    = "#" * int(30 * conf) + "-" * (30 - int(30 * conf))
        symbol = "[R]" if label == "Recyclable" else "[O]"
        flag   = "  ** tie-break **" if disagreed else ""
        cf = pf if lf == "Recyclable" else 1 - pf
        ct = pt if lt == "Recyclable" else 1 - pt
        print(f"\n  {symbol}  {label:<12}  [{bar}]  {conf*100:5.1f}%"
              f"   {inf_ms:.1f} ms{flag}")
        print(f"       Full : {lf} {cf*100:.1f}%")
        print(f"      Tight : {lt} {ct*100:.1f}%")
        print(f"  Saved -> {filepath}")
        print(f"  (SSH: type 'f' to flag as wrong)\n")

    # Cleanup
    cap.release()
    cleanup_gpio()
    print_summary()
    print("Exiting.")


if __name__ == "__main__":
    main()
