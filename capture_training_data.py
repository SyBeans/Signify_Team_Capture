"""
capture_training_data.py (FINAL + Selective Add + Cross-Member Help + FAST)
Per-sample confirm + sample count menu + FULL resume support.

NEW in this version:
  ⭐ Time-based recording (4.0s wall clock) — FPS-independent
  ⭐ MediaPipe downsampled to 320x240 — 4x faster, same landmarks
  ⭐ Timestamp-based resampling — consistent velocity features
  ⭐ Auto-retry on too-slow captures (MIN_FRAMES_OK = 20)

CLI modes:
  (default)              → capture all incomplete signs
  --list                 → show status of all your assigned signs (no camera)
  --add 3,7,11           → APPEND more samples to specific sign IDs
  --add 3,7,11 --add-count 10   → append exactly N per sign (skips menu)
  --force                → allow --add on sign IDs outside your assignment
  --redo 42              → WIPE + recapture specific sign ID
  --fresh                → wipe ALL data and start over
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import json
import argparse
from datetime import datetime
import time
import sys  # ⭐ ADDED: for clean exit

# ============================================
# ARGUMENTS
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Your name")
parser.add_argument("--output", default="team_data", help="Output folder")
parser.add_argument("--redo", type=int, default=None, help="Wipe + recapture specific sign ID")
parser.add_argument("--add", type=str, default=None,
                    help="Append more samples to sign IDs (comma-separated, e.g. 3,7,11)")
parser.add_argument("--add-count", type=int, default=None,
                    help="How many more samples per sign (used with --add; skips menu)")
parser.add_argument("--force", action="store_true",
                    help="Allow --add on sign IDs outside your assignment "
                         "(data goes to YOUR folder, correct label — audit trail safe)")
parser.add_argument("--list", action="store_true", help="Show status of all assigned signs")
parser.add_argument("--fresh", action="store_true", help="Start fresh (delete old)")
args = parser.parse_args()

MEMBER_NAME = args.name.lower()
OUTPUT_DIR = f"{args.output}/{MEMBER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# LOAD ASSIGNMENTS
# ============================================
with open("team_assignments.json", "r") as f:
    cfg = json.load(f)

settings = cfg["capture_settings"]
FPS = settings["fps"]
TOTAL_FRAMES = settings["total_frames_captured"]    # legacy — kept for compat
MODEL_FRAMES = settings["model_frames"]             # 30 — resample target
MAX_HANDS = settings["max_hands"]

# ⭐ NEW: time-based recording
RECORD_SECONDS = float(settings.get("duration_seconds", 4.0))
MIN_FRAMES_OK = 20   # retry if fewer frames captured
PROCESS_WIDTH = 320  # ⭐ downsampled MediaPipe input
PROCESS_HEIGHT = 240

member_data = cfg["phase_2_fsl_split"].get(MEMBER_NAME)
if not member_data:
    print(f"❌ Member '{MEMBER_NAME}' not found!")
    print(f"   Available: {list(cfg['phase_2_fsl_split'].keys())}")
    exit(1)

SIGN_LIST = list(zip(member_data["signs"], member_data["sign_names"]))

# ============================================
# FSL-105 FULL LABEL TABLE (for --force)
# ============================================
def load_full_fsl_labels():
    candidates = [
        "../Signify_Model/landmarks/hand/labels.json",
        "../Signify_Model/landmarks/merged/master_labels.json",
        "../Signify_Model/landmarks/merged/labels.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                labels = json.load(f)
            if len(labels) >= 105:
                return labels[:105]
    fallback = {}
    for m in cfg["phase_2_fsl_split"].values():
        for sid, name in zip(m["signs"], m["sign_names"]):
            fallback[sid] = name
    if fallback:
        return [fallback[i] for i in range(105)]
    return None

FSL_105_LABELS = load_full_fsl_labels()

# ============================================
# PATHS
# ============================================
PROGRESS_PATH = f"{OUTPUT_DIR}/progress.json"
X_PATH = f"{OUTPUT_DIR}/X_partial.npy"
Y_PATH = f"{OUTPUT_DIR}/y_partial.npy"
LABELS_PATH = f"{OUTPUT_DIR}/labels.json"


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"member": MEMBER_NAME, "signs": {}, "last_sign_id": None}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def load_existing_data():
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X = list(np.load(X_PATH))
        y = list(np.load(Y_PATH))
        return X, y
    return [], []


def save_partial_data(X_all, y_all):
    if len(X_all) > 0:
        np.save(X_PATH, np.array(X_all, dtype=np.float32))
        np.save(Y_PATH, np.array(y_all, dtype=np.int32))


def clear_all_data():
    for p in [X_PATH, Y_PATH, LABELS_PATH, PROGRESS_PATH]:
        if os.path.exists(p):
            os.remove(p)


# ============================================
# MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_FEATURES = 63

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1   # keep Full model (accuracy). Set to 0 for Lite if too slow
)


# ============================================
# SIGN STATUS (--list)
# ============================================
def show_sign_status(progress):
    print(f"\n{'=' * 64}")
    print(f"📋 Sign Status — {MEMBER_NAME.upper()} ({len(SIGN_LIST)} signs)")
    print(f"{'=' * 64}")
    print(f"{'ID':>4} | {'Name':<22} | {'Samples':>8} | Done")
    print("-" * 64)

    total_samples = 0
    completed = 0
    for sid, name in SIGN_LIST:
        entry = progress["signs"].get(str(sid), {})
        samples = entry.get("samples", 0)
        done = entry.get("completed", False)
        total_samples += samples
        if done:
            completed += 1
        mark = "✅" if done else "❌"
        print(f"{sid:>4} | {name:<22} | {samples:>8} | {mark}")

    print("-" * 64)
    print(f"Completed: {completed}/{len(SIGN_LIST)} | Total samples: {total_samples}")
    print(f"{'=' * 64}")
    print("\nTo add more samples to specific signs:")
    print(f"   python {os.path.basename(__file__)} --name {MEMBER_NAME} --add 3,7,11 --add-count 10")
    print(f"To help with a sign outside your assignment:")
    print(f"   python {os.path.basename(__file__)} --name {MEMBER_NAME} --add 7 --add-count 10 --force")
    print(f"To wipe and recapture a sign:")
    print(f"   python {os.path.basename(__file__)} --name {MEMBER_NAME} --redo 42")
    print(f"{'=' * 64}\n")


# ============================================
# SAMPLE COUNT MENU (15/40)
# ============================================
def choose_sample_count(sign_name):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"SIGN: {sign_name}",
                    (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(frame, "How many samples?",
                    (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 200), (240, 270), (0, 150, 0), -1)
        cv2.putText(frame, "1 = 15 samples",
                    (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 200), (460, 270), (0, 150, 200), -1)
        cv2.putText(frame, "2 = 40 samples",
                    (300, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 310), (460, 380), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Quit",
                    (220, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'): return 15
        elif key == ord('2'): return 40
        elif key == ord('q') or key == ord('Q'): return None


# ============================================
# ADD-COUNT MENU
# ============================================
def choose_add_count(sign_name, current_count):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"SIGN: {sign_name}",
                    (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Current: {current_count} samples",
                    (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "How many MORE to add?",
                    (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 220), (240, 290), (0, 150, 0), -1)
        cv2.putText(frame, "1 = +5", (120, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 220), (460, 290), (0, 150, 200), -1)
        cv2.putText(frame, "2 = +10", (320, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 330), (240, 400), (150, 100, 0), -1)
        cv2.putText(frame, "3 = +20", (120, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 330), (460, 400), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Cancel", (320, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'): return 5
        elif key == ord('2'): return 10
        elif key == ord('3'): return 20
        elif key == ord('q') or key == ord('Q'): return None


# ============================================
# PER-SAMPLE MENU
# ============================================
def show_sample_menu(sign_name, sample_num, total_samples):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"SIGN: {sign_name}",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{total_samples}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Accept this sample?",
                    (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.rectangle(frame, (40, 190), (300, 260), (0, 180, 0), -1)
        cv2.putText(frame, "SPACE = Accept",
                    (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (340, 190), (600, 260), (200, 150, 0), -1)
        cv2.putText(frame, "R = Redo",
                    (390, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (40, 290), (300, 360), (200, 0, 0), -1)
        cv2.putText(frame, "N = Skip",
                    (100, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (340, 290), (600, 360), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Save & Quit",
                    (360, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '): return "accept"
        elif key in (ord('r'), ord('R')): return "redo"
        elif key in (ord('n'), ord('N')): return "skip"
        elif key in (ord('q'), ord('Q')): return "quit"


# ============================================
# HELPER — extract landmarks vector from valid_hands
# ============================================
def extract_landmarks_vec(valid_hands):
    hand1 = []
    for lm in valid_hands[0].landmark:
        hand1.extend([lm.x, lm.y, lm.z])
    if len(valid_hands) >= 2:
        hand2 = []
        for lm in valid_hands[1].landmark:
            hand2.extend([lm.x, lm.y, lm.z])
    else:
        hand2 = [0.0] * HAND_FEATURES
    return hand1 + hand2


# ============================================
# ⭐ CAPTURE ONE SAMPLE — TIME-BASED + FAST
# ============================================
def capture_one_sample(cap, sign_name, sample_num, total_samples):
    """
    Returns:
      - ndarray (30, 252) if successful
      - None if user cancelled (ESC)
      - "RETRY" if too slow (<MIN_FRAMES_OK frames in RECORD_SECONDS)
    """
    fps = 0.0
    fps_frames = 0
    fps_last_time = time.time()

    # --- PREVIEW LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ⭐ Downsample for MediaPipe
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        preview = hands.process(frame_rgb)

        valid_hands = []
        if preview.multi_hand_landmarks:
            valid_hands.append(preview.multi_hand_landmarks[0])
            if len(preview.multi_hand_landmarks) >= 2:
                wrist1 = preview.multi_hand_landmarks[0].landmark[0]
                wrist2 = preview.multi_hand_landmarks[1].landmark[0]
                dist = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
                if dist > 0.12:
                    valid_hands.append(preview.multi_hand_landmarks[1])

        if valid_hands:
            for hl in valid_hands:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
                )
                for idx, lm in enumerate(hl.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        fps_frames += 1
        now = time.time()
        if now - fps_last_time >= 0.5:
            fps = fps_frames / (now - fps_last_time)
            fps_frames = 0
            fps_last_time = now

        if valid_hands:
            status_color = (0, 255, 0)
            status_text = f"🟢 HAND ({len(valid_hands)})"
        else:
            status_color = (0, 0, 255)
            status_text = "🔴 SHOW HAND!"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"Sign: {sign_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{total_samples}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"SPACE = record ({RECORD_SECONDS:.0f}s) | ESC = skip", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == 27:
            return None

    # --- COUNTDOWN (drain buffer) ---
    for i in range(3, 0, -1):
        for _ in range(3):  # drain stale frames
            cap.read()
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}...", (260, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1000)

    # --- ⭐ RECORDING LOOP — TIME-BASED ---
    captured = []  # list of (timestamp_s, landmarks_vec)
    start_time = time.time()
    fps_frames = 0
    fps_last_time = start_time

    while (time.time() - start_time) < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time() - start_time
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ⭐ Downsample for MediaPipe
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        valid_hands = []
        if results.multi_hand_landmarks:
            valid_hands.append(results.multi_hand_landmarks[0])
            if len(results.multi_hand_landmarks) >= 2:
                wrist1 = results.multi_hand_landmarks[0].landmark[0]
                wrist2 = results.multi_hand_landmarks[1].landmark[0]
                dist = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
                if dist > 0.12:
                    valid_hands.append(results.multi_hand_landmarks[1])

        if valid_hands:
            for hl in valid_hands:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
                )
                for idx, lm in enumerate(hl.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            captured.append((timestamp, extract_landmarks_vec(valid_hands)))

        fps_frames += 1
        now = time.time()
        if now - fps_last_time >= 0.5:
            fps = fps_frames / (now - fps_last_time)
            fps_frames = 0
            fps_last_time = now

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        elapsed = time.time() - start_time
        progress = min(elapsed / RECORD_SECONDS, 1.0)

        cv2.putText(frame, f"RECORDING {elapsed:.1f}/{RECORD_SECONDS:.1f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Sign: {sign_name}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if valid_hands:
            status_color = (0, 255, 0)
            status_text = f"🟢 HAND ({len(valid_hands)})"
        else:
            status_color = (0, 0, 255)
            status_text = "🔴 NO HAND!"

        cv2.putText(frame, status_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"Frames captured: {len(captured)}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        bar_w = int(progress * (w - 40))
        cv2.rectangle(frame, (20, 440), (20 + bar_w, 470), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 440), (w - 20, 470), (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1)

    # --- ⭐ CHECK: enough frames captured? ---
    if len(captured) < MIN_FRAMES_OK:
        print(f"   ⚠️  Only {len(captured)} frames captured in {RECORD_SECONDS}s — too slow")
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 200), (w, 280), (0, 0, 150), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.putText(frame, f"TOO SLOW: {len(captured)} frames",
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(frame, "Please redo this sample",
                        (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Signify Capture", frame)
            cv2.waitKey(1500)
        return "RETRY"

    # --- ⭐ TIME-BASED RESAMPLE to MODEL_FRAMES ---
    captured.sort(key=lambda x: x[0])
    target_times = np.linspace(0.0, RECORD_SECONDS, MODEL_FRAMES)

    selected = []
    for t in target_times:
        nearest_idx = min(range(len(captured)), key=lambda i: abs(captured[i][0] - t))
        selected.append(captured[nearest_idx][1])

    positions = np.array(selected, dtype=np.float32)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    return np.concatenate([positions, velocities], axis=1)


# ============================================
# ADD MODE
# ============================================
def resolve_sign_id(sid):
    matching = [(s, n) for s, n in SIGN_LIST if s == sid]
    if matching:
        return matching[0]
    if args.force and FSL_105_LABELS is not None:
        if 0 <= sid < len(FSL_105_LABELS):
            return (sid, FSL_105_LABELS[sid])
    return (None, None)


def run_add_mode(add_ids, add_count, X_all, y_all, progress):
    signs_to_process = []
    forced_signs = []

    for sid in add_ids:
        resolved_sid, resolved_name = resolve_sign_id(sid)
        if resolved_sid is None:
            if not args.force:
                print(f"⚠️  Sign ID {sid} not in {MEMBER_NAME}'s assignment — skipping "
                      f"(use --force to help with signs outside your range)")
            else:
                print(f"⚠️  Sign ID {sid} out of range for FSL-105 (0–104) — skipping")
            continue
        signs_to_process.append((resolved_sid, resolved_name))
        in_own = any(s == sid for s, _ in SIGN_LIST)
        if not in_own:
            forced_signs.append(sid)

    if not signs_to_process:
        print("❌ No valid sign IDs to process")
        return

    print(f"\n{'=' * 60}")
    print(f"➕ ADD MODE — {len(signs_to_process)} sign(s)")
    print(f"{'=' * 60}")
    for sid, name in signs_to_process:
        entry = progress["signs"].get(str(sid), {})
        tag = "  🔓 (--force: not in your assignment)" if sid in forced_signs else ""
        print(f"   {sid:>3}: {name:<22} (current: {entry.get('samples', 0)}){tag}")

    if forced_signs:
        print(f"\n⚠️  {len(forced_signs)} sign(s) outside your assignment.")
        print(f"   Data WILL be saved to team_data/{MEMBER_NAME}/ with correct labels.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    try:
        for sign_id, sign_name in signs_to_process:
            entry = progress["signs"].get(str(sign_id), {})
            existing_count = entry.get("samples", 0)

            if add_count is not None:
                additional = add_count
            else:
                additional = choose_add_count(sign_name, existing_count)
                if additional is None:
                    print(f"   ⏭️  Skipped {sign_name}")
                    continue

            total_samples = existing_count + additional
            start_from = existing_count

            print(f"\n{'=' * 60}")
            print(f"📝 {sign_name}: +{additional} more ({existing_count} → {total_samples})")
            print(f"{'=' * 60}")

            samples_captured = 0
            while samples_captured < additional:
                current_num = start_from + samples_captured + 1
                sample = capture_one_sample(cap, sign_name, current_num, total_samples)

                # ⭐ Handle RETRY / cancel
                if sample == "RETRY":
                    print(f"   ⚠️  Sample {current_num} too slow — retrying same sample")
                    continue
                if sample is None:
                    print(f"   ⏭️  Sample {current_num} cancelled")
                    break

                choice = show_sample_menu(sign_name, current_num, total_samples)

                if choice == "accept":
                    X_all.append(sample)
                    y_all.append(sign_id)
                    samples_captured += 1
                    print(f"   ✅ Added {current_num}/{total_samples}")
                    progress["signs"][str(sign_id)] = {
                        "name": sign_name,
                        "samples": start_from + samples_captured,
                        "total_samples": total_samples,
                        "completed": True,
                    }
                    progress["last_sign_id"] = sign_id
                    save_partial_data(X_all, y_all)
                    save_progress(progress)
                elif choice == "redo":
                    continue
                elif choice == "skip":
                    samples_captured += 1
                    continue
                elif choice == "quit":
                    raise KeyboardInterrupt

            print(f"   ✅ {sign_name} → {start_from + samples_captured} total samples")

    except KeyboardInterrupt:
        print("\n⚠️  Add mode interrupted — partial data saved")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        save_partial_data(X_all, y_all)
        save_progress(progress)
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)   # <-- Forces Linux to close the window
        cv2.waitKey(1)
        print("✅ Camera released.")
        sys.exit(0)      # <-- Kills the Python process cleanly


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 60)
    print(f"🎬 SIGNIFY CAPTURE — {MEMBER_NAME.upper()}")
    print(f"   Total signs: {len(SIGN_LIST)}")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"   Recording: {RECORD_SECONDS:.1f}s wall time (FPS-independent)")
    print(f"   MediaPipe input: {PROCESS_WIDTH}x{PROCESS_HEIGHT} (fast mode)")
    if args.force:
        print(f"   🔓 --force enabled")
    print("=" * 60)

    if args.fresh:
        print("🆕 --fresh: deleting all previous data...")
        clear_all_data()

    X_all, y_all = load_existing_data()
    progress = load_progress()

    if X_all:
        print(f"📂 Found existing data: {len(X_all)} samples")

    # MODE 1 — LIST
    if args.list:
        show_sign_status(progress)
        return

    # MODE 2 — ADD
    if args.add is not None:
        add_ids = [int(x.strip()) for x in args.add.split(",")]
        run_add_mode(add_ids, args.add_count, X_all, y_all, progress)
        return

    # MODE 3/4 — REDO or NORMAL
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Check actual FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera reports {actual_fps:.1f} FPS")

    try:
        if args.redo is not None:
            target_id = args.redo
            old_count = sum(1 for y in y_all if y == target_id)
            X_all[:] = [x for x, y in zip(X_all, y_all) if y != target_id]
            y_all[:] = [y for y in y_all if y != target_id]
            if str(target_id) in progress["signs"]:
                del progress["signs"][str(target_id)]
            pending_signs = [(sid, name) for sid, name in SIGN_LIST if sid == target_id]
            print(f"🔁 Redo mode: sign {target_id} — wiped {old_count} old samples")
        else:
            pending_signs = []
            for sid, name in SIGN_LIST:
                entry = progress["signs"].get(str(sid), {})
                if not entry.get("completed"):
                    pending_signs.append((sid, name))
            print(f"📋 Signs to capture: {len(pending_signs)}/{len(SIGN_LIST)}")

        if not pending_signs:
            print("✅ All signs complete!")
            cap.release()
            cv2.destroyAllWindows()
            return

        for idx, (sign_id, sign_name) in enumerate(pending_signs):
            print(f"\n{'=' * 60}")
            print(f"📝 SIGN {idx+1}/{len(pending_signs)}: {sign_name}")
            print(f"{'=' * 60}")

            entry = progress["signs"].get(str(sign_id), {})
            in_progress = entry.get("samples", 0) > 0 and not entry.get("completed")
            skip_this_sign = False

            if in_progress:
                existing_count = entry["samples"]
                stored_total = entry.get("total_samples", 20)
                print(f"⚠️  This sign has {existing_count}/{stored_total} samples in progress.")
                print("   [C] Continue   [R] Redo all   [S] Skip")

                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, f"SIGN: {sign_name}", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Already: {existing_count}/{stored_total} samples",
                            (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Resume what?", (40, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                cv2.rectangle(frame, (40, 200), (300, 270), (0, 180, 0), -1)
                cv2.putText(frame, "C = Continue", (80, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.rectangle(frame, (340, 200), (600, 270), (200, 150, 0), -1)
                cv2.putText(frame, "R = Redo all", (380, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.rectangle(frame, (40, 290), (300, 360), (100, 100, 100), -1)
                cv2.putText(frame, "S = Skip sign", (80, 335),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.imshow("Signify Capture", frame)
                key = cv2.waitKey(0) & 0xFF

                if key in (ord('c'), ord('C')):
                    total_samples = stored_total
                    start_from = existing_count
                    additional = total_samples - start_from
                elif key in (ord('r'), ord('R')):
                    X_all[:] = [x for x, y in zip(X_all, y_all) if y != sign_id]
                    y_all[:] = [y for y in y_all if y != sign_id]
                    total_samples = choose_sample_count(sign_name + " (redo)")
                    if total_samples is None:
                        raise KeyboardInterrupt
                    start_from = 0
                    additional = total_samples
                elif key in (ord('s'), ord('S')):
                    skip_this_sign = True
                    total_samples = 0
                    start_from = 0
                    additional = 0
                else:
                    total_samples = 0
                    start_from = 0
                    additional = 0
            else:
                total_samples = choose_sample_count(sign_name)
                if total_samples is None:
                    raise KeyboardInterrupt
                start_from = 0
                additional = total_samples

            if skip_this_sign:
                print(f"   ⏭️  Skipped")
                continue

            samples_captured = 0
            while samples_captured < additional:
                current_num = start_from + samples_captured + 1
                sample = capture_one_sample(cap, sign_name, current_num, total_samples)

                # ⭐ Handle RETRY / cancel
                if sample == "RETRY":
                    print(f"   ⚠️  Sample {current_num} too slow — retrying same sample")
                    continue
                if sample is None:
                    print(f"   ⏭️  Sample {current_num} cancelled")
                    break

                choice = show_sample_menu(sign_name, current_num, total_samples)

                if choice == "accept":
                    X_all.append(sample)
                    y_all.append(sign_id)
                    samples_captured += 1
                    print(f"   ✅ Sample {current_num}/{total_samples} saved")
                    progress["signs"][str(sign_id)] = {
                        "name": sign_name,
                        "samples": start_from + samples_captured,
                        "total_samples": total_samples,
                        "completed": (start_from + samples_captured) >= total_samples,
                    }
                    progress["last_sign_id"] = sign_id
                    save_partial_data(X_all, y_all)
                    save_progress(progress)
                elif choice == "redo":
                    continue
                elif choice == "skip":
                    samples_captured += 1
                    continue
                elif choice == "quit":
                    raise KeyboardInterrupt

            if samples_captured >= additional:
                progress["signs"][str(sign_id)] = {
                    "name": sign_name,
                    "samples": start_from + samples_captured,
                    "total_samples": total_samples,
                    "completed": True,
                }
                save_partial_data(X_all, y_all)
                save_progress(progress)
                print(f"   ✅ SIGN COMPLETE: {sign_name} ({start_from + samples_captured} samples)")

    except KeyboardInterrupt:
        print("\n⚠️  Saving and quitting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        save_progress(progress)

        if X_all:
            X = np.array(X_all, dtype=np.float32)
            y = np.array(y_all, dtype=np.int32)
            np.save(f"{OUTPUT_DIR}/X.npy", X)
            np.save(f"{OUTPUT_DIR}/y.npy", y)

            labels_captured = []
            for sid_str, v in progress["signs"].items():
                if v.get("samples", 0) > 0:
                    labels_captured.append({"id": int(sid_str), "name": v["name"]})

            with open(f"{OUTPUT_DIR}/labels.json", "w") as f:
                json.dump(labels_captured, f, indent=2)

            completed_count = sum(1 for v in progress["signs"].values() if v.get("completed"))
            with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
                json.dump({
                    "member": MEMBER_NAME,
                    "date": datetime.now().isoformat(),
                    "signs_completed": completed_count,
                    "signs_in_progress": len(labels_captured) - completed_count,
                    "total_samples": len(X),
                    "feature_shape": list(X.shape),
                }, f, indent=2)

            print(f"\n✅ Saved {len(X)} samples")
            print(f"✅ Signs tracked: {len(labels_captured)}")

        # ⭐ FORCE CLOSE WINDOWS ON LINUX
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)   # <-- Forces Linux to close the window
        cv2.waitKey(1)
        print("✅ Camera released.")
        sys.exit(0)      # <-- Kills the Python process cleanly


if __name__ == "__main__":
    main()