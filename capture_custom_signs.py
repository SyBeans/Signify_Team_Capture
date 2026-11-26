"""
capture_custom_signs.py (Auto-Resume + Phrase-Friendly + FAST FPS Edition)
- Sample counts: 15, 20, 40 (60 removed)
- Time-based recording (4.0s wall clock) — FPS-independent
- MediaPipe downsampled to 320x240 — 4x faster, same landmarks
- Type sign name in OpenCV UI. Spaces auto-convert to underscores.
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time
import argparse

# ============================================
# ARGUMENTS
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Your name (e.g., vincent)")
args = parser.parse_args()

MEMBER_NAME = args.name.lower()

# ⭐ CHANGED: time-based instead of frame-count
RECORD_SECONDS = 4.0      # capture 4 seconds wall time (matches FSL-105)
MODEL_FRAMES = 30         # resample target
MIN_FRAMES_OK = 20        # warn if fewer than this captured (unreliable)
PROCESS_WIDTH = 320       # ⭐ downsampled MediaPipe input (4x faster)
PROCESS_HEIGHT = 240

MAX_HANDS = 2
HAND_FEATURES = 63
MAX_SIGN_NAME_LEN = 30

# ============================================
# MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
    model_complexity=1    # ⭐ keep Full model for accuracy
                          #    (use 0 if still too slow — Lite model 2x faster)
)

# ============================================
# SCAN FOR UNFINISHED SIGNS
# ============================================
def get_unfinished_signs():
    base_dir = f"custom_data/{MEMBER_NAME}"
    unfinished = []
    if not os.path.exists(base_dir):
        return unfinished

    for sign_name in os.listdir(base_dir):
        sign_dir = os.path.join(base_dir, sign_name)
        if not os.path.isdir(sign_dir): continue

        progress_path = os.path.join(sign_dir, "progress.json")
        x_path = os.path.join(sign_dir, "X.npy")

        if os.path.exists(progress_path) and os.path.exists(x_path):
            with open(progress_path) as f:
                prog = json.load(f)
            if not prog.get("completed", False):
                current_count = len(np.load(x_path))
                total = prog.get("total_samples", 20)
                unfinished.append((sign_name, current_count, total))
    return unfinished

# ============================================
# STARTUP MENU
# ============================================
def show_startup_menu(unfinished):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "SIGNIFY CAPTURE MENU", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        y_offset = 130
        if unfinished:
            cv2.putText(frame, "Unfinished Signs:", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, (name, curr, total) in enumerate(unfinished):
                if i >= 4: break
                cv2.rectangle(frame, (60, y_offset), (580, y_offset + 50), (0, 100, 150), -1)
                cv2.putText(frame, f"[{i+1}] {name} ({curr}/{total})", (80, y_offset + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                y_offset += 60
        else:
            cv2.putText(frame, "No unfinished signs found.", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        cv2.rectangle(frame, (60, 380), (580, 440), (0, 180, 0), -1)
        cv2.putText(frame, "[N] Type a NEW sign", (200, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "[ESC] Quit", (60, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            return "quit", None
        elif key == ord('n') or key == ord('N'):
            return "new", None
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(unfinished):
                return "resume", unfinished[idx][0]

# ============================================
# TEXT INPUT UI
# ============================================
def get_sign_name_ui():
    h, w = 480, 640
    current_text = ""
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "ENTER SIGN NAME", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Type with SPACES (e.g. I NEED DOCTOR)", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Spaces auto-convert to underscores", (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.rectangle(frame, (60, 200), (580, 270), (255, 255, 255), 2)
        cv2.putText(frame, current_text + "_", (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if len(current_text) > 0:
            preview = current_text.upper().replace(" ", "_")
            cv2.putText(frame, f"Will save as: {preview}", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.putText(frame, f"{len(current_text)}/{MAX_SIGN_NAME_LEN} chars", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "ENTER = Confirm   BACKSPACE = Delete   ESC = Back", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None
        elif key == 13:
            if len(current_text) > 0:
                return current_text.upper().replace(" ", "_")
        elif key == 8:
            current_text = current_text[:-1]
        elif 32 <= key <= 126:
            if len(current_text) < MAX_SIGN_NAME_LEN:
                current_text += chr(key)

# ============================================
# SAMPLE COUNT MENU (15/20/40)
# ============================================
def choose_sample_count(sign_name):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"SIGN: {sign_name}", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "How many samples?", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 200), (240, 270), (0, 150, 0), -1)
        cv2.putText(frame, "1 = 15 samples", (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 200), (460, 270), (0, 150, 200), -1)
        cv2.putText(frame, "2 = 20 samples", (300, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (60, 310), (240, 380), (150, 100, 0), -1)
        cv2.putText(frame, "3 = 40 samples", (80, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (280, 310), (460, 380), (100, 100, 100), -1)
        cv2.putText(frame, "ESC = Back", (320, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'): return 15
        elif key == ord('2'): return 20
        elif key == ord('3'): return 40
        elif key == 27: return None

# ============================================
# RESUME MENU
# ============================================
def show_resume_menu(sign_name, existing_count, stored_total):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"SIGN: {sign_name}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, f"Already: {existing_count}/{stored_total} samples", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Resume what?", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.rectangle(frame, (40, 200), (300, 270), (0, 180, 0), -1)
        cv2.putText(frame, "C = Continue", (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (340, 200), (600, 270), (200, 150, 0), -1)
        cv2.putText(frame, "R = Redo all", (380, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (40, 290), (300, 360), (100, 100, 100), -1)
        cv2.putText(frame, "S = Skip sign", (80, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('c'), ord('C')]: return "continue"
        elif key in [ord('r'), ord('R')]: return "redo"
        elif key in [ord('s'), ord('S')]: return "skip"

# ============================================
# PER-SAMPLE MENU
# ============================================
def show_sample_menu(sign_name, sample_num, total_samples):
    h, w = 480, 640
    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        display_name = sign_name if len(sign_name) <= 28 else sign_name[:25] + "..."
        cv2.putText(frame, f"SIGN: {display_name}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{total_samples}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Accept this sample?", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.rectangle(frame, (40, 190), (300, 260), (0, 180, 0), -1)
        cv2.putText(frame, "SPACE = Accept", (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (340, 190), (600, 260), (200, 150, 0), -1)
        cv2.putText(frame, "R = Redo", (390, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (40, 290), (300, 360), (200, 0, 0), -1)
        cv2.putText(frame, "N = Skip", (100, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.rectangle(frame, (340, 290), (600, 360), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Save & Quit", (360, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord(' '): return "accept"
        elif key in [ord('r'), ord('R')]: return "redo"
        elif key in [ord('n'), ord('N')]: return "skip"
        elif key in [ord('q'), ord('Q')]: return "quit"

# ============================================
# ⭐ HELPER: extract landmarks from MediaPipe result at display scale
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
# CAPTURE ONE SAMPLE — ⭐ TIME-BASED + FAST
# ============================================
def capture_one_sample(cap, sign_name, sample_num, total_samples):
    fps = 0.0
    fps_frames = 0
    fps_last_time = time.time()

    # --- PREVIEW LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret: return None
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ⭐ Downsample for MediaPipe (4x faster, landmarks identical)
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        preview = hands.process(rgb)

        valid_hands = []
        if preview.multi_hand_landmarks:
            valid_hands.append(preview.multi_hand_landmarks[0])
            if len(preview.multi_hand_landmarks) >= 2:
                w1 = preview.multi_hand_landmarks[0].landmark[0]
                w2 = preview.multi_hand_landmarks[1].landmark[0]
                if np.sqrt((w1.x - w2.x)**2 + (w1.y - w2.y)**2) > 0.12:
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
                    if idx in [4, 8, 12, 16, 20]: cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                    else: cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        fps_frames += 1
        now = time.time()
        if now - fps_last_time >= 0.5:
            fps = fps_frames / (now - fps_last_time)
            fps_frames = 0
            fps_last_time = now

        status_color = (0, 255, 0) if valid_hands else (0, 0, 255)
        status_text = f"🟢 HAND ({len(valid_hands)})" if valid_hands else "🔴 SHOW HAND!"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        display_name = sign_name if len(sign_name) <= 35 else sign_name[:32] + "..."
        cv2.putText(frame, f"Sign: {display_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{total_samples}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "SPACE = record | ESC = skip", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Signify Custom Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): break
        if key == 27: return None

    # --- COUNTDOWN (drain buffer) ---
    for i in range(3, 0, -1):
        # Drain stale frames
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        if not ret: return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}...", (260, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.imshow("Signify Custom Capture", frame)
        time.sleep(1)       # 👈 Guarantees a 1-second wait
        cv2.waitKey(1)

    # ⭐⭐⭐ RECORDING LOOP — TIME-BASED ⭐⭐⭐
    captured = []  # list of (timestamp_seconds, landmarks_vec)
    start_time = time.perf_counter()

    fps_frames = 0
    fps_last_time = start_time

    while (time.time() - start_time) < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.01)
            continue

        timestamp = time.perf_counter() - start_time  # 0.0 → 4.0
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ⭐ Downsample for MediaPipe
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        valid_hands = []
        if results.multi_hand_landmarks:
            valid_hands.append(results.multi_hand_landmarks[0])
            if len(results.multi_hand_landmarks) >= 2:
                w1 = results.multi_hand_landmarks[0].landmark[0]
                w2 = results.multi_hand_landmarks[1].landmark[0]
                if np.sqrt((w1.x - w2.x)**2 + (w1.y - w2.y)**2) > 0.12:
                    valid_hands.append(results.multi_hand_landmarks[1])

        if valid_hands:
            # Draw landmarks
            for hl in valid_hands:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
                )
            # Store with timestamp
            captured.append((timestamp, extract_landmarks_vec(valid_hands)))

        # FPS display
        fps_frames += 1
        now = time.time()
        if now - fps_last_time >= 0.5:
            fps = fps_frames / (now - fps_last_time)
            fps_frames = 0
            fps_last_time = now

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        elapsed = time.time() - start_time
        progress = min(elapsed / RECORD_SECONDS, 1.0)

        cv2.putText(frame, f"RECORDING {elapsed:.1f}/{RECORD_SECONDS:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Sign: {display_name}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        status_color = (0, 255, 0) if valid_hands else (0, 0, 255)
        status_text = f"🟢 HAND ({len(valid_hands)})" if valid_hands else "🔴 NO HAND!"
        cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"Frames: {len(captured)}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        bar_w = int(progress * (w - 40))
        cv2.rectangle(frame, (20, 440), (20 + bar_w, 470), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 440), (w - 20, 470), (255, 255, 255), 2)

        cv2.imshow("Signify Custom Capture", frame)
        cv2.waitKey(1)

    # ⭐ CHECK: enough frames captured?
    if len(captured) < MIN_FRAMES_OK:
        print(f"⚠️  Only {len(captured)} frames captured in {RECORD_SECONDS}s — too slow, please redo")
        # Show error overlay for 2 seconds
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 200), (w, 280), (0, 0, 150), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.putText(frame, f"TOO SLOW: {len(captured)} frames", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(frame, "Please redo this sample", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Signify Custom Capture", frame)
            cv2.waitKey(1500)
        return None  # signal to caller that sample failed

    # ⭐ TIME-BASED RESAMPLE to 30 frames
    # Sort by timestamp (should already be sorted, but be safe)
    captured.sort(key=lambda x: x[0])

    # Target: 30 frames uniformly across [0, RECORD_SECONDS]
    target_times = np.linspace(0.0, RECORD_SECONDS, MODEL_FRAMES)

    selected = []
    for t in target_times:
        # Nearest timestamp match
        nearest_idx = min(range(len(captured)), key=lambda i: abs(captured[i][0] - t))
        selected.append(captured[nearest_idx][1])

    positions = np.array(selected, dtype=np.float32)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    return np.concatenate([positions, velocities], axis=1)


# ============================================
# MAIN LOOP
# ============================================
def main():
    print("="*60)
    print(f"🎬 SIGNIFY CUSTOM CAPTURE — {MEMBER_NAME.upper()}")
    print("="*60)
    print(f"Recording: {RECORD_SECONDS:.1f}s wall time (FPS-independent)")
    print(f"Processing: {PROCESS_WIDTH}x{PROCESS_HEIGHT} for MediaPipe (4x faster)")
    print(f"Sample counts: 15, 20, or 40")
    print("="*60)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # ⭐ request 30 FPS from camera

    # Check what FPS the camera actually gave us
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera reports {actual_fps:.1f} FPS")

    try:
        while True:
            unfinished = get_unfinished_signs()
            action, selected_sign = show_startup_menu(unfinished)

            if action == "quit":
                break
            elif action == "new":
                sign_name = get_sign_name_ui()
                if sign_name is None:
                    continue
            elif action == "resume":
                sign_name = selected_sign

            output_dir = f"custom_data/{MEMBER_NAME}/{sign_name}"
            os.makedirs(output_dir, exist_ok=True)
            X_path = f"{output_dir}/X.npy"
            progress_path = f"{output_dir}/progress.json"

            X_all = list(np.load(X_path)) if os.path.exists(X_path) else []
            progress = {"completed": False, "total_samples": 20}
            if os.path.exists(progress_path):
                with open(progress_path) as f: progress = json.load(f)

            total_samples = progress["total_samples"]
            start_from = 0
            skip_sign = False

            if len(X_all) > 0 and not progress.get("completed", False):
                choice = show_resume_menu(sign_name, len(X_all), total_samples)
                if choice == "continue":
                    start_from = len(X_all)
                elif choice == "redo":
                    X_all = []
                    start_from = 0
                    total_samples = choose_sample_count(sign_name)
                    if total_samples is None: continue
                elif choice == "skip":
                    skip_sign = True
            else:
                total_samples = choose_sample_count(sign_name)
                if total_samples is None: continue

            if skip_sign:
                continue

            while start_from < total_samples:
                current_num = start_from + 1
                sample = capture_one_sample(cap, sign_name, current_num, total_samples)

                if sample is None:
                    # Either cancelled or too slow — ask user
                    print(f"⚠️  Sample {current_num} failed/cancelled. Retrying...")
                    continue  # retry same sample number

                choice = show_sample_menu(sign_name, current_num, total_samples)

                if choice == "accept":
                    X_all.append(sample)
                    start_from += 1
                    print(f"✅ {sign_name} Sample {current_num} saved ({len(sample)} frames)")
                    np.save(X_path, np.array(X_all, dtype=np.float32))
                    progress["total_samples"] = total_samples
                    progress["completed"] = (start_from >= total_samples)
                    with open(progress_path, "w") as f: json.dump(progress, f)
                elif choice == "redo":
                    continue
                elif choice == "skip":
                    start_from += 1
                    continue
                elif choice == "quit":
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n⚠️ Quitting...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()