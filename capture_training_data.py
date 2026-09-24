"""
capture_training_data.py (FINAL)
Per-sample confirm + sample count menu + resume.

Features:
- After EACH sample: [SPACE] Accept | [R] Redo | [N] Skip | [Q] Quit
- Before EACH sign: choose 20/40/60 samples
- Resume support (asks what to do)
- Progress tracking
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import argparse
from datetime import datetime

# ============================================
# ARGUMENTS
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Your name")
parser.add_argument("--output", default="team_data", help="Output folder")
parser.add_argument("--redo", type=int, default=None, help="Redo specific sign ID")
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
TOTAL_FRAMES = settings["total_frames_captured"]
MODEL_FRAMES = settings["model_frames"]
MAX_HANDS = settings["max_hands"]

member_data = cfg["phase_2_fsl_split"].get(MEMBER_NAME)
if not member_data:
    print(f"❌ Member '{MEMBER_NAME}' not found!")
    print(f"   Available: {list(cfg['phase_2_fsl_split'].keys())}")
    exit(1)

SIGN_LIST = list(zip(member_data["signs"], member_data["sign_names"]))

# ============================================
# PROGRESS FILE
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
    """Load all previous samples."""
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X = list(np.load(X_PATH))
        y = list(np.load(Y_PATH))
        return X, y
    return [], []

def save_partial_data(X_all, y_all, labels_captured):
    """Save all accumulated data."""
    np.save(X_PATH, np.array(X_all, dtype=np.float32))
    np.save(Y_PATH, np.array(y_all, dtype=np.int32))
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_captured, f, indent=2)

def clear_all_data():
    """Wipe all previous data (for --fresh)."""
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
    min_tracking_confidence=0.5
)


# ============================================
# SAMPLE COUNT MENU (20/40/60)
# ============================================
def choose_sample_count(sign_name):
    """Show menu to pick 20/40/60 samples."""
    h, w = 480, 640

    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.putText(frame, f"SIGN: {sign_name}",
                    (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(frame, "How many samples?",
                    (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Option 1: 20
        cv2.rectangle(frame, (60, 200), (240, 270), (0, 150, 0), -1)
        cv2.putText(frame, "1 = 20 samples",
                    (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Option 2: 40
        cv2.rectangle(frame, (280, 200), (460, 270), (0, 150, 200), -1)
        cv2.putText(frame, "2 = 40 samples",
                    (300, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Option 3: 60
        cv2.rectangle(frame, (60, 310), (240, 380), (150, 100, 0), -1)
        cv2.putText(frame, "3 = 60 samples",
                    (80, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Quit
        cv2.rectangle(frame, (280, 310), (460, 380), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Quit",
                    (320, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            return 20
        elif key == ord('2'):
            return 40
        elif key == ord('3'):
            return 60
        elif key == ord('q') or key == ord('Q'):
            return None


# ============================================
# PER-SAMPLE MENU
# ============================================
def show_sample_menu(sign_name, sample_num, total_samples):
    """
    Menu after each sample captured.
    Returns: "accept", "redo", "skip", "quit"
    """
    h, w = 480, 640

    while True:
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.putText(frame, f"SIGN: {sign_name}",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{total_samples}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Accept this sample?",
                    (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Accept
        cv2.rectangle(frame, (40, 190), (300, 260), (0, 180, 0), -1)
        cv2.putText(frame, "SPACE = Accept",
                    (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Redo
        cv2.rectangle(frame, (340, 190), (600, 260), (200, 150, 0), -1)
        cv2.putText(frame, "R = Redo",
                    (390, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Skip
        cv2.rectangle(frame, (40, 290), (300, 360), (200, 0, 0), -1)
        cv2.putText(frame, "N = Skip",
                    (100, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Quit
        cv2.rectangle(frame, (340, 290), (600, 360), (100, 100, 100), -1)
        cv2.putText(frame, "Q = Save & Quit",
                    (360, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord(' '):
            return "accept"
        elif key == ord('r') or key == ord('R'):
            return "redo"
        elif key == ord('n') or key == ord('N'):
            return "skip"
        elif key == ord('q') or key == ord('Q'):
            return "quit"


# ============================================
# CAPTURE ONE SAMPLE (with live hand preview)
# ============================================
def capture_one_sample(cap, sign_name, sample_num, total_samples):
    """Capture a single 4-second sample."""
    # Wait for SPACE
    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview = hands.process(frame_rgb)

        if preview.multi_hand_landmarks:
            for hl in preview.multi_hand_landmarks:
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

        if preview.multi_hand_landmarks:
            status_color = (0, 255, 0)
            status_text = f"🟢 HAND ({len(preview.multi_hand_landmarks)})"
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
        cv2.putText(frame, "SPACE = record | ESC = skip", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == 27:
            return None

    # Countdown
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}...", (260, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1000)

    # Record
    frames_with_hands = []
    total_captured = 0

    while total_captured < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_detected_now = False

        if results.multi_hand_landmarks:
            hand_detected_now = True
            for hl in results.multi_hand_landmarks:
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

            hand1 = []
            for lm in results.multi_hand_landmarks[0].landmark:
                hand1.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) >= 2:
                hand2 = []
                for lm in results.multi_hand_landmarks[1].landmark:
                    hand2.extend([lm.x, lm.y, lm.z])
            else:
                hand2 = [0.0] * HAND_FEATURES
            frames_with_hands.append(hand1 + hand2)

        total_captured += 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"RECORDING {total_captured}/{TOTAL_FRAMES}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Sign: {sign_name}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if hand_detected_now:
            status_color = (0, 255, 0)
            status_text = f"🟢 HAND ({len(results.multi_hand_landmarks)})"
        else:
            status_color = (0, 0, 255)
            status_text = "🔴 NO HAND!"

        cv2.putText(frame, status_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"Hands: {len(frames_with_hands)}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        progress = total_captured / TOTAL_FRAMES
        bar_w = int(progress * (w - 40))
        cv2.rectangle(frame, (20, 440), (20 + bar_w, 470), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 440), (w - 20, 470), (255, 255, 255), 2)

        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1)

    # Pad + downsample
    if len(frames_with_hands) < MODEL_FRAMES:
        while len(frames_with_hands) < MODEL_FRAMES:
            if frames_with_hands:
                frames_with_hands.append(frames_with_hands[-1])
            else:
                frames_with_hands.append([0.0] * (HAND_FEATURES * 2))

    if len(frames_with_hands) >= MODEL_FRAMES:
        indices = np.linspace(0, len(frames_with_hands) - 1, MODEL_FRAMES, dtype=int)
        selected = [frames_with_hands[i] for i in indices]
    else:
        selected = frames_with_hands

    positions = np.array(selected, dtype=np.float32)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    return np.concatenate([positions, velocities], axis=1)


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 60)
    print(f"🎬 SIGNIFY CAPTURE — {MEMBER_NAME.upper()}")
    print(f"   Total signs: {len(SIGN_LIST)}")
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 60)

    # Handle --fresh
    if args.fresh:
        print("🆕 --fresh: deleting all previous data...")
        clear_all_data()

    # Load existing data
    X_all, y_all = load_existing_data()
    progress = load_progress()

    if X_all:
        print(f"📂 Found existing data: {len(X_all)} samples")
        print(f"📋 Signs tracked: {len(progress['signs'])}")

    # Build work queue
    if args.redo is not None:
        pending_signs = [(sid, name) for sid, name in SIGN_LIST if sid == args.redo]
        print(f"🔁 Redo mode: only sign {args.redo}")
    else:
        pending_signs = []
        for sid, name in SIGN_LIST:
            entry = progress["signs"].get(str(sid), {})
            if not entry.get("completed"):
                pending_signs.append((sid, name))
        print(f"📋 Signs to capture: {len(pending_signs)}/{len(SIGN_LIST)}")

    if not pending_signs:
        print("✅ All signs complete!")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    try:
        for idx, (sign_id, sign_name) in enumerate(pending_signs):
            print(f"\n{'=' * 60}")
            print(f"📝 SIGN {idx+1}/{len(pending_signs)}: {sign_name}")
            print(f"{'=' * 60}")

            # Check for resume
            entry = progress["signs"].get(str(sign_id), {})
            in_progress = entry.get("samples", 0) > 0 and not entry.get("completed")

            skip_this_sign = False

            if in_progress:
                existing_count = entry["samples"]
                print(f"⚠️  This sign has {existing_count} samples in progress.")
                print("   [C] Continue from where left off")
                print("   [R] Redo all")
                print("   [S] Skip this sign")

                # Show menu on a screen too
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, f"SIGN: {sign_name}",
                            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Already: {existing_count} samples",
                            (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Resume what?",
                            (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                cv2.rectangle(frame, (40, 200), (300, 270), (0, 180, 0), -1)
                cv2.putText(frame, "C = Continue",
                            (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.rectangle(frame, (340, 200), (600, 270), (200, 150, 0), -1)
                cv2.putText(frame, "R = Redo all",
                            (380, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.rectangle(frame, (40, 290), (300, 360), (100, 100, 100), -1)
                cv2.putText(frame, "S = Skip sign",
                            (80, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.imshow("Signify Capture", frame)
                key = cv2.waitKey(0) & 0xFF

                if key == ord('c') or key == ord('C'):
                    # Ask sample count again
                    total_samples = choose_sample_count(sign_name + " (resume)")
                    if total_samples is None:
                        raise KeyboardInterrupt
                    # Delete existing samples for this sign
                    filtered_X = [x for x, y in zip(X_all, y_all) if y != sign_id]
                    filtered_y = [y for y in y_all if y != sign_id]
                    X_all = filtered_X
                    y_all = filtered_y
                    start_from = existing_count
                    keep_existing = False

                elif key == ord('r') or key == ord('R'):
                    total_samples = choose_sample_count(sign_name + " (redo)")
                    if total_samples is None:
                        raise KeyboardInterrupt
                    # Delete existing samples
                    filtered_X = [x for x, y in zip(X_all, y_all) if y != sign_id]
                    filtered_y = [y for y in y_all if y != sign_id]
                    X_all = filtered_X
                    y_all = filtered_y
                    start_from = 0
                    keep_existing = False

                elif key == ord('s') or key == ord('S'):
                    skip_this_sign = True
                    start_from = 0
                    keep_existing = False
                    total_samples = 0
                else:
                    total_samples = 0
                    start_from = 0
                    keep_existing = False

            else:
                # Fresh sign → ask sample count
                total_samples = choose_sample_count(sign_name)
                if total_samples is None:
                    raise KeyboardInterrupt
                start_from = 0
                keep_existing = False

            if skip_this_sign:
                print(f"   ⏭️  Skipped")
                continue

            # Capture loop for this sign
            samples_captured = 0

            while samples_captured < total_samples:
                current_num = start_from + samples_captured + 1

                sample = capture_one_sample(cap, sign_name, current_num, total_samples)

                if sample is None:
                    print("   ⚠️  Capture cancelled")
                    break

                # Per-sample menu
                choice = show_sample_menu(sign_name, current_num, total_samples)

                if choice == "accept":
                    X_all.append(sample)
                    y_all.append(sign_id)
                    samples_captured += 1
                    print(f"   ✅ Sample {current_num}/{total_samples} saved")

                    # Save progress every 5 samples
                    if samples_captured % 5 == 0:
                        save_partial_data(X_all, y_all, [])
                        progress["signs"][str(sign_id)] = {
                            "name": sign_name,
                            "samples": start_from + samples_captured,
                            "completed": False
                        }
                        progress["last_sign_id"] = sign_id
                        save_progress(progress)

                elif choice == "redo":
                    print(f"   🔁 Redo sample {current_num}")
                    continue

                elif choice == "skip":
                    print(f"   ⏭️  Skipped sample {current_num}")
                    samples_captured += 1
                    continue

                elif choice == "quit":
                    raise KeyboardInterrupt

            # Sign complete
            if samples_captured >= total_samples:
                progress["signs"][str(sign_id)] = {
                    "name": sign_name,
                    "samples": start_from + samples_captured,
                    "completed": True
                }
                save_partial_data(X_all, y_all, [])
                save_progress(progress)
                print(f"   ✅ SIGN COMPLETE: {sign_name} ({total_samples} samples)")

    except KeyboardInterrupt:
        print("\n⚠️  Saving and quitting...")

    # Final save
    if X_all:
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int32)
        np.save(f"{OUTPUT_DIR}/X.npy", X)
        np.save(f"{OUTPUT_DIR}/y.npy", y)

        labels_captured = [
            {"id": int(k), "name": v["name"]}
            for k, v in progress["signs"].items() if v.get("completed")
        ]

        with open(f"{OUTPUT_DIR}/labels.json", "w") as f:
            json.dump(labels_captured, f, indent=2)

        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump({
                "member": MEMBER_NAME,
                "date": datetime.now().isoformat(),
                "signs_completed": len([1 for v in progress["signs"].values() if v.get("completed")]),
                "total_samples": len(X),
                "feature_shape": list(X.shape),
            }, f, indent=2)

        print(f"\n✅ Saved {len(X)} samples")
        print(f"✅ To {OUTPUT_DIR}/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()