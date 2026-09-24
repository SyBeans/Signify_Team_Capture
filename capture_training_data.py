"""
capture_training_data.py
Webcam sign capture for Signify team members.
Usage:
  python capture_training_data.py --name vincent
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import json
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Your name (e.g., vincent)")
parser.add_argument("--output", default="team_data", help="Output folder")
args = parser.parse_args()

MEMBER_NAME = args.name.lower()
OUTPUT_DIR = f"{args.output}/{MEMBER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("team_assignments.json", "r") as f:
    cfg = json.load(f)

settings = cfg["capture_settings"]
DURATION_SEC = settings["duration_seconds"]
FPS = settings["fps"]
TOTAL_FRAMES = settings["total_frames_captured"]
MODEL_FRAMES = settings["model_frames"]
SAMPLES_PER_SIGN = settings["samples_per_sign"]
MAX_HANDS = settings["max_hands"]

member_data = cfg["phase_2_fsl_split"].get(MEMBER_NAME)
if not member_data:
    print(f"❌ Member '{MEMBER_NAME}' not found!")
    print(f"   Available: {list(cfg['phase_2_fsl_split'].keys())}")
    exit(1)

SIGN_LIST = list(zip(member_data["signs"], member_data["sign_names"]))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_FEATURES = 63

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def capture_sample(cap, sign_id, sign_name, sample_num):
    print(f"\n🎬 {sign_name} — sample {sample_num+1}/{SAMPLES_PER_SIGN}")
    print("   Press SPACE to start | ESC to skip")

    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: {sign_name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(frame, f"Sample: {sample_num+1}/{SAMPLES_PER_SIGN}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE = record | ESC = skip", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Signify Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == 27:
            return None

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Starting in {i}...", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1000)

    frames_with_hands = []
    total_captured = 0

    while total_captured < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

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

        progress = total_captured / TOTAL_FRAMES
        bar_w = int(progress * (frame.shape[1] - 40))
        cv2.rectangle(frame, (20, 440), (20 + bar_w, 470), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 440), (frame.shape[1]-20, 470), (255, 255, 255), 2)
        cv2.putText(frame, f"RECORDING {total_captured}/{TOTAL_FRAMES}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Signify Capture", frame)
        cv2.waitKey(1)

    if len(frames_with_hands) < MODEL_FRAMES:
        while len(frames_with_hands) < MODEL_FRAMES:
            if frames_with_hands:
                frames_with_hands.append(frames_with_hands[-1])
            else:
                frames_with_hands.append([0.0] * (HAND_FEATURES * 2))

    if len(frames_with_hands) >= MODEL_FRAMES:
        indices = np.linspace(0, len(frames_with_hands)-1, MODEL_FRAMES, dtype=int)
        selected = [frames_with_hands[i] for i in indices]
    else:
        selected = frames_with_hands

    positions = np.array(selected, dtype=np.float32)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    combined = np.concatenate([positions, velocities], axis=1)
    return combined


def main():
    print("=" * 60)
    print(f"🎬 SIGNIFY CAPTURE — {MEMBER_NAME.upper()}")
    print(f"   Signs: {len(SIGN_LIST)}")
    print(f"   Samples/sign: {SAMPLES_PER_SIGN}")
    print(f"   Duration: {DURATION_SEC}s per sample")
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    X_all, y_all = [], []
    labels_captured = []

    try:
        for i, (sign_id, sign_name) in enumerate(SIGN_LIST):
            print(f"\n{'=' * 60}")
            print(f"📝 SIGN {i+1}/{len(SIGN_LIST)}: {sign_name}")
            print(f"{'=' * 60}")

            for sample_num in range(SAMPLES_PER_SIGN):
                sample = capture_sample(cap, sign_id, sign_name, sample_num)
                if sample is not None:
                    X_all.append(sample)
                    y_all.append(sign_id)

            if X_all:
                np.save(f"{OUTPUT_DIR}/X_partial.npy",
                        np.array(X_all, dtype=np.float32))
                np.save(f"{OUTPUT_DIR}/y_partial.npy",
                        np.array(y_all, dtype=np.int32))
                labels_captured.append({"id": sign_id, "name": sign_name})
                with open(f"{OUTPUT_DIR}/labels.json", "w") as f:
                    json.dump(labels_captured, f, indent=2)
                print(f"   💾 Progress saved ({len(X_all)} samples)")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    if X_all:
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int32)
        np.save(f"{OUTPUT_DIR}/X.npy", X)
        np.save(f"{OUTPUT_DIR}/y.npy", y)

        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump({
                "member": MEMBER_NAME,
                "date": datetime.now().isoformat(),
                "signs_captured": len(labels_captured),
                "total_samples": len(X),
                "samples_per_sign": SAMPLES_PER_SIGN,
                "feature_shape": list(X.shape),
            }, f, indent=2)

        print(f"\n✅ Captured {len(X)} samples for {len(labels_captured)} signs")
        print(f"✅ Saved to {OUTPUT_DIR}/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()