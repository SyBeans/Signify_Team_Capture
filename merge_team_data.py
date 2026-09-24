"""
merge_team_data.py
Combines FSL-105 extraction + all team data.
Run from team_capture/ folder.
"""

import numpy as np
import json
import os

# Paths
FSL_105_PATH = "../Signify_Model/landmarks/hand"
TEAM_DATA_PATH = "team_data"
OUTPUT_PATH = "../Signify_Model/landmarks/merged"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# LOAD FSL-105
# ============================================
print("=" * 60)
print("📥 Loading FSL-105")
print("=" * 60)

X_fsl = np.load(f"{FSL_105_PATH}/X_train.npy")
y_fsl = np.load(f"{FSL_105_PATH}/y_train.npy")
X_fsl_test = np.load(f"{FSL_105_PATH}/X_test.npy")
y_fsl_test = np.load(f"{FSL_105_PATH}/y_test.npy")

print(f"✅ FSL-105 train: {X_fsl.shape}")
print(f"✅ FSL-105 test:  {X_fsl_test.shape}")

all_X = [X_fsl]
all_y = [y_fsl]
all_X_test = [X_fsl_test]
all_y_test = [y_fsl_test]

# ============================================
# LOAD TEAM DATA
# ============================================
print("\n" + "=" * 60)
print("📥 Loading Team Data")
print("=" * 60)

members = sorted([
    d for d in os.listdir(TEAM_DATA_PATH)
    if os.path.isdir(f"{TEAM_DATA_PATH}/{d}")
])

if not members:
    print("⚠️  No team data found in team_data/")
    exit(1)

from sklearn.model_selection import train_test_split

for member in members:
    mp = f"{TEAM_DATA_PATH}/{member}"
    Xp, yp = f"{mp}/X.npy", f"{mp}/y.npy"

    if not os.path.exists(Xp):
        print(f"⚠️  Skipping {member} (no X.npy)")
        continue

    X_mem = np.load(Xp)
    y_mem = np.load(yp)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_mem, y_mem, test_size=0.1, random_state=42, stratify=y_mem
    )

    all_X.append(X_tr)
    all_y.append(y_tr)
    all_X_test.append(X_te)
    all_y_test.append(y_te)

    print(f"✅ {member}: {X_mem.shape}")

# ============================================
# COMBINE
# ============================================
print("\n" + "=" * 60)
print("🔗 Merging")
print("=" * 60)

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
X_final_test = np.concatenate(all_X_test, axis=0)
y_final_test = np.concatenate(all_y_test, axis=0)

print(f"✅ Train: {X_final.shape}")
print(f"✅ Test:  {X_final_test.shape}")
print(f"✅ Classes: {len(np.unique(y_final))}")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_final)
np.save(f"{OUTPUT_PATH}/y_train.npy", y_final)
np.save(f"{OUTPUT_PATH}/X_test.npy", X_final_test)
np.save(f"{OUTPUT_PATH}/y_test.npy", y_final_test)

print(f"\n✅ Saved to {OUTPUT_PATH}/")
print(f"✅ DONE! Now run: python train/train_model_v5.py")