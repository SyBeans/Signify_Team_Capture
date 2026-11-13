"""
merge_team_data.py (v2 — handles REST at ID 105)
Combines FSL-105 + Team Data + Custom REST sign.
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split

FSL_105_PATH = "../Signify_Model/landmarks/hand"
TEAM_DATA_PATH = "team_data"
CUSTOM_DATA_PATH = "custom_data"
OUTPUT_PATH = "../Signify_Model/landmarks/merged"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# 1. LOAD FSL-105 (IDs 0-104)
# ============================================
print("=" * 60); print("📥 Loading FSL-105"); print("=" * 60)
X_fsl = np.load(f"{FSL_105_PATH}/X_train.npy")
y_fsl = np.load(f"{FSL_105_PATH}/y_train.npy")
X_fsl_test = np.load(f"{FSL_105_PATH}/X_test.npy")
y_fsl_test = np.load(f"{FSL_105_PATH}/y_test.npy")

print(f"✅ FSL-105 train: {X_fsl.shape}")
print(f"✅ FSL-105 test:  {X_fsl_test.shape}")

all_X = [X_fsl]; all_y = [y_fsl]
all_X_test = [X_fsl_test]; all_y_test = [y_fsl_test]

# ============================================
# 2. LOAD TEAM DATA (IDs 0-104)
# ============================================
print("\n" + "=" * 60); print("📥 Loading Team Data"); print("=" * 60)

members = sorted([d for d in os.listdir(TEAM_DATA_PATH) if os.path.isdir(f"{TEAM_DATA_PATH}/{d}")])
if not members:
    print("⚠️  No team data found in team_data/")
    exit(1)

for member in members:
    Xp = f"{TEAM_DATA_PATH}/{member}/X.npy"
    yp = f"{TEAM_DATA_PATH}/{member}/y.npy"
    if not os.path.exists(Xp):
        print(f"⚠️  Skipping {member} (no X.npy)")
        continue

    X_mem = np.load(Xp)
    y_mem = np.load(yp)

    # Drop singletons (prevents stratify crash)
    u, c = np.unique(y_mem, return_counts=True)
    singles = u[c < 2]
    if len(singles) > 0:
        mask = ~np.isin(y_mem, singles)
        X_mem, y_mem = X_mem[mask], y_mem[mask]
        print(f"⚠️  {member}: Dropped {len(singles)} singleton classes: {singles.tolist()}")

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_mem, y_mem, test_size=0.1, random_state=42, stratify=y_mem
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_mem, y_mem, test_size=0.1, random_state=42
        )

    all_X.append(X_tr); all_y.append(y_tr)
    all_X_test.append(X_te); all_y_test.append(y_te)
    print(f"✅ {member}: {len(X_mem)} samples")

# ============================================
# 3. LOAD REST (ID 105) FROM custom_data/
# ============================================
print("\n" + "=" * 60); print("📥 Loading REST (ID 105)"); print("=" * 60)

rest_found = False
if os.path.exists(CUSTOM_DATA_PATH):
    for member in os.listdir(CUSTOM_DATA_PATH):
        rest_path = f"{CUSTOM_DATA_PATH}/{member}/REST/X.npy"
        if os.path.exists(rest_path):
            X_rest = np.load(rest_path)
            y_rest = np.full(len(X_rest), 105, dtype=np.int32)  # 🚨 Force ID 105
            
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_rest, y_rest, test_size=0.1, random_state=42
            )
            all_X.append(X_tr); all_y.append(y_tr)
            all_X_test.append(X_te); all_y_test.append(y_te)
            print(f"✅ {member}/REST: {len(X_rest)} samples (ID 105)")
            rest_found = True

if not rest_found:
    print("⚠️  NO REST DATA FOUND — training will be 105-class only!")
    print("⚠️  Capture REST first: python capture_custom_signs.py --name <member> --sign REST")

# ============================================
# 4. COMBINE & SAVE
# ============================================
print("\n" + "=" * 60); print("🔗 Merging"); print("=" * 60)

X_final = np.concatenate(all_X, axis=0); y_final = np.concatenate(all_y, axis=0)
X_final_test = np.concatenate(all_X_test, axis=0); y_final_test = np.concatenate(all_y_test, axis=0)

print(f"✅ Train: {X_final.shape}")
print(f"✅ Test:  {X_final_test.shape}")
print(f"✅ Total Classes: {len(np.unique(y_final))}")

# Verify 106 classes present
present = set(np.unique(y_final).tolist())
missing = [i for i in range(106) if i not in present]
if missing:
    print(f"⚠️  Missing classes: {missing[:20]}{'...' if len(missing) > 20 else ''}")
else:
    print(f"✅ All 106 classes present (0-104 FSL + 105 REST)")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_final)
np.save(f"{OUTPUT_PATH}/y_train.npy", y_final)
np.save(f"{OUTPUT_PATH}/X_test.npy", X_final_test)
np.save(f"{OUTPUT_PATH}/y_test.npy", y_final_test)

print(f"\n✅ Saved to {OUTPUT_PATH}/")
print(f"✅ DONE! Now run: python train/train_model_v5.py")