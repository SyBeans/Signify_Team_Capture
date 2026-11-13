"""
merge_custom_signs.py (COMPLETE — FSL-105 + Team + REST)
Merges:
  1. FSL-105 base (105 classes)
  2. All team data from team_data/<member>/ (fine-tune data)
  3. All custom signs from custom_data/<member>/<SIGN>/ (e.g. REST)

Auto-assigns IDs starting from 105 to custom signs.
Output: landmarks/merged/ (matches train_model_v5.py)
"""
import numpy as np
import os
import json

FSL_105_PATH = "../Signify_Model/landmarks/hand"
TEAM_DATA_PATH = "team_data"
CUSTOM_DATA_PATH = "custom_data"
OUTPUT_PATH = "../Signify_Model/landmarks/merged"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# 1. LOAD FSL-105 (IDs 0-104)
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
# 2. LOAD TEAM DATA (IDs 0-104)
# ============================================
print("\n" + "=" * 60)
print("📥 Loading Team Data")
print("=" * 60)

from sklearn.model_selection import train_test_split

members = sorted([
    d for d in os.listdir(TEAM_DATA_PATH)
    if os.path.isdir(f"{TEAM_DATA_PATH}/{d}")
])

if not members:
    print("⚠️  No team data found in team_data/")
else:
    for member in members:
        Xp = f"{TEAM_DATA_PATH}/{member}/X.npy"
        yp = f"{TEAM_DATA_PATH}/{member}/y.npy"

        if not os.path.exists(Xp):
            print(f"⚠️  Skipping {member} (no X.npy)")
            continue

        X_mem = np.load(Xp)
        y_mem = np.load(yp)

        # Drop singleton classes (prevents stratify crash)
        u, c = np.unique(y_mem, return_counts=True)
        singles = u[c < 2]
        if len(singles) > 0:
            mask = ~np.isin(y_mem, singles)
            X_mem = X_mem[mask]
            y_mem = y_mem[mask]
            print(f"⚠️  {member}: Dropped {len(singles)} singleton class(es)")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_mem, y_mem, test_size=0.1, random_state=42, stratify=y_mem
        )
        all_X.append(X_tr)
        all_y.append(y_tr)
        all_X_test.append(X_te)
        all_y_test.append(y_te)
        print(f"✅ {member}: {len(X_mem)} samples")

# ============================================
# 3. DISCOVER CUSTOM SIGNS (e.g. REST)
# ============================================
print("\n" + "=" * 60)
print("📥 Loading Custom Signs")
print("=" * 60)

unique_signs = set()
if os.path.exists(CUSTOM_DATA_PATH):
    for member in os.listdir(CUSTOM_DATA_PATH):
        member_dir = os.path.join(CUSTOM_DATA_PATH, member)
        if os.path.isdir(member_dir):
            for sign_name in os.listdir(member_dir):
                if os.path.isdir(os.path.join(member_dir, sign_name)):
                    unique_signs.add(sign_name)

if not unique_signs:
    print("⚠️  No custom signs found — training will be 105-class only")
    sign_to_id = {}
else:
    # Sort alphabetically so all members get the same ID mapping
    unique_signs = sorted(list(unique_signs))
    sign_to_id = {}
    next_id = 105
    for sign in unique_signs:
        sign_to_id[sign] = next_id
        print(f"  ✅ Assigned {sign} → ID {next_id}")
        next_id += 1

# Load custom data by folder name
if unique_signs and os.path.exists(CUSTOM_DATA_PATH):
    for member in os.listdir(CUSTOM_DATA_PATH):
        member_dir = os.path.join(CUSTOM_DATA_PATH, member)
        if not os.path.isdir(member_dir):
            continue

        for sign_name in os.listdir(member_dir):
            sign_dir = os.path.join(member_dir, sign_name)
            if not os.path.isdir(sign_dir):
                continue

            x_file = os.path.join(sign_dir, "X.npy")
            if not os.path.exists(x_file):
                continue

            X_custom = np.load(x_file)
            y_custom = np.full(len(X_custom), sign_to_id[sign_name], dtype=np.int32)

            # 90/10 split
            split_idx = int(len(X_custom) * 0.9)
            all_X.append(X_custom[:split_idx])
            all_y.append(y_custom[:split_idx])
            all_X_test.append(X_custom[split_idx:])
            all_y_test.append(y_custom[split_idx:])

            print(f"✅ {member}/{sign_name}: {len(X_custom)} samples (ID {sign_to_id[sign_name]})")

# ============================================
# 4. COMBINE
# ============================================
print("\n" + "=" * 60)
print("🔗 Merging")
print("=" * 60)

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
X_final_test = np.concatenate(all_X_test, axis=0)
y_final_test = np.concatenate(all_y_test, axis=0)

print(f"✅ Merged Train: {X_final.shape}")
print(f"✅ Merged Test:  {X_final_test.shape}")
print(f"✅ Total Classes: {len(np.unique(y_final))}")

# Verify all classes present
expected_classes = 105 + len(unique_signs)
present = set(np.unique(y_final).tolist())
missing = [i for i in range(expected_classes) if i not in present]
if missing:
    print(f"⚠️  Missing classes: {missing[:20]}")
else:
    print(f"✅ All {expected_classes} classes present")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_final)
np.save(f"{OUTPUT_PATH}/y_train.npy", y_final)
np.save(f"{OUTPUT_PATH}/X_test.npy", X_final_test)
np.save(f"{OUTPUT_PATH}/y_test.npy", y_final_test)

# ============================================
# 5. GENERATE labels_v5.json FOR THE APP
# ============================================
print("\n" + "=" * 60)
print("🏷️  Generating labels for Flutter app")
print("=" * 60)

# Try to load FSL-105 labels (multiple fallback locations)
master_labels = None

# Try 1: labels.json next to landmarks
if os.path.exists(f"{FSL_105_PATH}/labels.json"):
    with open(f"{FSL_105_PATH}/labels.json") as f:
        master_labels = json.load(f)
    print(f"✅ Loaded labels from {FSL_105_PATH}/labels.json")

# Try 2: labels.csv from FSL dataset
elif os.path.exists("../Signify_Model/datasets/FSL/labels.csv"):
    import pandas as pd
    df = pd.read_csv("../Signify_Model/datasets/FSL/labels.csv")
    master_labels = df['label'].tolist()
    print(f"✅ Loaded labels from datasets/FSL/labels.csv")

# Try 3: relative path
elif os.path.exists("datasets/FSL/labels.csv"):
    import pandas as pd
    df = pd.read_csv("datasets/FSL/labels.csv")
    master_labels = df['label'].tolist()
    print(f"✅ Loaded labels from datasets/FSL/labels.csv (relative)")

if master_labels is None:
    print("❌ Could not find FSL-105 labels! Skipping master_labels.json")
else:
    if len(master_labels) != 105:
        print(f"⚠️  Expected 105 FSL labels, got {len(master_labels)}")

    # Append custom signs (REST, etc.)
    for sign in unique_signs:
        master_labels.append(sign)

    with open(f"{OUTPUT_PATH}/master_labels.json", "w") as f:
        json.dump(master_labels, f, indent=2)

    print(f"✅ Saved master_labels.json ({len(master_labels)} labels)")
    print(f"   Last 5: {master_labels[-5:]}")

print("\n" + "=" * 60)
print("✅ DONE! Now run: python train/train_model_v5.py")
print("=" * 60)