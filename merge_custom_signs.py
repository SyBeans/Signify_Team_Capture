"""
merge_custom_signs.py
Automatically discovers all custom signs across all members,
assigns them IDs (105, 106, 107...), merges the data, and generates labels.json.
"""
import numpy as np
import os
import json

FSL_105_PATH = "../Signify_Model/landmarks/hand" # Update this if needed
CUSTOM_DATA_PATH = "custom_data"
OUTPUT_PATH = "../Signify_Model/landmarks/merged_v5"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 1. Discover All Unique Signs
print("🔍 Scanning for custom signs...")
unique_signs = set()
if os.path.exists(CUSTOM_DATA_PATH):
    for member in os.listdir(CUSTOM_DATA_PATH):
        member_dir = os.path.join(CUSTOM_DATA_PATH, member)
        if os.path.isdir(member_dir):
            for sign_name in os.listdir(member_dir):
                if os.path.isdir(os.path.join(member_dir, sign_name)):
                    unique_signs.add(sign_name)

if not unique_signs:
    print("❌ No custom signs found!")
    exit(1)

# 2. Assign IDs (Starting from 105)
# We sort them alphabetically just to ensure everyone gets the exact same ID mapping.
unique_signs = sorted(list(unique_signs))
sign_to_id = {}
next_id = 105

for sign in unique_signs:
    sign_to_id[sign] = next_id
    print(f"  ✅ Assigned {sign} -> ID {next_id}")
    next_id += 1

# 3. Load FSL-105
print("\n📥 Loading FSL-105...")
X_fsl = np.load(f"{FSL_105_PATH}/X_train.npy")
y_fsl = np.load(f"{FSL_105_PATH}/y_train.npy")
X_fsl_test = np.load(f"{FSL_105_PATH}/X_test.npy")
y_fsl_test = np.load(f"{FSL_105_PATH}/y_test.npy")

all_X, all_y = [X_fsl], [y_fsl]
all_X_test, all_y_test = [X_fsl_test], [y_fsl_test]

# 4. Load Custom Data by Name
print("\n📥 Loading Custom Team Data...")
for member in os.listdir(CUSTOM_DATA_PATH):
    member_dir = os.path.join(CUSTOM_DATA_PATH, member)
    if not os.path.isdir(member_dir): continue
    
    for sign_name in os.listdir(member_dir):
        sign_dir = os.path.join(member_dir, sign_name)
        if not os.path.isdir(sign_dir): continue
        
        x_file = os.path.join(sign_dir, "X.npy")
        if os.path.exists(x_file):
            X_custom = np.load(x_file)
            # Create y array based on the dynamically assigned ID
            y_custom = np.full(len(X_custom), sign_to_id[sign_name], dtype=np.int32)
            
            # 90/10 Split
            split_idx = int(len(X_custom) * 0.9)
            all_X.append(X_custom[:split_idx])
            all_y.append(y_custom[:split_idx])
            all_X_test.append(X_custom[split_idx:])
            all_y_test.append(y_custom[split_idx:])
            
            print(f"  ✅ {member}/{sign_name}: {len(X_custom)} samples (ID: {sign_to_id[sign_name]})")

# 5. Combine and Save
X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
X_final_test = np.concatenate(all_X_test, axis=0)
y_final_test = np.concatenate(all_y_test, axis=0)

print(f"\n✅ Merged Train: {X_final.shape}")
print(f"✅ Merged Test: {X_final_test.shape}")
print(f"✅ Total Classes: {len(np.unique(y_final))}")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_final)
np.save(f"{OUTPUT_PATH}/y_train.npy", y_final)
np.save(f"{OUTPUT_PATH}/X_test.npy", X_final_test)
np.save(f"{OUTPUT_PATH}/y_test.npy", y_final_test)

# 6. Generate master_labels.json for the Flutter App
# Read original 105 labels
with open(f"{FSL_105_PATH}/labels.json", "r") as f: # Assuming you have this
    master_labels = json.load(f)

# Append the new custom signs
for sign in unique_signs:
    master_labels.append(sign)

with open(f"{OUTPUT_PATH}/master_labels.json", "w") as f:
    json.dump(master_labels, f, indent=2)

print(f"\n✅ Saved master_labels.json (Total: {len(master_labels)} labels)")
print(f"✅ Saved to {OUTPUT_PATH}/")