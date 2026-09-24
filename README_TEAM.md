# Signify — Team Capture Instructions

## Your Task
Record ~21 FSL-105 signs using your webcam. Each sign 20 times. **~1 hour total.**

## Setup (One Time)

### 1. Clone repo
```bash

git clone <>
cd Signify_Team_Capture



2. Create your branch
bash

git checkout -b <your-name>-capture

Replace <your-name> with: vincent, jonathan, mark, charity, or kent.
3. Create Python environment
bash

python3.12 -m venv venv
source venv/bin/activate          # Linux/Mac
# OR
venv\Scripts\activate              # Windows

4. Install packages
bash

pip install -r requirements_capture.txt

Capture
bash

python capture_training_data.py --name <your-name>

Replace <your-name> with your actual name.
How It Works

    Screen shows sign name (e.g., "GOOD MORNING")

    Read the sign name

    Press SPACE to record

    3-second countdown appears

    Sign naturally for 4 seconds

    Auto-saves and moves to next sample

    Repeat 20 times per sign

    Auto-advances to next sign

    Continue through all your assigned signs (~21)

Tips

    ✅ Good lighting on your hands

    ✅ Plain background (wall behind you)

    ✅ Hand(s) fully visible in frame

    ✅ Sit ~1 meter from camera

    ✅ Sign at normal speed

    ✅ Press ESC to skip a bad sample

    ✅ Sign the SAME way every time (consistency matters!)

Signs You'll Capture

Each member gets ~21 signs. Your specific list:

    Vincent: Signs 0-20 (GOOD MORNING → ONE)

    Jonathan: Signs 21-41 (TWO → DECEMBER)

    Mark: Signs 42-62 (MONDAY → BOY)

    Charity: Signs 63-83 (GIRL → LIGHT)

    Kent: Signs 84-104 (DARK → NO SUGAR)

The script will only show YOUR signs — no need to memorize!
If You Need to Resume

The script auto-saves after each sign. If interrupted:
bash

python capture_training_data.py --name <your-name>

It will continue from where you left off.
When Done
Push your data back:
bash

cd ..
git add team_data/<your-name>/
git commit -m "<your-name>: captured N signs"
git push origin <your-name>-capture

Then message Vincent:

    Done! Branch: <your-name>-capture