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







Capturing REST 

When the team runs the command, the OpenCV UI does exactly what it did for FSL signs:

Type REST in the text box (already auto-capitalized)

Press Enter

See "How many samples?" → press 2 for 40 samples

Camera opens showing "SIGN: REST"

Press SPACE → countdown 3-2-1

Do a random hand gesture for 4 seconds

Accept/Redo/Skip menu → press SPACE

Repeat 40 times with different movements each time

Auto-saves to custom_data/<name>/REST/X.npy






Gesture Form (What People Should Actually Do)
This is the important part. REST is NOT a real gesture — it's a "not-a-sign" catch-all. Here's what your team should capture:

✅ DO capture these (each ~4 seconds):
Form	Example
Hands resting - Palms flat on table, forearms still
Scratching - Scratch your head, ear, or arm
Fixing hair - Run fingers through hair
Adjusting glasses - Push glasses up nose
Stretching - Arms up, yawn stretch
Fidgeting - Shake hands out, flex fingers randomly
Partially off-screen - One hand visible, slowly moving out
Transitional movement - Moving from one position to another without signing
What "pure rest" samples should look like
The 8 pure-rest samples should NOT be identical — vary the pose:

Pose	Hand position
1	Palms flat on table, both visible
2	Hands on lap, slightly visible
3	One hand on table, one hidden
4	Chin resting on one palm
5	Hands crossed on desk
6	Hand holding phone down at side
7	Hand touching ear (still, not scratching)
8	Both hands slightly raised but idle


✅ The safe zone for REST
These are definitively not signs in your 105-class dataset:

Safe REST form	Why safe
Scratch head/ear/neck	Open hand, random motion, no sign structure
Fix hair (running fingers through)	Fingers spread, motion chaotic
Adjust glasses	Fingers at face, tugging motion
Stretch arms overhead	Non-sign posture
Shake hands out	Non-sign motion
Rub eyes	Non-sign
Fidget with hair tie, ring, pen	Non-sign
Cross fingers on lap (hidden)	Hands not visible = motion gate blocks, no ghost possible
Yawn with hand near mouth	Non-sign
Adjust shirt collar, sleeve	Non-sign
🎯 The self-test rule for your team
Before pressing SPACE, each member should ask:

"If someone saw this pose frozen in a photo, would they recognize it as a real FSL sign?"

YES → skip it, don't record

NO → safe to record as REST




 The 40-Sample REST Breakdown
#	Category	Count	Example movements (each must be different)
1	Face fidgets	6	Rub eyes, scratch nose, adjust glasses, touch ear, tap cheek, touch eyebrow
2	Hair movements	4	Run fingers through, fix hair tie, tuck hair behind ear, adjust bangs
3	Hand fidgets (in air)	6	Shake hands out, flex fingers, fist-unfist, wiggle wrist, rotate hand, tap fingers
4	Reaching/transitional	6	Reach for cup, move hand across frame, adjust position, touch phone, reach for pen, pull hand back
5	Double-hand fidgets	5	Rub hands together, crack knuckles, cross fingers (not FSL), clasp hands, rest one hand on other
6	Adjusting clothing	4	Adjust shirt collar, pull sleeve, adjust belt, touch pocket
7	Slow movements	3	Slowly scratch face, slowly rotate wrist, slow open/close hand
8	Pure rest (still but visible)	6	Hands on table, hands on lap, chin on palm, hand touching ear (static), hand on armrest, one hand down
TOTAL	40	
🎯 Your "rubbing eyes" example
Rubbing eyes is 1 of the 6 face fidgets. So a valid face-fidget set would be:

Sample #	Movement
1	Rub eyes
2	Scratch nose
3	Adjust glasses (push up)
4	Touch ear
5	Tap cheek
6	Touch eyebrow
That's 6 face fidgets. Each is a different movement — the model needs to see that "moving hands near the face" = REST regardless of specific motion.



Capture REST 40 times:
  Sample 1: rub eyes
  Sample 2: scratch nose
  Sample 3: fix hair
  Sample 4: adjust glasses
  Sample 5: shake hands out
  Sample 6: stretch arms
  ...
  Sample 40: hand on table

→ Model learns: "Any movement that isn't a real sign = REST"
→ At inference: user scratches head → correctly predicts REST ✅
→ At inference: user fixes hair → correctly predicts REST ✅
→ At inference: user does anything non-sign → REST ✅