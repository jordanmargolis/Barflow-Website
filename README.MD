# BarFlow

Computer vision squat form analysis. Scores torso, hip hinge, and depth from video using MediaPipe pose detection, and gives specific coaching recommendations based on your results.

**Try it live: https://jordanmargolis.github.io/Barflow-Website/**

## How to Use

1. Open the link above.
2. Click the upload box and select a squat video from your device (MP4 or MOV), or click **Try a Sample Video** to see BarFlow work with a pre-loaded demo clip. No video required.
   - Best results come from a **lateral (side-on) view** with your full body in frame, from head to feet.
3. The video will play back with a skeleton overlay and live feedback on your torso position, hip hinge, and squat depth.
4. When the video ends, you'll see a form summary with:
   - An overall score out of 100
   - Individual scores for torso, hip hinge, and depth
   - The deepest knee angle you reached
   - Specific recommendations: exercises and stretches tailored to whichever areas scored lowest
5. Click **Analyze Another Video** to try a different clip.

Everything runs directly in your browser. No video is uploaded to a server, and nothing is stored or shared.

## Methodology

Scoring is grounded in published strength & conditioning research rather than arbitrary cutoffs, including the tibia-to-trunk parallel standard for torso positioning and phase-aware hip displacement patterns. Each rep is scored across three weighted categories (torso position, hip hinge, and squat depth), combined into a single overall score from 0 to 100.

## Tips for Best Results

- Film from directly to the side of the lifter, not at an angle.
- Make sure the camera can see the whole body throughout the movement. Cropped feet or a cropped head can throw off scoring.
- Good, even lighting helps the pose model track more accurately.
- Works best with a single person in frame.

## Local Development

No build step required, but you do need to serve the files through a local server rather than opening `index.html` directly, since the pose detection library is loaded as an ES module (blocked by browsers over a plain `file://` link).

```bash
git clone https://github.com/jordanmargolis/Barflow-Website.git
cd Barflow-Website
python3 -m http.server
```

Then visit `http://localhost:8000` in your browser.