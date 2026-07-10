# BarFlow

Computer vision squat form analysis. Scores torso, hip hinge, and depth from video using MediaPipe pose detection.

**Try it live: https://jordanmargolis.github.io/Barflow-Website/**

## How to Use

1. Open the link above.
2. Click the upload box and select a squat video from your device (MP4 or MOV).
   - Best results come from a **lateral (side-on) view** with your full body in frame, from head to feet.
3. The video will play back with a skeleton overlay and live feedback on your torso position, hip hinge, and squat depth.
4. When the video ends, you'll see a form summary with:
   - An overall score out of 100
   - Individual scores for torso, hip hinge, and depth
   - The deepest knee angle you reached
5. Click **Analyze Another Video** to try a different clip.

Everything runs directly in your browser — no video is uploaded to a server, and nothing is stored or shared.

## Tips for Best Results

- Film from directly to the side of the lifter, not at an angle.
- Make sure the camera can see the whole body throughout the movement — cropped feet or a cropped head can throw off scoring.
- Good, even lighting helps the pose model track more accurately.
- Works best with a single person in frame.

## Local Development

No build step required. Clone the repo and open `index.html` in a browser, or serve the folder locally:

```bash
git clone https://github.com/jordanmargolis/BarPath.git
cd BarPath
python3 -m http.server
```

Then visit `http://localhost:8000` in your browser.

## License

[Add your license here]
