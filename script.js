// ==============================
// BarFlow Web Demo
// JS port of barpath_mvp.py — same scoring logic, running client-side
// via MediaPipe's browser Pose model. No backend, nothing uploaded.
// ==============================

// MediaPipe Tasks Vision — loaded straight from CDN as an ES module.
// This is the browser-native replacement for the Python `mediapipe` package.
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// ==============================
// Scoring Thresholds (same values as barpath_mvp.py)
// ==============================

const TORSO_TIBIA_GOOD = 10;
const TORSO_TIBIA_WARNING = 20;
const KNEE_DEPTH_GOOD = 90;
const KNEE_DEPTH_ALMOST = 110;

const TORSO_WEIGHT = 40;
const HIP_WEIGHT = 35;
const DEPTH_WEIGHT = 25;

// 33-point pose landmark indices (same layout as Python mediapipe)
const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28
};

// ==============================
// DOM references
// ==============================

const uploadStage = document.getElementById("uploadStage");
const loadingStage = document.getElementById("loadingStage");
const analysisStage = document.getElementById("analysisStage");
const resultsStage = document.getElementById("resultsStage");
const errorStage = document.getElementById("errorStage");

const videoInput = document.getElementById("videoInput");
const sourceVideo = document.getElementById("sourceVideo");
const canvas = document.getElementById("overlayCanvas");
const ctx = canvas.getContext("2d");

const liveTorso = document.getElementById("liveTorso");
const liveHip = document.getElementById("liveHip");
const liveDepth = document.getElementById("liveDepth");

const overallNumber = document.getElementById("overallNumber");
const torsoNumber = document.getElementById("torsoNumber");
const hipNumber = document.getElementById("hipNumber");
const depthNumber = document.getElementById("depthNumber");
const deepestNote = document.getElementById("deepestNote");

const tryAnotherBtn = document.getElementById("tryAnotherBtn");
const errorRetryBtn = document.getElementById("errorRetryBtn");
const errorText = document.getElementById("errorText");

let poseLandmarker = null;
let drawingUtils = null;

// Per-video running state
let torsoScores = [];
let hipScores = [];
let minKneeAngle = 180;
let hipStartX = null;

// ==============================
// Stage helpers
// ==============================

function showStage(stage) {
  [uploadStage, loadingStage, analysisStage, resultsStage, errorStage].forEach(
    (s) => s.classList.add("hidden")
  );
  stage.classList.remove("hidden");
}

function showError(message) {
  errorText.textContent = message;
  showStage(errorStage);
}

// ==============================
// Math helpers (direct port of calculate_angle from Python)
// ==============================

function calculateAngle(a, b, c) {
  const radians =
    Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0]);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

function calculateTrunkAngle(shoulder, hip) {
  const virtualVertical = [hip[0], hip[1] - 0.1];
  return calculateAngle(shoulder, hip, virtualVertical);
}

function calculateTibiaAngle(knee, ankle) {
  const virtualVertical = [ankle[0], ankle[1] - 0.1];
  return calculateAngle(knee, ankle, virtualVertical);
}

function detectVisibleSide(landmarks) {
  const leftVis = landmarks[LM.LEFT_KNEE].visibility ?? 0;
  const rightVis = landmarks[LM.RIGHT_KNEE].visibility ?? 0;
  return leftVis >= rightVis ? "left" : "right";
}

// ==============================
// Scoring functions (direct port of Python scoring logic)
// ==============================

function scoreTorso(shoulder, hip, knee, ankle) {
  const kneeAngle = calculateAngle(hip, knee, ankle);

  if (kneeAngle >= 160) {
    return { score: null, label: "Torso: Waiting", status: "dim" };
  }

  const trunkAngle = calculateTrunkAngle(shoulder, hip);
  const tibiaAngle = calculateTibiaAngle(knee, ankle);
  const difference = Math.abs(trunkAngle - tibiaAngle);

  if (difference <= TORSO_TIBIA_GOOD) {
    return { score: 100, label: "Torso: Good", status: "good" };
  } else if (difference <= TORSO_TIBIA_WARNING) {
    return { score: 60, label: "Torso: Leaning", status: "warn" };
  } else {
    return { score: 20, label: "Torso: Excessive Lean", status: "bad" };
  }
}

function scoreHipHinge(hipX, ankleX, kneeAngle, hipStartXVal) {
  if (kneeAngle >= 160 || hipStartXVal === null) {
    return { score: null, label: "Hip Hinge: Waiting", status: "dim" };
  }

  const displacement = Math.abs(hipX - hipStartXVal);
  const hipBehindAnkle = hipX - ankleX > 0.03;

  if (displacement >= 0.06 && hipBehindAnkle) {
    return { score: 100, label: "Hip Hinge: Good", status: "good" };
  } else if (displacement >= 0.03 || hipBehindAnkle) {
    return { score: 60, label: "Hip Hinge: Warning", status: "warn" };
  } else {
    return { score: 20, label: "Hip Hinge: Poor", status: "bad" };
  }
}

function scoreDepth(kneeAngle) {
  if (kneeAngle <= KNEE_DEPTH_GOOD) {
    return { score: 100, label: "Depth: Good", status: "good" };
  } else if (kneeAngle <= KNEE_DEPTH_ALMOST) {
    return { score: 60, label: "Depth: Almost", status: "warn" };
  } else {
    return { score: 20, label: "Depth: Go Deeper", status: "bad" };
  }
}

function calculateOverallScore(torso, hip, depth) {
  const overall =
    (torso * TORSO_WEIGHT) / 100 + (hip * HIP_WEIGHT) / 100 + (depth * DEPTH_WEIGHT) / 100;
  return Math.trunc(overall);
}

function average(arr) {
  return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

// ==============================
// Model setup
// ==============================

async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
  drawingUtils = new DrawingUtils(ctx);
}

// ==============================
// Video handling
// ==============================

videoInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Reset per-video state
  torsoScores = [];
  hipScores = [];
  minKneeAngle = 180;
  hipStartX = null;

  try {
    if (!poseLandmarker) {
      showStage(loadingStage);
      await initPoseLandmarker();
    }

    const url = URL.createObjectURL(file);
    sourceVideo.src = url;

    sourceVideo.onloadedmetadata = () => {
      canvas.width = sourceVideo.videoWidth;
      canvas.height = sourceVideo.videoHeight;
      showStage(analysisStage);
      sourceVideo.play();
    };

    sourceVideo.onended = () => {
      finishAnalysis();
    };
  } catch (err) {
    console.error(err);
    showError(
      "Couldn't load the pose model or video. Check your connection and try again."
    );
  }
});

function renderLoop() {
  if (sourceVideo.paused || sourceVideo.ended) return;

  const nowMs = performance.now();
  const result = poseLandmarker.detectForVideo(sourceVideo, nowMs);

  ctx.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);

  if (result.landmarks && result.landmarks.length > 0) {
    const lm = result.landmarks[0];
    processFrame(lm);

    drawingUtils.drawLandmarks(lm, { radius: 3, color: "#ff5a1f" });
    drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, {
      color: "#f2f0ea",
      lineWidth: 2
    });
  }

  requestAnimationFrame(renderLoop);
}

sourceVideo.addEventListener("play", () => {
  requestAnimationFrame(renderLoop);
});

function toPoint(landmark) {
  return [landmark.x, landmark.y];
}

function processFrame(landmarks) {
  const side = detectVisibleSide(landmarks);

  const shoulder = toPoint(landmarks[side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER]);
  const hip = toPoint(landmarks[side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP]);
  const knee = toPoint(landmarks[side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE]);
  const ankle = toPoint(landmarks[side === "left" ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE]);

  const kneeAngle = calculateAngle(hip, knee, ankle);

  // Track deepest point reached (lowest knee angle) across the whole video
  if (kneeAngle < minKneeAngle) {
    minKneeAngle = kneeAngle;
  }

  const torsoData = scoreTorso(shoulder, hip, knee, ankle);
  const depthData = scoreDepth(kneeAngle); // live readout only, final score uses deepest point

  if (kneeAngle >= 160) {
    hipStartX = hip[0];
  }

  const hipData = scoreHipHinge(hip[0], ankle[0], kneeAngle, hipStartX);

  if (torsoData.score !== null) torsoScores.push(torsoData.score);
  if (hipData.score !== null) hipScores.push(hipData.score);

  updateLiveReadout(liveTorso, torsoData);
  updateLiveReadout(liveHip, hipData);
  updateLiveReadout(liveDepth, depthData);
}

function updateLiveReadout(el, data) {
  el.textContent = data.label;
  el.classList.remove("status-good", "status-warn", "status-bad");
  if (data.status === "good") el.classList.add("status-good");
  if (data.status === "warn") el.classList.add("status-warn");
  if (data.status === "bad") el.classList.add("status-bad");
}

// ==============================
// Results
// ==============================

function scoreClass(score) {
  if (score >= 80) return "score-good";
  if (score >= 60) return "score-warn";
  return "score-bad";
}

function finishAnalysis() {
  if (torsoScores.length === 0 || hipScores.length === 0) {
    showError(
      "Couldn't detect a full squat in that video. Try a clearer side-on angle with your whole body in frame."
    );
    return;
  }

  const finalTorso = Math.trunc(average(torsoScores));
  const finalHip = Math.trunc(average(hipScores));
  const finalDepthData = scoreDepth(minKneeAngle);
  const finalDepth = finalDepthData.score;
  const finalOverall = calculateOverallScore(finalTorso, finalHip, finalDepth);

  overallNumber.textContent = finalOverall;
  overallNumber.className = "overall-number " + scoreClass(finalOverall);

  torsoNumber.textContent = finalTorso;
  torsoNumber.className = "sub-score-number " + scoreClass(finalTorso);

  hipNumber.textContent = finalHip;
  hipNumber.className = "sub-score-number " + scoreClass(finalHip);

  depthNumber.textContent = finalDepth;
  depthNumber.className = "sub-score-number " + scoreClass(finalDepth);

  deepestNote.textContent = `Deepest knee angle reached: ${Math.trunc(minKneeAngle)}°`;

  showStage(resultsStage);
}

// ==============================
// Reset / try another
// ==============================

function resetToUpload() {
  sourceVideo.pause();
  sourceVideo.removeAttribute("src");
  sourceVideo.load();
  videoInput.value = "";
  showStage(uploadStage);
}

tryAnotherBtn.addEventListener("click", resetToUpload);
errorRetryBtn.addEventListener("click", resetToUpload);
