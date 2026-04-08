import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

async function initLandmarker() {
  statusEl.textContent = "狀態：載入模型中...";
  const resolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "VIDEO",
    numHands: 2,
  });
  statusEl.textContent = "狀態：模型就緒，請啟動攝影機";
}

async function startCamera() {
  if (!handLandmarker) await initLandmarker();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720 },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth || 960;
  canvas.height = video.videoHeight || 720;
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "狀態：偵測中";

  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "狀態：已停止";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/** 與自拍鏡像一致：畫面上的像素 x（未再 scale 時） */
function mirroredPixelX(normX) {
  return canvas.width - normX * canvas.width;
}

function sub3(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function len3(v) {
  return Math.hypot(v.x, v.y, v.z);
}

function norm3(v) {
  const l = len3(v) || 1;
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

function cross3(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

/**
 * 以手腕為原點的右手座標系（模型空間）：
 * X ≈ 腕 → 中指 MCP，Z ≈ 手掌法向（X × 腕→食指），Y = Z × X。
 */
function computeHandBasis(hand) {
  const w = hand[0];
  const toMiddle = sub3(hand[9], w);
  const toIndex = sub3(hand[5], w);
  let ex = norm3(toMiddle);
  if (len3(toMiddle) < 1e-9) ex = { x: 0, y: -1, z: 0 };
  let ez = norm3(cross3(ex, toIndex));
  if (len3(ez) < 1e-6) ez = { x: 0, y: 0, z: 1 };
  const ey = norm3(cross3(ez, ex));
  return { ex, ey, ez };
}

/** 將模型空間單位向量投影到畫布（簡化斜投影，讓 Z 深度有視覺分量） */
function projectAxisToCanvas(v, scale) {
  return {
    x: (v.x - 0.42 * v.z) * scale,
    y: (v.y + 0.32 * v.z) * scale,
  };
}

/** 手腕到各關節在畫布上的最遠距離（像素），用來讓三軸長度約涵蓋整隻手 */
function handScreenSpanPx(hand) {
  const w = hand[0];
  const ox = mirroredPixelX(w.x);
  const oy = w.y * canvas.height;
  let maxD = 0;
  for (let i = 1; i < hand.length; i++) {
    const p = hand[i];
    const px = mirroredPixelX(p.x);
    const py = p.y * canvas.height;
    const d = Math.hypot(px - ox, py - oy);
    if (d > maxD) maxD = d;
  }
  const cap = Math.min(canvas.width, canvas.height) * 0.48;
  const minSpan = 72;
  return Math.min(cap, Math.max(minSpan, maxD * 1.12));
}

function drawArrow2d(x0, y0, x1, y1, color, label, headSize) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  const ang = Math.atan2(y1 - y0, x1 - x0);
  const ah = headSize;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ah * Math.cos(ang - 0.5), y1 - ah * Math.sin(ang - 0.5));
  ctx.lineTo(x1 - ah * Math.cos(ang + 0.5), y1 - ah * Math.sin(ang + 0.5));
  ctx.closePath();
  ctx.fill();

  ctx.font = "bold 15px system-ui, sans-serif";
  ctx.textBaseline = "bottom";
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.lineWidth = 4;
  ctx.strokeText(label, x1 + 6, y1 - 4);
  ctx.fillStyle = color;
  ctx.fillText(label, x1 + 6, y1 - 4);
}

function drawWristAxisTriad(hand) {
  const w = hand[0];
  const { ex, ey, ez } = computeHandBasis(hand);
  const ox = mirroredPixelX(w.x);
  const oy = w.y * canvas.height;
  const axisLen = handScreenSpanPx(hand);
  const headSize = Math.max(10, Math.min(18, axisLen * 0.14));

  const axes = [
    { v: ex, color: "#ef4444", label: "X" },
    { v: ey, color: "#22c55e", label: "Y" },
    { v: ez, color: "#3b82f6", label: "Z" },
  ];

  for (const { v, color, label } of axes) {
    const p = projectAxisToCanvas(v, axisLen);
    drawArrow2d(ox, oy, ox + p.x, oy + p.y, color, label, headSize);
  }

  ctx.fillStyle = "#f8fafc";
  ctx.strokeStyle = "rgba(0,0,0,0.5)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(ox, oy, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawHandAxisTriads(result) {
  if (!result.landmarks) return;
  for (const hand of result.landmarks) {
    drawWristAxisTriad(hand);
  }
}

function drawHandGeometry(result) {
  if (!result.landmarks) return;

  ctx.lineWidth = 3;
  for (const hand of result.landmarks) {
    ctx.strokeStyle = "#22c55e";
    for (const [i, j] of connections) {
      const a = hand[i];
      const b = hand[j];
      ctx.beginPath();
      ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
      ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
      ctx.stroke();
    }

    ctx.fillStyle = "#fb923c";
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function loop() {
  if (!running || !handLandmarker) return;

  if (video.currentTime === lastVideoTime) {
    requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = video.currentTime;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const result = handLandmarker.detectForVideo(video, performance.now());

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawHandGeometry(result);
  ctx.restore();

  drawHandAxisTriads(result);

  ctx.save();
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "rgba(15,23,42,0.85)";
  ctx.strokeStyle = "rgba(148,163,184,0.6)";
  ctx.lineWidth = 1;
  roundRectPath(ctx, 8, 8, Math.min(canvas.width - 16, 560), 52, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#e2e8f0";
  ctx.fillText("RGB 軸以手腕為原點：紅 X＝腕→中指 MCP；綠 Y、藍 Z＝右手正交系。", 16, 28);
  ctx.fillText("畫布上為斜投影（非真實公制）；雙手時各手各有一組軸。", 16, 44);
  ctx.restore();

  requestAnimationFrame(loop);
}

function roundRectPath(c, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  c.beginPath();
  c.moveTo(x + rr, y);
  c.arcTo(x + w, y, x + w, y + h, rr);
  c.arcTo(x + w, y + h, x, y + h, rr);
  c.arcTo(x, y + h, x, y, rr);
  c.arcTo(x, y, x + w, y, rr);
  c.closePath();
}

startBtn.addEventListener("click", () => {
  startCamera().catch((e) => {
    console.error(e);
    const reason = e && e.name ? `${e.name}${e.message ? `: ${e.message}` : ""}` : "UnknownError";
    statusEl.textContent = `狀態：啟動失敗（${reason}）`;
  });
});

stopBtn.addEventListener("click", stopCamera);

initLandmarker().catch((e) => {
  console.error(e);
  statusEl.textContent = "狀態：模型載入失敗";
});
