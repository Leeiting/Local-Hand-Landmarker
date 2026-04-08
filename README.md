# Local-Hand-Landmarker
A lightweight, local implementation of the MediaPipe Hand Landmarker. Real-time 21-point hand skeleton tracking with high precision and low latency.


一個基於本地運行的手部關鍵點偵測工具，深度復刻並優化了 **MediaPipe Hand Landmarker** 的核心功能。本專案旨在提供高效、低延遲的手部 21 個關鍵點（Landmarks）追蹤解決方案，適用於手勢識別、人機互動與 AR 應用。

## ✨ 特色功能

* **本地推論 (Local Inference):** 無需上傳雲端，保護隱私且減少網路延遲。
* **精準追蹤:** 完整還原 21 個手部關節點。
* **實時性能:** 針對 CPU/GPU 進行優化，支援高影格率（FPS）檢測。
* **易於集成:** 模組化設計，可輕鬆嵌入其他 Python 或 C++ 專案。

## 📍 關鍵點分佈 (21 Landmarks)

本專案遵循 MediaPipe 標準的手部座標定義：
0. WRIST (腕部)
1-4. THUMB (拇指)
5-8. INDEX_FINGER (食指)
... (以此類推)

## 🚀 快速上手

### 環境需求
* Python 3.8+
* OpenCV
* (列出你使用的其他框架，例如 NumPy, TensorFlow 或 PyTorch)

