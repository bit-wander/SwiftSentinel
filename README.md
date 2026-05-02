# 🚁 UAV Object Detection and Tracking

An end-to-end computer vision pipeline for detecting and tracking Unmanned Aerial Vehicles (UAVs) using YOLOv8. 

This project provides a complete framework covering dataset preparation, model training, evaluation, real-time tracking, and edge device export. It is built to be highly modular, reproducible, and ready for deployment on UAV hardware.

---

## 📁 Project Structure

```text
object_detection_and_tracking/
├── configs/                  # YAML configuration files
│   ├── train_config.yaml     # Training parameters
│   ├── test_config.yaml      # Evaluation metrics parameters
│   └── infer_config.yaml     # Inference and tracking parameters
├── src/                      # Source code for the pipeline
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation and metrics generation
│   ├── infer.py              # Inference and tracking script
│   └── export.py             # Export to ONNX, TensorRT, CoreML, etc.
├── notebook/                 # Exploratory Jupyter Notebooks
├── data/                     # (Ignored) Datasets and test media
├── runs/                     # (Ignored) YOLOv8 output logs and weights
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Setup & Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd object_detection_and_tracking
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Pipeline Usage

### 1. Training
Configure your dataset and training parameters in `configs/train_config.yaml`.
*Note: To track experiments with ClearML, uncomment the `clearml` section in the config.*

Run the training script:
```bash
python src/train.py --config configs/train_config.yaml
```

### 2. Evaluation
To evaluate your trained model against a validation or test set and generate standard metrics (mAP50, mAP50-95):

Update `configs/test_config.yaml` to point to your dataset and trained weights, then run:
```bash
python src/evaluate.py --config configs/test_config.yaml
```
*Results will be displayed in the terminal and saved in `evaluation_results.json`.*

### 3. Inference & Tracking
You can run real-time inference or tracking on video files, image directories, or webcam streams. Edit `configs/infer_config.yaml` to set your target `source`, `track` flag (to enable ByteTrack or BoT-SORT), and thresholds.

```bash
python src/infer.py --config configs/infer_config.yaml
```

### 4. Edge Export
Deploying to a Jetson Nano, Raspberry Pi, or other edge devices requires optimized model formats. You can easily export the `.pt` weights to formats like ONNX or TensorRT (`.engine`).

```bash
python src/export.py --weights runs/detect/train/weights/best.pt --format onnx
```

