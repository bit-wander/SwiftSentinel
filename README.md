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

## 📊 Dataset Information

The model is trained on a custom dataset (`master_dataset`) consisting of aerial imagery for UAV detection. 

### Classes & Backgrounds
- `0`: `uav` (Unmanned Aerial Vehicle)
- **Background Images**: The dataset intentionally includes background images (images with no UAVs, represented by empty or missing label files) to teach the model to reduce False Positives (FP) during inference.

### Dataset Distribution
The combined dataset contains a total of **15,865 images**, carefully split and balanced to include background examples:

| Split | Total Images | UAV Images | Background Images |
| :--- | :--- | :--- | :--- |
| **Train** | 11,921 | 11,567 | 354 |
| **Valid** | 2,096 | 1,995 | 101 |
| **Test** | 1,848 | 1,797 | 51 |
| **Total** | **15,865** | **15,359** | **506** |

### Structure & Sample Data
The dataset follows the standard YOLO format and is split into `train`, `valid`, and `test` directories.

**Directory Structure:**
```text
master_dataset/
├── train/
│   ├── images/ (e.g., frame_001.jpg)
│   └── labels/ (e.g., frame_001.txt)
├── val/
...
```

**Sample Annotation (`.txt` file):**
Each label file contains one row per bounding box in the format: `<class_id> <x_center> <y_center> <width> <height>` (normalized between 0 and 1).
```text
# class_id  x_center   y_center   width      height
0           0.5432     0.3121     0.0543     0.0215
```

**Visualization Samples:**

<p align="center">
  <img src="assets/samples/sample_1.jpg" width="31%" alt="UAV Dataset Sample 1" />
  <img src="assets/samples/sample_2.jpg" width="31%" alt="UAV Dataset Sample 2" />
  <img src="assets/samples/sample_3.jpg" width="31%" alt="UAV Dataset Sample 3" />
</p>


---

## 🧠 Pipeline Usage

### 1. Training
Configure your dataset and training parameters in `configs/train_config.yaml`, or override them directly via the command line.
*Note: To track experiments with ClearML, uncomment the `clearml` section in the config.*

Run the training script:
```bash
# Run with defaults from yaml
python src/train.py --config configs/train_config.yaml

# Override model, epochs, and batch size via CLI
python src/train.py --model yolov8s.pt --epochs 50 --batch 64
```

### 2. Evaluation
To evaluate your trained model against a validation or test set and generate standard metrics (mAP50, mAP50-95):

Update `configs/test_config.yaml` to point to your dataset and trained weights, or override them directly via the command line:
```bash
# Run with defaults from yaml
python src/evaluate.py --config configs/test_config.yaml

# Override weights, split, and confidence threshold via CLI
python src/evaluate.py --weights runs/detect/train/weights/best.pt --split val --conf 0.50
```
*Results will be displayed in the terminal and saved in `evaluation_results.json`.*

### 3. Inference & Tracking
You can run real-time inference or tracking on video files, image directories, or webcam streams. Edit `configs/infer_config.yaml` to set your target `source`, `track` flag (to enable tracking), and thresholds, or override them dynamically via the command line.

```bash
# Run inference with defaults from yaml
python src/infer.py --config configs/infer_config.yaml

# Override source and enable tracking via CLI
python src/infer.py --source data/test_video.mp4 --track --conf 0.4
```

### 4. Edge Export
Deploying to a Jetson Nano, Raspberry Pi, or other edge devices requires optimized model formats. You can easily export the `.pt` weights to formats like ONNX or TensorRT (`.engine`).

```bash
python src/export.py --weights runs/detect/train/weights/best.pt --format onnx
```

