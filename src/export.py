import argparse
import os
from ultralytics import YOLO

def export_model(weights_path, format, imgsz, dynamic):
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    print(f"Loading model from {weights_path}...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Starting export to '{format}' format...")
    print("Note: Exporting to TensorRT (.engine) or CoreML may require additional system dependencies.")
    
    # Export the model
    path = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=dynamic
    )
    
    print(f"\nModel exported successfully!")
    print(f"Saved to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 Model to Edge Device Formats")
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='Path to trained YOLO PyTorch weights (.pt)')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'engine', 'tflite', 'coreml', 'torchscript', 'openvino'], help='Target export format')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for export')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic axes (useful for ONNX/TensorRT if batch size or image size varies)')
    
    args = parser.parse_args()
    export_model(args.weights, args.format, args.imgsz, args.dynamic)
