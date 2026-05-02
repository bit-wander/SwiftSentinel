import os
from ultralytics import YOLO
import argparse
import yaml
import json

def evaluate_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_yaml = config['dataset']['yaml_path']
    model_weights = config['testing']['model_weights']
    imgsz = config['testing']['imgsz']
    batch = config['testing']['batch']
    device = config['testing']['device']
    conf = config.get('testing', {}).get('conf', 0.25)
    iou = config.get('testing', {}).get('iou', 0.6)
    split = config.get('testing', {}).get('split', 'test')
    
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return
        
    # Prevent ultralytics from tracking activity
    os.environ['YOLO_SETTINGS_SYNC'] = 'False'
    
    # Initialize YOLO model
    print(f"Loading model from {model_weights}...")
    model = YOLO(model_weights)
    
    # Evaluate the model
    print(f"Starting evaluation on '{split}' split...")
    metrics = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        split=split,
        plots=True
    )
    
    # Print evaluation metrics
    print("\n" + "="*40)
    print("         EVALUATION METRICS")
    print("="*40)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    
    # Prepare results dictionary for saving
    results_dict = {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
    }
    
    # Extract class-specific metrics if available
    try:
        class_names = model.names
        class_metrics = {}
        for i, class_name in class_names.items():
            if i < len(metrics.box.maps):
                class_metrics[class_name] = float(metrics.box.maps[i])
        results_dict["class_metrics_mAP50-95"] = class_metrics
        
        print("\n--- Class-specific mAP50-95 ---")
        for cls_name, cls_map in class_metrics.items():
            print(f"{cls_name}: {cls_map:.4f}")
    except Exception as e:
        print(f"\nCould not extract class-specific metrics: {e}")

    print("="*40 + "\n")

    # Save to file
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved evaluation metrics to {output_file}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Model")
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to test config file')
    args = parser.parse_args()
    
    evaluate_model(args.config)
