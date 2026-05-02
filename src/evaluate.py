import os
from ultralytics import YOLO
import argparse
import yaml
import json

def evaluate_model(config_path, args=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_yaml = config['dataset']['yaml_path']
    
    model_weights = getattr(args, 'weights', None) if args else None
    model_weights = model_weights if model_weights is not None else config['testing']['model_weights']
    
    imgsz = getattr(args, 'imgsz', None) if args else None
    imgsz = imgsz if imgsz is not None else config['testing']['imgsz']
    
    batch = getattr(args, 'batch', None) if args else None
    batch = batch if batch is not None else config['testing']['batch']
    
    device = getattr(args, 'device', None) if args else None
    device = device if device is not None else config['testing']['device']
    
    conf = getattr(args, 'conf', None) if args else None
    conf = conf if conf is not None else config.get('testing', {}).get('conf', 0.25)
    
    iou = getattr(args, 'iou', None) if args else None
    iou = iou if iou is not None else config.get('testing', {}).get('iou', 0.6)
    
    split = getattr(args, 'split', None) if args else None
    split = split if split is not None else config.get('testing', {}).get('split', 'test')
    
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

    # Save sample predictions for visualization
    try:
        import random
        from pathlib import Path
        
        # Attempt to read the dataset yaml to find the split directory
        with open(dataset_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        split_dir = data_config.get(split) # typically 'val' or 'test'
        if split_dir:
            # Handle relative paths based on data.yaml location
            base_dir = os.path.dirname(os.path.abspath(dataset_yaml))
            if not os.path.isabs(split_dir):
                split_dir = os.path.join(base_dir, split_dir)
            
            # Find all images in the split directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            test_images = []
            for ext in image_extensions:
                test_images.extend(list(Path(split_dir).rglob(f'*{ext}')))
                test_images.extend(list(Path(split_dir).rglob(f'*{ext.upper()}')))
                
            if test_images:
                num_samples = config.get('testing', {}).get('save_samples', 5)
                num_samples = min(num_samples, len(test_images))
                
                print(f"Running predictions on {num_samples} sample images for visualization...")
                sample_images = [str(p) for p in random.sample(test_images, num_samples)]
                
                # Run prediction
                model.predict(
                    source=sample_images,
                    save=True,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    project="runs/detect",
                    name=f"eval_samples_{split}",
                    exist_ok=True
                )
                print(f"Saved individual sample predictions to runs/detect/eval_samples_{split}\n")
    except Exception as e:
        print(f"Notice: Could not automatically generate sample predictions. ({e})")

    # Save to file
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved evaluation metrics to {output_file}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Model")
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to test config file')
    parser.add_argument('--weights', type=str, help='Path to model weights (overrides yaml)')
    parser.add_argument('--imgsz', type=int, help='Image size (overrides yaml)')
    parser.add_argument('--batch', type=int, help='Batch size (overrides yaml)')
    parser.add_argument('--device', type=str, help='Device, e.g., 0 or cpu (overrides yaml)')
    parser.add_argument('--conf', type=float, help='Confidence threshold (overrides yaml)')
    parser.add_argument('--iou', type=float, help='NMS IoU threshold (overrides yaml)')
    parser.add_argument('--split', type=str, help='Split to evaluate on: val or test (overrides yaml)')
    args = parser.parse_args()
    
    evaluate_model(args.config, args)
