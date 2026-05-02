import argparse
import yaml
import os
from ultralytics import YOLO

def run_inference(config_path, args=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    infer_config = config['inference']
    
    model_weights = getattr(args, 'weights', None) if args else None
    model_weights = model_weights if model_weights is not None else infer_config['model_weights']
    
    source = getattr(args, 'source', None) if args else None
    source = source if source is not None else infer_config['source']
    
    imgsz = getattr(args, 'imgsz', None) if args else None
    imgsz = imgsz if imgsz is not None else infer_config.get('imgsz', 640)
    
    conf = getattr(args, 'conf', None) if args else None
    conf = conf if conf is not None else infer_config.get('conf', 0.25)
    
    iou = infer_config.get('iou', 0.45)
    
    device = getattr(args, 'device', None) if args else None
    device = device if device is not None else infer_config.get('device', '')
    
    track = getattr(args, 'track', False) if args and getattr(args, 'track', False) else infer_config.get('track', False)
    tracker = infer_config.get('tracker', 'bytetrack.yaml')
    
    save_result = infer_config.get('save_result', True)
    save_txt = infer_config.get('save_txt', False)
    show = infer_config.get('show', False)

    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at '{model_weights}'")
        return

    if not os.path.exists(source) and source != '0':
        print(f"Warning: Source '{source}' not found. Make sure the path is correct or use '0' for webcam.")

    print(f"Loading model from {model_weights}...")
    model = YOLO(model_weights)
    
    action = 'tracking' if track else 'inference'
    print(f"Running {action} on source: {source}")
    
    # Set up the common arguments for prediction/tracking
    kwargs = {
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": device,
        "save": save_result,
        "save_txt": save_txt,
        "show": show,
        "stream": True  # Use stream=True for memory efficiency on long videos
    }

    if track:
        kwargs["tracker"] = tracker
        results = model.track(**kwargs)
    else:
        results = model.predict(**kwargs)

    # We must iterate through the generator to actually process the frames
    for r in results:
        pass
        
    print(f"\n{action.capitalize()} completed.")
    if save_result:
        print("Check the 'runs/detect/' directory for saved outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 Inference and Tracking")
    parser.add_argument('--config', type=str, default='configs/infer_config.yaml', help='Path to inference config file')
    parser.add_argument('--weights', type=str, help='Path to model weights (overrides yaml)')
    parser.add_argument('--source', type=str, help='Source (image/video path or 0 for webcam) (overrides yaml)')
    parser.add_argument('--imgsz', type=int, help='Image size (overrides yaml)')
    parser.add_argument('--conf', type=float, help='Confidence threshold (overrides yaml)')
    parser.add_argument('--device', type=str, help='Device, e.g., 0 or cpu (overrides yaml)')
    parser.add_argument('--track', action='store_true', help='Enable tracking (overrides yaml to True if set)')
    args = parser.parse_args()
    
    run_inference(args.config, args)
