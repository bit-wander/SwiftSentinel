import argparse
import yaml
import os
from ultralytics import YOLO

def run_inference(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    infer_config = config['inference']
    model_weights = infer_config['model_weights']
    source = infer_config['source']
    imgsz = infer_config.get('imgsz', 640)
    conf = infer_config.get('conf', 0.25)
    iou = infer_config.get('iou', 0.45)
    device = infer_config.get('device', '')
    
    track = infer_config.get('track', False)
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
    args = parser.parse_args()
    
    run_inference(args.config)
