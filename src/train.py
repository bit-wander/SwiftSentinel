import os
from ultralytics import YOLO
import argparse
import yaml

def train_model(config_path, args=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup ClearML environment variables if provided
    if 'clearml' in config and config['clearml']:
        for key, value in config['clearml'].items():
            if value:
                os.environ[key] = str(value)
            
    dataset_yaml = config['dataset']['yaml_path']
    epochs = getattr(args, 'epochs', None) if args else None
    epochs = epochs if epochs is not None else config['training']['epochs']
    
    imgsz = getattr(args, 'imgsz', None) if args else None
    imgsz = imgsz if imgsz is not None else config['training']['imgsz']
    
    batch = getattr(args, 'batch', None) if args else None
    batch = batch if batch is not None else config['training']['batch']
    
    workers = config['training']['workers']
    
    device = getattr(args, 'device', None) if args else None
    device = device if device is not None else config['training']['device']
    
    model_name = getattr(args, 'model', None) if args else None
    model_name = model_name if model_name is not None else config['training']['model']
    
    # Prevent ultralytics from tracking activity (as in original notebook)
    os.environ['YOLO_SETTINGS_SYNC'] = 'False'
    
    # Initialize YOLO model
    model = YOLO(model_name)
    
    # Train the model
    print(f"Starting training for {epochs} epochs on device(s) {device}...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        plots=True
    )
    
    print("Training completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Model")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, help='Model name or path to weights (overrides yaml)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides yaml)')
    parser.add_argument('--batch', type=int, help='Batch size (overrides yaml)')
    parser.add_argument('--imgsz', type=int, help='Image size (overrides yaml)')
    parser.add_argument('--device', type=str, help='Device, e.g., 0 or cpu (overrides yaml)')
    args = parser.parse_args()
    
    train_model(args.config, args)
