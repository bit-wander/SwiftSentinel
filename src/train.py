import os
from ultralytics import YOLO
import argparse
import yaml

def train_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup ClearML environment variables if provided
    if 'clearml' in config and config['clearml']:
        for key, value in config['clearml'].items():
            if value:
                os.environ[key] = str(value)
            
    dataset_yaml = config['dataset']['yaml_path']
    epochs = config['training']['epochs']
    imgsz = config['training']['imgsz']
    batch = config['training']['batch']
    workers = config['training']['workers']
    device = config['training']['device']
    model_name = config['training']['model']
    
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
    args = parser.parse_args()
    
    train_model(args.config)
