from ultralytics import YOLO

def main():
    print("Initializing YOLOv12m model for Ablation Study...")
    
    # Using standard YOLOv12m architecture. 
    # We load it directly from the pretrained weights to let Ultralytics auto-download it.
    model = YOLO("yolo12m.pt")
    
    # Execute Distributed Data Parallel (DDP) training across 6 GPUs
    print("Launching DDP Training on devices: 0,1,2,3,4,5")
    results = model.train(
        data="data/data_pca_10k.yaml",  # Strictly using the 10k subset
        epochs=20, 
        batch=128, 
        device="0,1,2,3,4,5", 
        optimizer="SGD", 
        momentum=0.9, 
        weight_decay=5e-4, 
        lr0=1e-2, 
        rect=True,
        imgsz=[288, 384],
        workers=32,
        mosaic=0.0, 
        mixup=0.0, 
        cutmix=0.0, 
        copy_paste=0.0, 
        fliplr=0.5, 
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        name="ablation_yolov12m_10k"
    )

if __name__ == '__main__':
    main()
