from ultralytics import YOLO

def main():
    print("Initializing YOLOv8m-P2 High-Resolution model...")
    model = YOLO("configs/rank1_yolov8m.yaml").load("yolov8m.pt") #since we have CUSTOMISED yolo arch, so using this ...
    #this basically means taking trained weights of yolov8m and training it on our custom arch(with new layer wts rando ini.
    
    # Execute Distributed Data Parallel (DDP) training across 6 GPUs
    print("Launching DDP Training on devices: 0,1,2,3,4,5")
    results = model.train(
        data="data/data_pca_10k.yaml", 
        epochs=20, 
        batch=128, 
        device="0,1,2,3,4,5", 
        optimizer="SGD", 
        momentum=0.9, 
        weight_decay=5e-4, 
        lr0=1e-2, 
        rect=True, #rectangular img OK, no need to resize 
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
        name="rank1_train_full"
    )

if __name__ == '__main__':
    main()
