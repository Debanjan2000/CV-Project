from ultralytics import YOLO
import ultralytics.utils.loss as loss_module
import torch
import torch.nn as nn

# ---------------------------------------------------------
# CUSTOM FOCAL LOSS INJECTION (MONKEY-PATCHING ULTRALYTICS)
# ---------------------------------------------------------
# We save the original loss class
Original_v8DetectionLoss = loss_module.v8DetectionLoss

# We create our custom loss class that inherits the original
class Focal_v8DetectionLoss(Original_v8DetectionLoss):
    def __init__(self, model, bce_pos_weight=None):
        super().__init__(model)
        print("\n[+] SUCCESS: Custom Focal Loss Intercepted and Activated!\n")
        
        # Override the standard BCE loss with a custom Focal Loss
        class CustomBCEFocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=1.5):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(reduction='none')
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, pred, target):
                loss = self.bce(pred, target)
                pred_prob = torch.sigmoid(pred)
                p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                modulating_factor = (1.0 - p_t) ** self.gamma
                return (alpha_t * modulating_factor * loss).mean()
                
        # Replace the classifier's loss engine
        self.bce = CustomBCEFocalLoss(alpha=0.25, gamma=1.5)

# Replace the original class in the library's memory with our custom one
loss_module.v8DetectionLoss = Focal_v8DetectionLoss
# ---------------------------------------------------------

def main():
    print("Initializing YOLOv8m-P2 High-Resolution model...")
    model = YOLO("configs/rank1_yolov8m.yaml").load("yolov8m.pt")
    
    print("Launching DDP Training on devices: 0,1,2,3,4,5")
    results = model.train(
        data="data/data_pca.yaml", 
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
        name="rank1_train_custom_focal"
    )

if __name__ == '__main__':
    main()
