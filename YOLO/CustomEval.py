import torch
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def yolo_txt_to_tensor(txt_path, img_width, img_height):
    """Converts a YOLO .txt file into the format needed for evaluation"""
    if not Path(txt_path).exists():
        return {'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.int64), 'scores': torch.empty((0,))}

    boxes, labels, scores = [], [], []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            # YOLO format: center_x, center_y, width, height (normalized)
            cx, cy, w, h = map(float, parts[1:5])
            
            # If your blackbox outputs confidence scores, it's usually the 6th column.
            # Ground truth files won't have this, so default to 1.0
            score = float(parts[5]) if len(parts) > 5 else 1.0

            # Convert normalized YOLO to absolute xyxy (x_min, y_min, x_max, y_max)
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            x2 = (cx + w / 2) * img_width
            y2 = (cy + h / 2) * img_height

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            scores.append(score)

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4)),
        'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
        'scores': torch.tensor(scores, dtype=torch.float32) if scores else torch.empty((0,))
    }

def evaluate_folders():
    # Setup Paths
    gt_folder = Path("datasets/ThermalData/labels/val") # Path to true labels
    pred_folder = Path("path/to/your/blackbox/predictions") # Path to your blackbox outputs
    
    # Image dimensions (assumed 640x640 based on your training script)
    IMG_W, IMG_H = 640, 640 

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    # Loop through all ground truth files
    for gt_file in gt_folder.glob("*.txt"):
        pred_file = pred_folder / gt_file.name
        
        # Read Ground Truth
        target = yolo_txt_to_tensor(gt_file, IMG_W, IMG_H)
        
        # Read Blackbox Prediction
        pred = yolo_txt_to_tensor(pred_file, IMG_W, IMG_H)
        
        # Add to metric calculator
        metric.update([pred], [target])

    # Calculate final scores
    results = metric.compute()
    print(f"Blackbox mAP@50:      {results['map_50'].item():.4f}")
    print(f"Blackbox mAP@50-95:   {results['map'].item():.4f}")

if __name__ == '__main__':
    evaluate_folders()