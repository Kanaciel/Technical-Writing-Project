from pathlib import Path
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
from PIL import Image


#------------variables--------------#
root_path = Path(__file__).resolve().parent.parent
thermal_test_images_path = root_path / "Data" / "dataset" / "test"/"images"
thermal_test_labels_path = root_path / "Data" / "dataset" / "test"/"labels"
thermal_train_images_path = root_path / "Data" / "dataset" / "train"/"images"
thermal_train_labels_path = root_path / "Data" / "dataset" / "train"/"labels"


thermal_class_names = [
    "human"
]


trans_img_to_tensor = transforms.Compose([
      transforms.Resize((640,640)),
      transforms.Grayscale(num_output_channels=3),  
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


#--------------classes------------------#

class YoloThermalDataset(Dataset):
      def __init__(self, images_dir, labels_dir, transform=None):
            self.images_dir = Path(images_dir)
            self.labels_dir = Path(labels_dir)
            self.transform = transform
            
            # Get all image files
            valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff')
            self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])

      def __len__(self):
        return len(self.image_files)
      
      def __getitem__(self, idx):
        # Load Image (Convert to RGB in case thermal images are 1-channel grayscale)
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
        # Load Corresponding YOLO Label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = self.labels_dir / label_name
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # YOLO format: class_id center_x center_y width height
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    labels.append(int(class_id))
                    boxes.append([cx, cy, w, h])
        
        # Convert to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target


#------functions------#      

# adjust for different amount of labels per images
def collate_fn(batch):
    return tuple(zip(*batch))




def create_train_and_test_dataset(batch_size):
    # Initialize the separated Train and Test datasets
      train_dataset = YoloThermalDataset(images_dir=thermal_train_images_path, 
                                       labels_dir=thermal_train_labels_path, 
                                       transform=trans_img_to_tensor)
    
      test_dataset = YoloThermalDataset(images_dir=thermal_test_images_path, 
                                      labels_dir=thermal_test_labels_path, 
                                      transform=trans_img_to_tensor)

    # Note: collate_fn is added to handle variable number of bounding boxes
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

      return train_dataset, test_dataset, train_loader, test_loader


## Display random img from dataset with bounding boxes
def show_random_sample(dataset, class_names=None):
    index = random.randint(0, len(dataset) - 1)
    img, target = dataset[index]

    # Convert tensor (C,H,W) → (H,W,C)
    if isinstance(img, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")

    img_h, img_w, _ = img.shape

    # Draw YOLO bounding boxes
    boxes = target["boxes"]
    labels = target["labels"]
    
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        
        # Un-normalize YOLO relative coordinates to actual pixel coordinates
        pixel_cx, pixel_cy = cx * img_w, cy * img_h
        pixel_w, pixel_h = w * img_w, h * img_h
        
        # Calculate bottom-left corner for matplotlib
        x_min = pixel_cx - (pixel_w / 2)
        y_min = pixel_cy - (pixel_h / 2)
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), pixel_w, pixel_h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label text
        if class_names:
            class_id = labels[i].item()
            class_name = class_names[class_id]
            ax.text(x_min, y_min - 5, class_name, color='white', backgroundcolor='red', fontsize=10, fontweight='bold')

    plt.title(f"Thermal Sample: {len(boxes)} object(s) detected")
    plt.show()

#-------------testing---------------#


if __name__ == "__main__":
    # Test setting up the loaders
      train_dataset, test_dataset, train_loader, test_loader = create_train_and_test_dataset(batch_size=4)
    
    # Show a sample from the training set
      show_random_sample(train_dataset, thermal_class_names)
      pass