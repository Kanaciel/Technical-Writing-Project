from ultralytics import YOLO
from pathlib import Path
import torch
import random

current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



root_path = Path(__file__).resolve().parent.parent
yaml_path = (root_path / "YOLO" /"ThermalData.yaml").resolve()
output_path = (root_path / "Models").resolve()
original_yolo_model_path = (root_path / "Models" / "yolo26n.pt").resolve()
best_yolo_model_path =  (root_path / "Models" / "best.pt").resolve()


def train_model():
      YOLO_model = YOLO(str(original_yolo_model_path))

      print("Starting training...")

      results = YOLO_model.train(data = str(yaml_path),
                              epochs = 2,
                              batch = 32,
                              imgsz= 640,
                              device = current_device,
                              project = str(output_path),
                              name="yolo_thermal",
                              optimizer = 'auto',
                              workers= 1
                              )

      print(f"Training complete. Best model saved at: {results.save_dir}/weights/best.pt")

def test_model():
     # Load YOUR custom trained model
      model = YOLO(str(best_yolo_model_path))
      img_source = r"D:\class VGU edition\Technical Writing\Technical-Writing-Project\Data\A9A85578-8C0C-43B3-BE0F-DE2A7507F9CA.jpeg"
      # Run inference on a test image
      results = model(source = img_source)

      # Display the image with the bounding boxes drawn on it
      results[0].show()
      
      # If you want to save the image instead of showing it:
      # results[0].save('prediction.jpg')


def eval_model():
      YOLO_model = YOLO(str(best_yolo_model_path))
      metrics = YOLO_model.val(data=str(yaml_path), batch = 32)
      print(metrics.confusion_matrix.matrix)



if __name__ == '__main__':
      #train_model()
      #eval_model()
      test_model()