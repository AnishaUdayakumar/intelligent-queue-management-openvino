from PIL import Image
from ultralytics import YOLO
import os
import cv2

def main():
    # Path to the image file
    IMAGE_PATH = "data/pexels-catia-matos-1604200.jpg"

    # Name of the YOLO model we want to use
    DET_MODEL_NAME = "yolov8m"

    # Load the image
    image = Image.open(IMAGE_PATH)

    # Create a YOLO object detection model
    det_model = YOLO(f"model/{DET_MODEL_NAME}.pt")

    folder_path = 'model/'
    file_path = os.path.join(folder_path, 'yolov8m.pt')
    
    # confirm that the image was saved successfully
    if os.path.exists(file_path):
        print(f"Yolov8 Pytorch model saved successfully at {file_path}")
    else:
        print(f"Failed to save model at {file_path}")
        
    # Export the model to OpenVINO format
    det_model.export(format="openvino", dynamic=True, half=True)
    
    folder_path = 'model/yolov8m_openvino_model/'
    file_path = os.path.join(folder_path, 'yolov8m.xml')
    
    # confirm that the image was saved successfully
    if os.path.exists(file_path):
        print(f"Yolov8 IR model converted successfully at {file_path}")
    else:
        print(f"Failed to save model at {file_path}")
    
    # Detect objects of class 0 (person) in the image
    res = det_model(image, classes=[0])[0]
    
    
if __name__ == "__main__":
    main()
