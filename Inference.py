from PIL import Image
from ultralytics import YOLO
import os

def main():
    # Path to the image file
    IMAGE_PATH = "data/pexels-catia-matos-1604200.jpg"

    # Name of the YOLO model we want to use
    DET_MODEL_NAME = "yolov8m"

    # Load the image
    image = Image.open(IMAGE_PATH)

    # Create a YOLO object detection model
    det_model = YOLO(f"model/{DET_MODEL_NAME}.pt")
    
    # class 0 means person
    res = det_model(IMAGE_PATH, classes=[0])[0]

    # create the folder if it doesn't exist
    folder_path = 'inference/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # create the image and save it to the folder
    im = Image.fromarray(res.plot(line_width=3)[:, :, ::-1])
    file_path = os.path.join(folder_path, 'pexels-catia-matos-1604200_pt.png')
    im.save(file_path)

    # confirm that the image was saved successfully
    if os.path.exists(file_path):
        print(f"Inference of Pytorch model saved successfully at {file_path}")
    else:
        print(f"Failed to save image at {file_path}")
    
    # create a model
    det_model = YOLO(f"model/yolov8m_openvino_model")

    # class 0 means person
    res = det_model(IMAGE_PATH, classes=[0])[0]

    # create the image and save it to the folder
    im = Image.fromarray(res.plot(line_width=3)[:, :, ::-1])
    file_path = os.path.join(folder_path, 'pexels-catia-matos-1604200_ir.png')
    im.save(file_path)

    # confirm that the image was saved successfully
    if os.path.exists(file_path):
        print(f"Inference of IR model saved successfully at {file_path}")
    else:
        print(f"Failed to save image at {file_path}")
    
    
if __name__ == "__main__":
    main()
