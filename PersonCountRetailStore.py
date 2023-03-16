import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main(video_path):
    video_info = sv.VideoInfo.from_video_path(video_path)
    resolution_wh = (video_info.width, video_info.height)

    colors = sv.ColorPalette.default()
    polygons = [
        np.array([
            [464, 121],
            [850, 383],
            [927, 275],
            [594, 88]
        ], np.int32),
        np.array([
            [178, 144],
            [427, 570],
            [661, 426],
            [342, 60]
        ], np.int32)
    ]
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


    zones = [
        sv.PolygonZone(
            polygon=polygon, 
            frame_resolution_wh=resolution_wh
        )
        for index, polygon
        in enumerate(polygons)
    ]
    
    for index, zone in enumerate(zones):
        zone.zone_id = index
        
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone, 
            color=colors.by_idx(index), 
            thickness=6,
            text_thickness=8,
            text_scale=4
        )
        for index, zone
        in enumerate(zones)
    ]
    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index), 
            thickness=4, 
            text_thickness=4, 
            text_scale=2
            )
        for index
        in range(len(polygons))
    ]

    det_model = YOLO(f"model/yolov8m_openvino_model")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect persons in the current frame
        results = det_model(frame, classes=[0])[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

        # annotate the frame with the detected persons within each zone
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            frame = zone_annotator.annotate(scene=frame)
            print(f"Checkout Lane: {zone.zone_id}, Customer count: {len(detections_filtered)}")

        # write the annotated frame to the output video file
        out.write(frame)
        # show the output live
        cv2.imshow('Output Live', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='path to input video')
   

    args = parser.parse_args()

    main(args.video)


