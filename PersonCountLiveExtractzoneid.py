import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main(camera_source):
    colors = sv.ColorPalette.default()
    polygons = [
        np.array([
            [315, 0],
            [315, 470],
            [0, 470],
            [1, 1]
        ], np.int32),
        np.array([
            [325, 0],
            [325, 470],
            [640, 470],
            [640, 0]
        ], np.int32)
    ]
    
    cap = cv2.VideoCapture(camera_source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    zones = [
        sv.PolygonZone(
            polygon=polygon, 
            frame_resolution_wh=(frame_width, frame_height)  # you can adjust this to your desired resolution
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
            print(f"zone_id: {zone.zone_id}, person count: {len(detections_filtered)}")

        # write the annotated frame to the output video file
        out.write(frame)

        # show the output live
        cv2.imshow('Output Live', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='camera source index')
    args = parser.parse_args()

    main(args.camera)
