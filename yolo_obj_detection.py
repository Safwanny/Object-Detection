import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
allowed_classes = ['person', 'bottle', 'car']

# Open webcam (adjust index if needed)
cap = cv2.VideoCapture(1)

# Initialize tracker
tracker = Sort()
tracked_ids = set()

# Map class label to folder
class_folders = {
    "person": "detected_people",
    "bottle": "detected_bottles",
    "car": "detected_cars"
}

# Start video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model.predict(frame, conf=0.5, iou=0.4)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf, cls_id])  # Add class ID for later use

    # Convert to NumPy array for SORT
    if len(detections) > 0:
        dets_np = np.array([d[:5] for d in detections])  # exclude cls_id for tracker
    else:
        dets_np = np.empty((0, 5))

    # Update SORT tracker
    tracks = tracker.update(dets_np)

    # Draw results
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        # Match the track with detection to get class label
        matched_det = None
        for det in detections:
            dx1, dy1, dx2, dy2 = det[:4]
            if abs(x1 - dx1) < 10 and abs(y1 - dy1) < 10:
                matched_det = det
                break

        if matched_det is None:
            continue

        cls_id = matched_det[5]
        label = model.names[cls_id]

        if label not in allowed_classes:
            continue

        color = (255, 0, 0) if track_id in tracked_ids else (0, 255, 0)
        text = f"{label} ID:{track_id}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#        # Save crop if it's a new object
#        if track_id not in tracked_ids:
#            crop = frame[y1:y2, x1:x2]
#
#            # Check crop validity
#            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
#                folder = class_folders[label]
#                save_path = os.path.join(folder, f"{label}_{track_id}.jpg")
#                cv2.imwrite(save_path, crop)

        tracked_ids.add(track_id)

    # Show output
    cv2.imshow("YOLOv8 object detection and tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
