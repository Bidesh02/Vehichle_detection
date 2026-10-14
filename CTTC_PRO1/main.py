import cv2
import os
from ultralytics import YOLO

# 1. Set model path with automatic fallback
custom_model_path = "helmet_yolov8n.pt"

if os.path.exists(custom_model_path):
    print(f"Loading custom helmet model: {custom_model_path}")
    model = YOLO(custom_model_path)
else:
    print(f"'{custom_model_path}' not found.")
    print("Auto-downloading standard 'yolov8n.pt' so the script works immediately...")
    # This downloads automatically on first run and prevents FileNotFoundError
    model = YOLO("yolov8n.pt")

# 2. Define your video path
video_path = "test_video1.mp4"

if not os.path.exists(video_path):
    print(f"Error: The video file '{video_path}' was not found in your folder.")
    exit()

# 3. Open the video source
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video source: {video_path}")
    exit()

print("Success! Running detection. Press 'q' to exit.")

# 4. Process the video frames in a loop
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video stream.")
        break

    # Run YOLOv8 inference (conf=0.4 keeps detections with >= 40% confidence)
    results = model.predict(frame, conf=0.4, verbose=False)

    # Loop through detected boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Color coding: Red for missing helmet/safety gear, Green for positive detection
        if any(word in class_name.lower() for word in ["no", "without", "head"]):
            color = (0, 0, 255)  # Red (BGR)
        else:
            color = (0, 255, 0)  # Green (BGR)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Create label with confidence score
        label = f"{class_name} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )

    # Display video output
    cv2.imshow("Helmet Detection Player", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# 5. Clean up resources
cap.release()
cv2.destroyAllWindows()