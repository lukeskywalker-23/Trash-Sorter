import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 inference on the frame
    # verbose=False prevents the console from being flooded with output for each frame
    results = model.predict(frame, verbose=False)

    # The 'results' object contains bounding boxes, classes, and confidence levels
    # You can process these results here (e.g., draw custom annotations using OpenCV functions)
    # The 'plot()' method from the results object can draw the detections on the frame automatically
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv11 Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
