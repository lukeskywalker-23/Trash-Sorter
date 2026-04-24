import cv2
from ultralytics import YOLO
import serial
import time 

ser = serial.Serial('COM3', 9600, timeout=1)  # need to Replace 'COM3' with Arduino's port
time.sleep(2)  # Wait for the serial connection to initialize

# Load model
model = YOLO(r"\Users\000000070\Downloads\best.pt")

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

    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv11 Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
