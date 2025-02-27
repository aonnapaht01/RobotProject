import torch
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('C:/Users/ACER/OneDrive - Chiang Mai University/Desktop/TrainTray/runs/detect/train2/weights/best.pt')  

cap = cv2.VideoCapture(1)

focal_length = 554 # Adjust Focal length based on calibration in pixel (F = MaxTransmission rate/(2 * tan(FOV/2))) 
actual_width = 8  # Actual width of DentalTray in cm (Vertical Top View)
#actual_length = 12 # Actual length of DentalTray in cm (Horizontal Top View) and Change actual_width to actual_length

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    object_count = 0  

    for result in results:
        for box in result.boxes:
            object_count += 1  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0].item()  
            cls = int(box.cls[0].item())  
            label = f"{model.names[cls]}: {conf:.2f}"  
            
            # Calculate the center coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            
            # Estimate distance (z) using the known width of the object
            distance_z = (focal_length * actual_width) / width 
            
            # Convert to real-world coordinates
            image_center_x = frame.shape[1] // 2
            image_center_y = frame.shape[0] // 2
            real_x = (center_x - image_center_x) * (distance_z / focal_length)
            real_y = (center_y - image_center_y) * (distance_z / focal_length)
            
            coordinate_text = f'({real_x:.2f}, {real_y:.2f}, {distance_z:.2f} cm)'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, coordinate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            print(f'Coordinates: X={real_x:.2f} cm, Y={real_y:.2f} cm, Z={distance_z:.2f} cm, Conf: {conf:.2f}, Class: {model.names[cls]}')
    
    cv2.putText(frame, f'DentalTray: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv11 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
