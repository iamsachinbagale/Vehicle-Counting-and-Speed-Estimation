import math
import cv2
import pandas as pd
from ultralytics import YOLO
import time
from math import dist
import datetime
from utils import Tracker

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Open video capture
cap = cv2.VideoCapture('vehVid2.mp4')

# Read class names from file
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Create a window
cv2.namedWindow('Video')

# Initialize Tracker
tracker = Tracker()

# Define line coordinates
cy1 = 222
cy2 = 300
offset = 6

# Initialize dictionaries for tracking vehicles and counters
vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Speed dictionary to store speeds of each vehicle
speed_dict = {}

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



# Initialize video writer
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_file = f'output_{current_date}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLO inference
    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    list = []

    # Extract relevant information about vehicles
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c or 'truck' in c or 'bicycle' in c or 'bus' in c or 'motorcycle' in c:
            list.append([x1, y1, x2, y2])

    # Update the tracker
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Check if the vehicle crossed the lower line
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            # Check if the vehicle crossed the upper line
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 20  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6

                    # Store speed in the dictionary
                    speed_dict[id] = a_speed_kh

                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    print(f"Vehicle {id}: Speed = {a_speed_kh:.2f} Km/h")

        # Check if the vehicle crossed the upper line
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            # Check if the vehicle crossed the lower line
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6

                    # Store speed in the dictionary
                    speed_dict[id] = a_speed_kh1

                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    print(f"Vehicle {id}: Speed = {a_speed_kh1:.2f} Km/h")

    # Draw lines on the frame
    cv2.line(frame, (177, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, ('L1'), (277, cy1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, ('L2'), (182, cy2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, 'Total Vehicles: {}'.format(len(list)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0
                                                                                                       ), 2)

    # Display the frame
    cv2.imshow("Video", frame)
    writer.write(frame)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and writer
cap.release()
writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print speeds along with vehicle IDs when the video ends
print("Speeds of Vehicles:")
for vehicle_id, speed in speed_dict.items():
    print(f"Vehicle {vehicle_id}: Speed = {speed:.2f} Km/h")
