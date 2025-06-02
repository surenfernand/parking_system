import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8l.pt')

cap = cv2.VideoCapture('videos/parking_lot2.mp4')

with open("model.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

  # Area Mark
areas = {
    22: [(2, 277), (2, 381), (69, 381), (69, 277)],
    23: [(75, 277), (75, 381), (134, 381), (134, 277)],
    24: [(138, 277), (138, 381), (193, 381), (193, 277)],
    25: [(196, 277), (196, 381), (256, 381), (256, 277)],
    26: [(263, 277), (263, 381), (319, 381), (319, 277)],
    27: [(327, 277), (327, 381), (382, 381), (382, 277)],
    28: [(390, 277), (390, 381), (445, 381), (445, 277)],
    29: [(450, 277), (450, 381), (507, 381), (507, 277)],
    30: [(515, 277), (515, 381), (567, 381), (567, 277)],
    31: [(576, 277), (576, 381), (632, 381), (632, 277)],
    32: [(2, 386), (2, 499), (69, 499), (69, 386)],
    33: [(72, 386), (72, 499), (130, 499), (130, 386)],
    34: [(136, 386), (136, 499), (185, 499), (185, 386)],
    35: [(196, 386), (196, 499), (247, 499), (247, 386)],
    36: [(260, 386), (260, 499), (309, 499), (309, 386)],
    37: [(323, 386), (323, 499), (373, 499), (373, 386)],
    38: [(383, 386), (383, 499), (435, 499), (435, 386)],
    39: [(448, 386), (448, 499), (503, 499), (503, 386)],
    40: [(513, 386), (513, 496), (566, 496), (566, 386)],
    41: [(575, 386), (575, 498), (631, 498), (631, 386)]
}

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

   
    area_detections = {k: [] for k in areas.keys()}

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        label = class_list[class_id]

        if label in ['car', 'cell phone', 'suitcase']:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for area_id, points in areas.items():
                if cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False) >= 0:
                    area_detections[area_id].append(label)
                    break


    for area_id, points in areas.items():
        color = (0, 255, 255) if len(area_detections[area_id]) >= 1 else (255, 255, 255)
        text_color = (0, 0, 255) if len(area_detections[area_id]) >= 1 else (255, 255, 255)
        cv2.polylines(frame, [np.array(points, np.int32)], True, color, 2)
      

  # Count Display
    total_spots = len(areas)
    available_spots = 0

    for area_id, points in areas.items():
        occupied = len(area_detections[area_id]) >= 1
        if not occupied:
            available_spots += 1
        color = (0, 255, 255) if occupied else (255, 255, 255)
        cv2.polylines(frame, [np.array(points, np.int32)], True, color, 2)


    text = f"Available Parking Spaces: {available_spots} / {total_spots}"
    cv2.rectangle(frame, (10, 10), (250, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Parking Manager", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
