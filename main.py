from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict

# ---- BEÁLLÍTÁSOK ----
VIDEO_PATH = "video2.mp4"
TARGET_WIDTH = 900
CONF_THRESHOLD = 0.45

VEHICLES = ["car", "truck", "bus", "motorbike"]

# Fix számlálóvonal pozíció (a kép 60%-ánál)
LINE_POS = 0.60

# Modell
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Hiba: nem sikerült megnyitni a videót!")
    exit()

# ---- SZÁMLÁLÁS ----
count_in = 0
count_out = 0
counted_ids = set()
track_history = defaultdict(lambda: deque(maxlen=20))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kép átméretezése
    h_o, w_o = frame.shape[:2]
    new_w = TARGET_WIDTH
    new_h = int(h_o * (new_w / w_o))
    frame = cv2.resize(frame, (new_w, new_h))

    line_y = int(new_h * LINE_POS)

    # YOLO + tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=CONF_THRESHOLD, verbose=False)

    active_ids = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls_id in zip(boxes, ids, classes):
            label = model.names[int(cls_id)]

            if label not in VEHICLES:
                continue

            active_ids.append(track_id)

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Előzmények tárolása
            track_history[track_id].append((cx, cy))

            # === SZÁMLÁLÁS LOGIKA ===
            if track_id not in counted_ids:
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2][1]

                    # Átlépte a számlálóvonalat?
                    crossed = (prev_y < line_y and cy >= line_y) or \
                              (prev_y > line_y and cy <= line_y)

                    if crossed:
                        counted_ids.add(track_id)

                        # Mozgás irányának meghatározása
                        if cy > prev_y:
                            count_in += 1     # lefelé
                        else:
                            count_out += 1    # felfelé

            # --- DOB0Z + ID ---
            color = (0, 255, 0) if track_id in counted_ids else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Nyomvonal kirajzolása ---
            pts = track_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

    # takarítás
    for tid in list(track_history.keys()):
        if tid not in active_ids:
            del track_history[tid]

    # --- FIX számlálóvonal ---
    cv2.line(frame, (0, line_y), (new_w, line_y), (255, 0, 255), 2)
    cv2.putText(frame, "Szamlalo vonal", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # --- UI panel ---
    cv2.rectangle(frame, (0, 0), (new_w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Be (Le): {count_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Ki (Fel): {count_out}", (new_w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Forgalom Szamlalo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---- ÖSSZEGZŐ KÉPKERET ----
summary_img = np.zeros((400, 700, 3), dtype="uint8")

total = count_in + count_out

cv2.putText(summary_img, "Vegeredmeny", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

cv2.putText(summary_img, f"Bejovo (Le): {count_in}", (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.putText(summary_img, f"Kimeno (Fel): {count_out}", (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

cv2.putText(summary_img, f"OSSZESEN: {total}", (30, 300),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

cv2.imshow("Osszegzes", summary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
