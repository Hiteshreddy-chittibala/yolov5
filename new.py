import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load(
    '/home/hiteshreddy/yolov5',
    'custom',
    path='/home/hiteshreddy/yolov5/runs/train/exp21/weights/best.pt',
    source='local'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    frame = results.render()[0]

    cv2.imshow("Bottle Cap Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
