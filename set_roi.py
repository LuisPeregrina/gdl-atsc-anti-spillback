import cv2
import numpy as np

points = []
img = None

def click_event(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Select ROI", img)

        if len(points) == 4:
            print("ROI points:", points)
            cv2.destroyAllWindows()

cap = cv2.VideoCapture("samples/288312_tiny.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
success, frame = cap.read()
if not success:
    raise RuntimeError("Could not read a sample frame from the video.")

img = frame.copy()
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select ROI", click_event)
cv2.imshow("Select ROI", img)
cv2.waitKey(0)

if len(points) != 4:
    raise RuntimeError(f"Expected 4 points, got {len(points)}")

roi_pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
print("ROI points:", roi_pts)
