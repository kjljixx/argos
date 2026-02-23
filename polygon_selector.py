import cv2
import numpy as np
import sys

polygons = []
current_polygon = []
img = None
display = None

def redraw():
  global display
  display = img.copy()
  for poly in polygons:
    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    for p in poly:
      cv2.circle(display, p, 4, (0, 200, 0), -1)
  if current_polygon:
    pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(display, [pts], False, (0, 150, 255), 2)
    for p in current_polygon:
      cv2.circle(display, p, 4, (0, 100, 255), -1)
  cv2.imshow("Polygon Picker", display)

def mouse_cb(event, x, y, flags, param):
  global current_polygon
  if event == cv2.EVENT_LBUTTONDOWN:
    current_polygon.append((x, y))
    redraw()
  elif event == cv2.EVENT_RBUTTONDOWN:
    if len(current_polygon) >= 3:
      polygons.append(current_polygon)
      current_polygon = []
      redraw()

def load_frame(path):
  if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
      raise ValueError(f"Could not read video: {path}")
    return frame
  else:
    frame = cv2.imread(path)
    if frame is None:
      raise ValueError(f"Could not read image: {path}")
    return frame

PATH = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14.mp4"

img = load_frame(PATH)
img = cv2.resize(img, (640, 640))
cv2.namedWindow("Polygon Picker")
cv2.setMouseCallback("Polygon Picker", mouse_cb)
redraw()

print("Left click: add point | Right click: close polygon | U: undo point | C: clear current | Q/Enter: finish")

while True:
  key = cv2.waitKey(20) & 0xFF
  if key == ord('u'):
    if current_polygon:
      current_polygon.pop()
      redraw()
  elif key == ord('c'):
    current_polygon = []
    redraw()
  elif key in (ord('q'), 13):
    break

cv2.destroyAllWindows()

if current_polygon and len(current_polygon) >= 3:
  polygons.append(current_polygon)

print(f"\n{len(polygons)} polygon(s):\n")
for i, poly in enumerate(polygons):
  arr = np.array(poly, dtype=np.int32)
  print(f"{repr(arr)}\n")