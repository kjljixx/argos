import cv2
import supervision as sv

video_path = r"C:\Users\kjlji\Videos\Captures\2025-2026 Season_ Bensalem Area Qualifier - YouTube — Zen Browser 2026-02-18 20-27-14.mp4"

generator = sv.get_video_frames_generator(video_path)
 
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b, g, r = img[y, x]
        cv2.putText(img, f"{b},{g},{r}", (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

img = None

for frame_index, frame in enumerate(generator):
  img = cv2.resize(frame, (640, 640))
  cv2.imshow("Frame", img)
  cv2.setMouseCallback('Frame', click_event)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  break


